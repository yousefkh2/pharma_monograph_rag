#!/usr/bin/env python3
"""Build dense vector index for monograph chunks using Ollama embeddings.

This script skips the LightRAG knowledge-graph pipeline and focuses solely on
writing dense vectors + chunk metadata for downstream retrieval. It expects the
parsed corpus produced by ``ingest_sections.py`` and an Ollama server hosting
``bge-m3`` (or another embedding model with the same shape).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import httpx
import numpy as np
from nano_vectordb import NanoVectorDB

DEFAULT_BATCH_SIZE = 32
DEFAULT_CONCURRENCY = 4
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0
EMBEDDING_ENDPOINT = "/api/embeddings"


@dataclass(slots=True)
class ChunkRecord:
    """Represents a chunk ready for embedding."""

    chunk_id: str
    text: str
    doc_id: str
    section_id: str | None
    section_title: str | None
    section_code: str | None
    chunk_index: int | None
    chunk_type: str
    source_url: str | None
    drug_title: str | None


def iter_chunk_records(json_path: Path) -> Iterator[ChunkRecord]:
    """Yield chunk records from the parsed monograph JSON file."""

    with json_path.open("r", encoding="utf-8") as handle:
        docs = json.load(handle)

    for doc in docs:
        doc_id = doc.get("doc_id") or "unknown_doc"
        source_url = doc.get("source_url")
        drug_title = doc.get("drug_title")

        intro = doc.get("intro") or {}
        intro_text = (intro.get("text") or "").strip()
        if intro_text:
            yield ChunkRecord(
                chunk_id=f"{doc_id}::intro",
                text=intro_text,
                doc_id=doc_id,
                section_id="intro",
                section_title="Introduction",
                section_code=None,
                chunk_index=None,
                chunk_type="intro",
                source_url=source_url,
                drug_title=drug_title,
            )

        for section in doc.get("sections", []):
            section_id = section.get("section_id") or "unknown_section"
            section_title = section.get("section_title")
            section_code = section.get("section_code")
            for chunk in section.get("chunks", []):
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                chunk_idx_raw = chunk.get("chunk_index")
                try:
                    chunk_index = int(chunk_idx_raw) if chunk_idx_raw is not None else None
                except (TypeError, ValueError):
                    chunk_index = None
                chunk_id = f"{doc_id}::{section_id}::{chunk_idx_raw}" if chunk_idx_raw is not None else f"{doc_id}::{section_id}"
                yield ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    doc_id=doc_id,
                    section_id=section_id,
                    section_title=section_title,
                    section_code=section_code,
                    chunk_index=chunk_index,
                    chunk_type="section",
                    source_url=source_url,
                    drug_title=drug_title,
                )


def chunked(iterable: Sequence[ChunkRecord], size: int) -> Iterable[Sequence[ChunkRecord]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def load_processed_ids(metadata_path: Path) -> set[str]:
    processed: set[str] = set()
    if not metadata_path.exists():
        return processed
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = data.get("chunk_id")
            if chunk_id:
                processed.add(chunk_id)
    return processed


async def ensure_model_available(client: httpx.AsyncClient, model: str) -> None:
    resp = await client.get("/api/tags")
    resp.raise_for_status()
    payload = resp.json()
    available = {item.get("name") for item in payload.get("models", [])}
    if model not in available:
        raise RuntimeError(
            f"Embedding model '{model}' is not available in Ollama. Run 'ollama pull {model}' "
            "and ensure the Ollama service is running."
        )


async def embed_text(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    text: str,
    retries: int,
    backoff: float,
) -> list[float]:
    async with semaphore:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await client.post(
                    EMBEDDING_ENDPOINT,
                    json={"model": model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if not embedding:
                    raise RuntimeError("Missing 'embedding' in Ollama response")
                return embedding
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                await asyncio.sleep(backoff * attempt)
        assert last_error is not None
        raise RuntimeError("Failed to obtain embedding after retries") from last_error


async def embed_batch(
    client: httpx.AsyncClient,
    model: str,
    batch: Sequence[ChunkRecord],
    concurrency: int,
    retries: int,
    backoff: float,
) -> list[list[float]]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(embed_text(client, semaphore, model, record.text, retries, backoff))
        for record in batch
    ]
    return await asyncio.gather(*tasks)


def build_metadata_dict(record: ChunkRecord, embedding_dim: int) -> dict:
    return {
        "chunk_id": record.chunk_id,
        "doc_id": record.doc_id,
        "section_id": record.section_id,
        "section_title": record.section_title,
        "section_code": record.section_code,
        "chunk_index": record.chunk_index,
        "chunk_type": record.chunk_type,
        "source_url": record.source_url,
        "drug_title": record.drug_title,
        "embedding_dim": embedding_dim,
        "text": record.text,
    }


async def run_pipeline(args: argparse.Namespace) -> None:
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "chunk_metadata.jsonl"
    vectordb_path = output_dir / "nano_chunks.json"

    all_chunks = list(iter_chunk_records(json_path))
    processed_ids = load_processed_ids(metadata_path)
    pending_chunks = [chunk for chunk in all_chunks if chunk.chunk_id not in processed_ids]

    if not pending_chunks:
        print("No new chunks to embed. Existing metadata covers all records.")
        return

    embedding_model = args.model
    timeout = httpx.Timeout(args.request_timeout, connect=args.request_timeout)
    async with httpx.AsyncClient(base_url=args.ollama_host, timeout=timeout) as client:
        await ensure_model_available(client, embedding_model)

        vectordb = NanoVectorDB(args.embedding_dim, storage_file=str(vectordb_path))
        total_existing = len(processed_ids)
        start_time = time.time()

        with metadata_path.open("a", encoding="utf-8") as meta_fp:
            processed = 0
            for batch in chunked(pending_chunks, args.batch_size):
                embeddings = await embed_batch(
                    client,
                    embedding_model,
                    batch,
                    concurrency=args.concurrency,
                    retries=args.retries,
                    backoff=args.retry_backoff,
                )

                data_points = []
                metadata_lines = []
                for record, vector in zip(batch, embeddings):
                    np_vector = np.asarray(vector, dtype=np.float32)
                    if np_vector.shape[0] != args.embedding_dim:
                        raise ValueError(
                            f"Embedding dimension mismatch for {record.chunk_id}: "
                            f"expected {args.embedding_dim}, got {np_vector.shape[0]}"
                        )
                    data_points.append(
                        {
                            "__id__": record.chunk_id,
                            "__vector__": np_vector,
                            "doc_id": record.doc_id,
                            "section_id": record.section_id,
                            "section_title": record.section_title,
                            "section_code": record.section_code,
                            "chunk_index": record.chunk_index,
                            "chunk_type": record.chunk_type,
                            "source_url": record.source_url,
                            "drug_title": record.drug_title,
                        }
                    )
                    metadata_lines.append(build_metadata_dict(record, args.embedding_dim))

                vectordb.upsert(data_points)
                vectordb.store_additional_data(
                    model=embedding_model,
                    embedding_dim=args.embedding_dim,
                    total_records=len(vectordb),
                    metadata_file=str(metadata_path),
                    source_json=str(json_path),
                )
                vectordb.save()

                for line in metadata_lines:
                    meta_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
                meta_fp.flush()

                processed += len(batch)
                elapsed = time.time() - start_time
                overall = total_existing + processed
                print(
                    f"Embedded {processed}/{len(pending_chunks)} new chunks "
                    f"({overall} total). Elapsed {elapsed:.1f}s",
                    flush=True,
                )

    print("Embedding pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense embedding index")
    parser.add_argument(
        "json_path",
        nargs="?",
        default="parsed_output/all.json",
        help="Path to parsed monograph JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="vector_store",
        help="Directory where the vector DB and metadata will be stored",
    )
    parser.add_argument(
        "--model",
        default="bge-m3",
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        help="Expected embedding dimensionality",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to embed per batch (controls save cadence)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Concurrent embedding requests to Ollama",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout for Ollama embedding requests (seconds)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of retries per chunk on transient Ollama errors",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Base backoff (seconds) between retries (multiplied by attempt)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Base URL for the Ollama server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        print("\nEmbedding interrupted by user", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
