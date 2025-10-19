#!/usr/bin/env python3
"""Build a frozen candidate pool combining BM25 and dense (BGE) monograph scores."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import BM25Index, default_tokenize
from retrieval.search_utils import BM25SearchEngine, load_metadata
from nano_vectordb import NanoVectorDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 + BGE candidate pool")
    parser.add_argument("--queries", type=Path, default=Path("eval/datasets/queries.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("vector_store/chunk_metadata.jsonl"))
    parser.add_argument("--bm25-index", type=Path, default=Path("vector_store/bm25_index.pkl"))
    parser.add_argument("--vectordb", type=Path, default=Path("vector_store/nano_chunks.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/datasets/candidate_pool.jsonl"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def load_queries(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # JSON array format (e.g., perturbed variants)
    if text.lstrip().startswith("["):
        data = json.loads(text)
        queries: List[Dict[str, object]] = []
        for entry in data:
            base_qid = entry.get("qid") or entry.get("id")
            variants = entry.get("variants") or []
            if variants:
                for idx, variant in enumerate(variants, start=1):
                    variant_query = variant.get("query") or variant.get("question") or variant.get("text") or ""
                    if not variant_query:
                        continue
                    variant_label = variant.get("type") or str(idx)
                    variant_qid = variant.get("qid") or (f"{base_qid}_{variant_label}" if base_qid else variant_label)
                    queries.append(
                        {
                            "qid": variant_qid,
                            "query": variant_query,
                            "base_qid": base_qid,
                            "variant_type": variant.get("type"),
                        }
                    )
            else:
                query_text = entry.get("query") or entry.get("question") or entry.get("text") or ""
                if query_text:
                    queries.append({
                        "qid": base_qid,
                        "query": query_text,
                        "base_qid": base_qid,
                    })
        return queries

    # Default JSONL (one object per line)
    queries: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        queries.append(rec)
    return queries


def make_snippet(text: str, limit: int = 220) -> str:
    snippet = text.replace("\n", " ").strip()
    if len(snippet) > limit:
        snippet = snippet[:limit].rstrip() + "…"
    return snippet


def embed_query(query: str, *, client: httpx.Client, model: str) -> np.ndarray:
    response = client.post("/api/embeddings", json={"model": model, "prompt": query})
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError("Embedding response missing 'embedding'")
    return np.asarray(embedding, dtype=np.float32)


def summarize_hits(hits: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    per_doc: Dict[str, List[tuple[float, Dict[str, object]]]] = defaultdict(list)
    for hit in hits:
        doc_id = hit.get("doc_id")
        if not doc_id:
            continue
        score_raw = hit.get("score")
        if score_raw is None:
            score_raw = hit.get("__metrics__")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            continue
        if score != score:  # NaN guard
            continue
        per_doc[doc_id].append((score, hit))

    summary: Dict[str, Dict[str, object]] = {}
    for doc_id, scored_hits in per_doc.items():
        scored_hits.sort(key=lambda pair: pair[0], reverse=True)
        scores = [score for score, _ in scored_hits]
        top3 = scores[:3]
        top3_mean = mean(top3) if top3 else 0.0
        score_max = scores[0] if scores else 0.0
        best_chunk = scored_hits[0][1]
        summary[doc_id] = {
            "top3_mean": top3_mean,
            "max": score_max,
            "best_chunk_id": best_chunk.get("chunk_id") or best_chunk.get("__id__"),
        }
    return summary


def main() -> None:
    args = parse_args()

    queries = load_queries(args.queries)
    metadata_records = load_metadata(args.metadata)
    if not metadata_records:
        raise RuntimeError(f"No metadata records found at {args.metadata}")

    chunk_map: Dict[str, Dict[str, object]] = {}
    doc_meta: Dict[str, Dict[str, object]] = {}
    embedding_dim: Optional[int] = None
    for record in metadata_records:
        chunk_id = record.get("chunk_id")
        if chunk_id:
            chunk_map[chunk_id] = record
        doc_id = record.get("doc_id")
        if doc_id and doc_id not in doc_meta:
            doc_meta[doc_id] = record
        if embedding_dim is None and record.get("embedding_dim"):
            embedding_dim = int(record["embedding_dim"])
    if embedding_dim is None:
        raise RuntimeError("Unable to infer embedding dimension from metadata")

    bm25_index = BM25Index.load(args.bm25_index, tokenize=default_tokenize)
    bm25_engine = BM25SearchEngine(bm25_index, metadata_records)

    vectordb = NanoVectorDB(embedding_dim, storage_file=str(args.vectordb))
    vectordb.pre_process()

    client = httpx.Client(base_url=args.ollama_host, timeout=args.timeout)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for entry in queries:
            qid = entry.get("qid") or entry.get("id")
            base_qid = entry.get("base_qid")
            variant_type = entry.get("variant_type")
            query_text = entry.get("query") or entry.get("question") or entry.get("text")
            if not qid or not query_text:
                continue

            bm25_hits, _ = bm25_engine.search(
                query_text,
                top_k=args.top_k,
                retrieve_k=args.top_k,
                include_text=False,
            )

            try:
                embedding = embed_query(query_text, client=client, model=args.embed_model)
                dense_results = vectordb.query(embedding, top_k=args.top_k)
            except Exception as exc:
                dense_results = []
                print(f"[WARN] Dense retrieval failed for {qid}: {exc}")

            dense_hits: List[Dict[str, object]] = []
            for item in dense_results:
                chunk_id = item.get("__id__")
                dense_hits.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": item.get("doc_id"),
                        "score": item.get("__metrics__", 0.0),
                    }
                )

            bm25_summary = summarize_hits(bm25_hits)
            dense_summary = summarize_hits(dense_hits)

            candidate_doc_ids = set(bm25_summary.keys()) | set(dense_summary.keys())

            candidates: List[Dict[str, object]] = []
            for doc_id in candidate_doc_ids:
                meta = doc_meta.get(doc_id, {})
                title = meta.get("drug_title") or meta.get("doc_id") or doc_id

                snippet_chunk_id = bm25_summary.get(doc_id, {}).get("best_chunk_id")
                if not snippet_chunk_id:
                    snippet_chunk_id = dense_summary.get(doc_id, {}).get("best_chunk_id")

                snippet = ""
                if snippet_chunk_id:
                    chunk_record = chunk_map.get(snippet_chunk_id)
                    if chunk_record:
                        snippet = make_snippet(chunk_record.get("text", ""))

                candidate = {
                    "doc_id": doc_id,
                    "bm25_top3": bm25_summary.get(doc_id, {}).get("top3_mean"),
                    "bm25_max": bm25_summary.get(doc_id, {}).get("max"),
                    "bge_top3": dense_summary.get(doc_id, {}).get("top3_mean"),
                    "bge_max": dense_summary.get(doc_id, {}).get("max"),
                    "title": title,
                    "rep_snippet": snippet,
                }
                candidates.append(candidate)

            def sort_key(item: Dict[str, object]) -> float:
                scores = [score for score in (
                    item.get("bm25_top3"),
                    item.get("bge_top3"),
                    item.get("bm25_max"),
                    item.get("bge_max"),
                ) if isinstance(score, (int, float))]
                return max(scores) if scores else float("-inf")

            candidates.sort(key=sort_key, reverse=True)

            payload = {
                "qid": qid,
                "query": query_text,
                "base_qid": base_qid,
                "variant_type": variant_type,
                "candidates": candidates,
            }
            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    client.close()
    print(f"Candidate pool saved to {output_path}")


if __name__ == "__main__":
    main()
