#!/usr/bin/env python3
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

from retrieval.search_utils import load_metadata
from nano_vectordb import NanoVectorDB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect raw (non-normalized) dense scores for a query")
    p.add_argument("--query", help="Free-text query")
    p.add_argument("--qid", help="If set, load query text from eval/datasets/queries.jsonl by qid")
    p.add_argument("--queries", type=Path, default=Path("eval/datasets/queries.jsonl"))
    p.add_argument("--metadata", type=Path, default=Path("vector_store/chunk_metadata.jsonl"))
    p.add_argument("--vectordb", type=Path, default=Path("vector_store/nano_chunks.json"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def load_query_text(args: argparse.Namespace) -> str:
    if args.query:
        return args.query
    if not args.qid:
        raise SystemExit("Provide --query or --qid")
    # read from queries.jsonl
    with args.queries.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("qid") == args.qid:
                return rec.get("query") or rec.get("question") or ""
    raise SystemExit(f"qid {args.qid} not found in {args.queries}")


def embed_query(query: str, *, client: httpx.Client, model: str) -> np.ndarray:
    resp = client.post("/api/embeddings", json={"model": model, "prompt": query})
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("embedding")
    if not emb:
        raise RuntimeError("Embedding response missing 'embedding'")
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    args = parse_args()
    query = load_query_text(args).strip()
    if not query:
        raise SystemExit("Empty query")

    # Load metadata to map chunk_id -> human fields and infer embedding dim
    metadata = load_metadata(args.metadata)
    chunk_map: Dict[str, dict] = {}
    embed_dim: Optional[int] = None
    for rec in metadata:
        if not embed_dim and rec.get("embedding_dim"):
            embed_dim = int(rec["embedding_dim"])  # type: ignore[arg-type]
        cid = rec.get("chunk_id")
        if cid:
            chunk_map[cid] = rec
    if embed_dim is None:
        raise RuntimeError("Could not infer embedding_dim from metadata")

    vectordb = NanoVectorDB(embed_dim, storage_file=str(args.vectordb))
    vectordb.pre_process()
    client = httpx.Client(base_url=args.ollama_host, timeout=args.timeout)

    try:
        emb = embed_query(query, client=client, model=args.embed_model)
        dense_hits = vectordb.query(emb, top_k=args.top_k)
    finally:
        client.close()

    print(f"\nQuery: {query}")
    print(f"Top {args.top_k} raw dense chunk hits (no normalization):\n")
    per_doc: Dict[str, List[float]] = defaultdict(list)
    for i, hit in enumerate(dense_hits, start=1):
        cid = hit.get("__id__")
        score = float(hit.get("__metrics__", 0.0) or 0.0)
        doc_id = hit.get("doc_id") or (chunk_map.get(cid, {}).get("doc_id") if cid else None)
        rec = chunk_map.get(cid, {})
        title = rec.get("drug_title") or doc_id or "?"
        section = rec.get("section_title") or ""
        per_doc[doc_id].append(score)
        print(f"#{i:02d} score={score:.6f}  doc={doc_id}  section={section}  title={title}")

    print("\nPer-monograph summary (from these hits):")
    rows = []
    for doc_id, scores in per_doc.items():
        if not scores:
            continue
        scores_sorted = sorted(scores, reverse=True)
        top3 = scores_sorted[:3]
        rows.append((mean(top3), max(scores_sorted), doc_id))
    rows.sort(reverse=True)
    for top3_mean, max_val, doc_id in rows[:10]:
        t = chunk_map.get(next((cid for cid, r in chunk_map.items() if r.get("doc_id") == doc_id), None), {}).get("drug_title") or doc_id
        print(f"doc={doc_id}  top3_mean={top3_mean:.6f}  max={max_val:.6f}  title={t}")


if __name__ == "__main__":
    main()

