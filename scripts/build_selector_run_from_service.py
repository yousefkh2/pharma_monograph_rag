#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import httpx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a monograph-level run from the LLM selector service (/rerank)")
    p.add_argument("--service-url", default="http://localhost:8001", help="Base URL for hybrid_service")
    p.add_argument("--queries", type=Path, default=Path("eval/datasets/queries.jsonl"))
    p.add_argument("--output", type=Path, default=Path("eval/runs_selector/selector.jsonl"))
    p.add_argument("--top-k", type=int, default=20, help="Top chunks to request from service for rerank (service caps at 20)")
    p.add_argument("--dense-weight", type=float, default=0.6)
    p.add_argument("--bm25-candidates", type=int, default=200)
    p.add_argument("--dense-candidates", type=int, default=50)
    p.add_argument("--timeout", type=float, default=60.0)
    return p.parse_args()


def load_queries(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        out: List[Dict[str, object]] = []
        for entry in data:
            base_qid = entry.get("qid") or entry.get("id")
            variants = entry.get("variants") or []
            if variants:
                for idx, v in enumerate(variants, start=1):
                    q = v.get("query") or v.get("question") or v.get("text") or ""
                    if not q:
                        continue
                    vt = v.get("type") or str(idx)
                    v_qid = v.get("qid") or (f"{base_qid}_{vt}" if base_qid else vt)
                    out.append({"qid": v_qid, "query": q, "base_qid": base_qid, "variant_type": v.get("type")})
            else:
                q = entry.get("query") or entry.get("question") or entry.get("text") or ""
                if q:
                    out.append({"qid": base_qid, "query": q, "base_qid": base_qid})
        return out
    # JSONL
    out: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def aggregate_monograph_scores(chunks: List[Dict[str, object]]) -> Dict[str, float]:
    per_doc: Dict[str, List[float]] = defaultdict(list)
    for hit in chunks:
        doc_id = hit.get("doc_id")
        if not doc_id:
            continue
        # use the combined score returned by the service
        score = hit.get("score")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        per_doc[doc_id].append(score_f)
    monograph_scores: Dict[str, float] = {}
    for doc_id, scores in per_doc.items():
        scores.sort(reverse=True)
        top3 = scores[:3]
        monograph_scores[doc_id] = mean(top3) if top3 else 0.0
    return monograph_scores


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries)

    client = httpx.Client(base_url=args.service_url, timeout=args.timeout)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for entry in queries:
            qid = entry.get("qid")
            query_text = entry.get("query")
            base_qid = entry.get("base_qid")
            variant_type = entry.get("variant_type")
            if not qid or not query_text:
                continue

            top_k = min(args.top_k, 20)
            try:
                resp = client.post("/rerank", json={
                    "query": query_text,
                    "top_k": top_k,
                    "dense_weight": args.dense_weight,
                    "bm25_candidates": args.bm25_candidates,
                    "dense_candidates": args.dense_candidates,
                })
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                # Write an empty result to keep alignment
                payload = {"qid": qid, "base_qid": base_qid, "variant_type": variant_type, "query": query_text, "results": []}
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                continue

            selection = data.get("selection", {})
            original_hits = data.get("original_results", [])

            mono_scores = aggregate_monograph_scores(original_hits)
            # Build results by monograph score desc
            ranked_docs = sorted(mono_scores.items(), key=lambda kv: kv[1], reverse=True)
            results: List[Dict[str, object]] = []
            for doc_id, score in ranked_docs:
                results.append({"doc_id": doc_id, "title": "", "score": score})

            # Force the LLM-selected monograph (if any) to rank 1, retaining its score if already present
            sel_doc = selection.get("doc_id")
            if sel_doc:
                existing = None
                for item in results:
                    if item.get("doc_id") == sel_doc:
                        existing = item
                        break
                if existing is None:
                    existing = {"doc_id": sel_doc, "title": "", "score": float("inf")}
                else:
                    results.remove(existing)
                results.insert(0, existing)

            payload = {
                "qid": qid,
                "base_qid": base_qid,
                "variant_type": variant_type,
                "query": query_text,
                "results": results,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    client.close()
    print(f"Selector run saved to {out_path}")


if __name__ == "__main__":
    main()
