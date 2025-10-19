#!/usr/bin/env python3
"""Compute retrieval metrics (monograph-level) for ranked run files.

Inputs
- Gold map JSON: maps qid -> [acceptable doc_id, ...]
- One or more run files in JSONL: each line
    {"qid": "Q1", "query": "...", "results": [{"doc_id": "...", "title": "...", "score": ...}, ...]}

Outputs
- Prints metrics per run to stdout
- Writes eval/runs/metrics.json with a dict of run_name -> metrics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked monograph runs against a gold map")
    parser.add_argument("--gold-map", type=Path, default=Path("eval/datasets/tiny_gold_map.json"))
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=[
            Path("eval/runs/bm25.jsonl"),
            Path("eval/runs/bge.jsonl"),
            Path("eval/runs/hybrid.jsonl"),
            Path("eval/runs/titles_subtitles.jsonl"),
        ],
        help="Run files to score (JSONL)",
    )
    parser.add_argument("--output", type=Path, default=Path("eval/runs/metrics.json"))
    return parser.parse_args()


def load_gold(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_run(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def first_gold_rank(results: List[dict], gold_doc_ids: List[str]) -> Optional[int]:
    if not results or not gold_doc_ids:
        return None
    gold = set(gold_doc_ids)
    seen = set()
    rank = 0
    for item in results:
        doc_id = item.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        rank += 1
        seen.add(doc_id)
        if doc_id in gold:
            return rank
    return None


def get_gold_candidates(entry: dict, gold_map: Dict[str, List[str]]) -> List[str]:
    qid = entry.get("qid")
    base_qid = entry.get("base_qid")
    if base_qid and base_qid in gold_map:
        return gold_map[base_qid]
    if qid in gold_map:
        return gold_map[qid]
    if isinstance(qid, str) and "_" in qid:
        prefix = qid.split("_", 1)[0]
        if prefix in gold_map:
            return gold_map[prefix]
    return []


def compute_run_metrics(path: Path, gold_map: Dict[str, List[str]]) -> Dict[str, float]:
    total = 0
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    cov_at_10 = 0
    rr_sum = 0.0
    ranks_found: List[int] = []

    for entry in iter_run(path):
        qid = entry.get("qid")
        results = entry.get("results") or []
        if not qid or not isinstance(results, list):
            continue
        gold = get_gold_candidates(entry, gold_map)
        total += 1

        rank = first_gold_rank(results, gold)
        if rank is not None:
            if rank == 1:
                hit_at_1 += 1
            if rank <= 3:
                hit_at_3 += 1
            if rank <= 5:
                hit_at_5 += 1
            if rank <= 10:
                cov_at_10 += 1
            rr_sum += 1.0 / rank
            ranks_found.append(rank)
        else:
            # MRR contribution is 0.0 by definition when missing
            pass

    def ratio(x: int) -> float:
        return (x / total) if total else 0.0

    mrr = (rr_sum / total) if total else 0.0
    avg_rank_found = (sum(ranks_found) / len(ranks_found)) if ranks_found else 0.0

    return {
        "queries": float(total),
        "hit_at_1": ratio(hit_at_1),
        "hit_at_3": ratio(hit_at_3),
        "hit_at_5": ratio(hit_at_5),
        "mrr": mrr,
        "avg_rank_found": avg_rank_found,
        "coverage_at_10": ratio(cov_at_10),
    }


def main() -> None:
    args = parse_args()
    gold_map = load_gold(args.gold_map)

    metrics_by_run: Dict[str, Dict[str, float]] = {}
    for run_path in args.runs:
        if not run_path.exists():
            print(f"[WARN] Missing run file: {run_path}")
            continue
        metrics = compute_run_metrics(run_path, gold_map)
        metrics_by_run[run_path.name] = metrics
        print(f"\n{run_path.name}")
        print(f"  queries         : {int(metrics['queries'])}")
        print(f"  Hit@1/3/5      : {metrics['hit_at_1']:.3f} / {metrics['hit_at_3']:.3f} / {metrics['hit_at_5']:.3f}")
        print(f"  MRR             : {metrics['mrr']:.3f}")
        print(f"  Avg rank (found): {metrics['avg_rank_found']:.2f}")
        print(f"  Coverage@10     : {metrics['coverage_at_10']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(metrics_by_run, fh, indent=2)
    print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
