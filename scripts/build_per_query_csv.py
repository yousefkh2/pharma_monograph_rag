#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-query CSV from run files and gold map")
    parser.add_argument("--gold", type=Path, default=Path("eval/datasets/tiny_gold_map.json"))
    parser.add_argument("--bm25", type=Path, default=Path("eval/runs/bm25.jsonl"))
    parser.add_argument("--bge", type=Path, default=Path("eval/runs/bge.jsonl"))
    parser.add_argument("--hybrid", type=Path, default=Path("eval/runs/hybrid.jsonl"))
    parser.add_argument("--titles", type=Path, default=Path("eval/runs/titles_subtitles.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("eval/runs/per_query.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("eval/runs/metrics.json"))
    parser.add_argument("--summary-csv", type=Path, default=Path("eval/runs/summary_for_chart.csv"))
    return parser.parse_args()


def load_gold(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_run(path: Path) -> Dict[str, dict]:
    by_qid: Dict[str, dict] = {}
    if not path.exists():
        return by_qid
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("qid")
            if qid:
                by_qid[qid] = rec
    return by_qid


def top1_doc(run_entry: Optional[dict]) -> str:
    if not run_entry:
        return ""
    results = run_entry.get("results") or []
    if not results:
        return ""
    return results[0].get("doc_id") or ""


def rank_of_first_gold(run_entry: Optional[dict], gold_ids: List[str]) -> Optional[int]:
    if not run_entry or not gold_ids:
        return None
    gold = set(gold_ids)
    rank = 0
    seen = set()
    for item in run_entry.get("results") or []:
        doc_id = item.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        rank += 1
        seen.add(doc_id)
        if doc_id in gold:
            return rank
    return None


def write_per_query_csv(
    out_path: Path,
    gold_map: Dict[str, List[str]],
    runs: Dict[str, Dict[str, dict]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "qid",
            "base_qid",
            "variant_type",
            "gold_ids",
            "bm25_top1",
            "bge_top1",
            "hybrid_top1",
            "titles_top1",
            "rank_bm25",
            "rank_bge",
            "rank_hybrid",
            "rank_titles",
        ])

        all_qids = sorted(gold_map.keys(), key=lambda x: (len(x), x))
        seen = set()
        for qid in sorted(set(list(runs["bm25"].keys()) + list(runs["bge"].keys()) + list(runs["hybrid"].keys()) + list(runs["titles"].keys()))):
            if qid in seen:
                continue
            seen.add(qid)
            gold_ids = gold_map.get(qid, [])
            bm25_e = runs["bm25"].get(qid)
            bge_e = runs["bge"].get(qid)
            hybrid_e = runs["hybrid"].get(qid)
            titles_e = runs["titles"].get(qid)

            base_qid = None
            variant_type = None
            for entry in (bm25_e, bge_e, hybrid_e, titles_e):
                if entry:
                    base_qid = entry.get("base_qid") or base_qid
                    variant_type = entry.get("variant_type") or variant_type
            if not gold_ids and base_qid and base_qid in gold_map:
                gold_ids = gold_map[base_qid]
            elif not gold_ids and isinstance(qid, str) and "_" in qid:
                prefix = qid.split("_", 1)[0]
                if prefix in gold_map:
                    gold_ids = gold_map[prefix]

            row = [
                qid,
                base_qid or "",
                variant_type or "",
                ";".join(gold_ids),
                top1_doc(bm25_e),
                top1_doc(bge_e),
                top1_doc(hybrid_e),
                top1_doc(titles_e),
                rank_of_first_gold(bm25_e, gold_ids) or "",
                rank_of_first_gold(bge_e, gold_ids) or "",
                rank_of_first_gold(hybrid_e, gold_ids) or "",
                rank_of_first_gold(titles_e, gold_ids) or "",
            ]
            writer.writerow(row)


def write_summary_csv(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.exists():
        return
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    rows = []
    mapping = {
        "bm25.jsonl": "bm25",
        "bge.jsonl": "bge",
        "hybrid.jsonl": "hybrid",
        "titles_subtitles.jsonl": "titles_subtitles",
    }
    for fname, name in mapping.items():
        m = metrics.get(fname)
        if not m:
            continue
        rows.append({
            "retriever": name,
            "hit_at_1": m.get("hit_at_1", 0.0),
            "mrr": m.get("mrr", 0.0),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["retriever", "hit_at_1", "mrr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    gold_map = load_gold(args.gold)
    runs = {
        "bm25": load_run(args.bm25),
        "bge": load_run(args.bge),
        "hybrid": load_run(args.hybrid),
        "titles": load_run(args.titles),
    }
    write_per_query_csv(args.out_csv, gold_map, runs)
    write_summary_csv(args.metrics, args.summary_csv)
    print(f"per-query CSV saved to {args.out_csv}")
    if args.summary_csv:
        print(f"summary CSV for chart saved to {args.summary_csv}")


if __name__ == "__main__":
    main()
