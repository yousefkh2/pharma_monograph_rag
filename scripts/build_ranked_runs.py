#!/usr/bin/env python3
"""Generate retrieval run files from frozen candidate pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.search_utils import default_tokenize, load_metadata, build_doc_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ranked monograph runs from candidate pool")
    parser.add_argument("--candidate-pool", type=Path, default=Path("eval/datasets/candidate_pool.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("vector_store/chunk_metadata.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/runs"))
    parser.add_argument("--titles-weight", type=float, default=1.0, help="Scale factor for titles+subtitles score")
    return parser.parse_args()


def load_candidate_pool(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_z_scores(values: List[Optional[float]]) -> List[float]:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return [0.0 for _ in values]
    mean_val = sum(numeric) / len(numeric)
    variance = sum((v - mean_val) ** 2 for v in numeric) / len(numeric)
    std = variance ** 0.5
    if std <= 1e-9:
        return [0.0 for _ in values]
    result: List[float] = []
    for v in values:
        if v is None:
            result.append(0.0)
        else:
            result.append((v - mean_val) / std)
    return result


def prepare_doc_tokens(metadata: List[dict]) -> Dict[str, set]:
    doc_index = build_doc_index(metadata)
    tokens_map: Dict[str, set] = {}
    for doc_id, info in doc_index.items():
        tokens = set(info.canonical_tokens)
        tokens.update(info.synonyms)
        tokens_map[doc_id] = tokens
    for record in metadata:
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        section_title = record.get("section_title")
        if section_title:
            tokens_map.setdefault(doc_id, set()).update(default_tokenize(section_title))
    return tokens_map


def titles_score(query_tokens: set, doc_tokens: set, weight: float) -> float:
    if not doc_tokens:
        return 0.0
    overlap = query_tokens.intersection(doc_tokens)
    if not overlap:
        return 0.0
    raw = len(overlap)
    bonus = sum(1.0 for token in overlap if len(token) > 6)
    return weight * (raw + 0.25 * bonus)


def trim_results(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    trimmed: List[Dict[str, object]] = []
    for item in results:
        trimmed.append({
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "score": item.get("score", 0.0),
        })
    return trimmed


def main() -> None:
    args = parse_args()
    candidate_pool = load_candidate_pool(args.candidate_pool)
    metadata = load_metadata(args.metadata)
    doc_tokens_map = prepare_doc_tokens(metadata)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "bm25": (output_dir / "bm25.jsonl").open("w", encoding="utf-8"),
        "bge": (output_dir / "bge.jsonl").open("w", encoding="utf-8"),
        "hybrid": (output_dir / "hybrid.jsonl").open("w", encoding="utf-8"),
        "titles": (output_dir / "titles_subtitles.jsonl").open("w", encoding="utf-8"),
    }

    try:
        for record in candidate_pool:
            qid = record.get("qid")
            base_qid = record.get("base_qid") or (qid.split("_", 1)[0] if isinstance(qid, str) and "_" in qid else qid)
            variant_type = record.get("variant_type")
            query = record.get("query")
            candidates = record.get("candidates") or []

            bm25_vals = [safe_float(item.get("bm25_top3")) for item in candidates]
            bge_vals = [safe_float(item.get("bge_top3")) for item in candidates]
            bm25_z = compute_z_scores(bm25_vals)
            bge_z = compute_z_scores(bge_vals)

            bm25_results: List[Dict[str, object]] = []
            bge_results: List[Dict[str, object]] = []
            hybrid_results: List[Dict[str, object]] = []
            titles_results: List[Dict[str, object]] = []

            query_tokens = set(default_tokenize(query or ""))

            for idx, item in enumerate(candidates):
                doc_id = item.get("doc_id")
                title = item.get("title")

                bm25_score = bm25_vals[idx]
                bm25_results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "score": bm25_score if bm25_score is not None else 0.0,
                    "_sort": bm25_score if bm25_score is not None else float("-inf"),
                })

                bge_score = bge_vals[idx]
                bge_results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "score": bge_score if bge_score is not None else 0.0,
                    "_sort": bge_score if bge_score is not None else float("-inf"),
                })

                hybrid_score = 0.0
                if bm25_vals[idx] is not None:
                    hybrid_score += 0.5 * bm25_z[idx]
                if bge_vals[idx] is not None:
                    hybrid_score += 0.5 * bge_z[idx]
                hybrid_results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "score": hybrid_score,
                })

                doc_tokens = doc_tokens_map.get(doc_id, set())
                titles_results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "score": titles_score(query_tokens, doc_tokens, args.titles_weight),
                })

            bm25_results.sort(key=lambda x: x.get("_sort", float("-inf")), reverse=True)
            bge_results.sort(key=lambda x: x.get("_sort", float("-inf")), reverse=True)
            hybrid_results.sort(key=lambda x: x["score"], reverse=True)
            titles_results.sort(key=lambda x: x["score"], reverse=True)

            for lst in (bm25_results, bge_results):
                for item in lst:
                    item.pop("_sort", None)

            payload_base = {
                "qid": qid,
                "base_qid": base_qid,
                "variant_type": variant_type,
                "query": query,
            }
            payloads = {
                "bm25": {**payload_base, "results": trim_results(bm25_results)},
                "bge": {**payload_base, "results": trim_results(bge_results)},
                "hybrid": {**payload_base, "results": trim_results(hybrid_results)},
                "titles": {**payload_base, "results": trim_results(titles_results)},
            }

            for key, fh in files.items():
                fh.write(json.dumps(payloads[key], ensure_ascii=False) + "\n")
    finally:
        for fh in files.values():
            fh.close()

    print(f"Runs saved to {output_dir}")


if __name__ == "__main__":
    main()
