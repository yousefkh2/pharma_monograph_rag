#!/usr/bin/env python3
"""Summarize judge outputs across multiple models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_judge_file(path: Path) -> Dict:
    data = json.loads(path.read_text())
    if "summary" not in data:
        raise ValueError(f"Judge file {path} missing 'summary'")
    return data


def summarize(files: List[Path]) -> None:
    rows = []
    for file in files:
        data = load_judge_file(file)
        summary = data["summary"]
        results = data.get("results", [])
        verdict_counts = {"pass": 0, "manual_review": 0, "fail": 0}
        for item in results:
            verdict = (item.get("verdict") or "").lower()
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1
        model_name = file.stem.replace("_judge", "")
        rows.append({
            "model": model_name,
            "count": summary.get("count", len(results)),
            "mean": summary.get("mean_score", 0),
            "median": summary.get("median_score", 0),
            "pass": verdict_counts["pass"],
            "manual_review": verdict_counts["manual_review"],
            "fail": verdict_counts["fail"],
            "path": str(file),
        })

    if not rows:
        print("No judge files provided.")
        return

    header = ["Model", "Mean", "Median", "Pass", "Manual", "Fail", "Count"]
    print(" | ".join(header))
    print(" | ".join(["-" * len(h) for h in header]))
    for row in rows:
        print(
            f"{row['model']} | {row['mean']:.1f} | {row['median']:.1f} | "
            f"{row['pass']} | {row['manual_review']} | {row['fail']} | {row['count']}"
        )
    print()
    for row in rows:
        print(f"{row['model']}: {row['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judge JSON files")
    parser.add_argument("files", nargs="+", help="Paths to *_judge.json outputs")
    args = parser.parse_args()

    summarize([Path(f) for f in args.files])


if __name__ == "__main__":
    main()
