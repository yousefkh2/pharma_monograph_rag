#!/usr/bin/env python3
"""
Patch eval/datasets/candidate_pool.jsonl to update Q12 to oseltamivir.

This replaces the outdated Paxlovid entry with an influenza/oseltamivir
renal dosing query and a minimal, relevant candidate doc list.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = ROOT / "eval" / "datasets" / "candidate_pool.jsonl"


def build_candidate(doc_id: str, title: str, snippet: str = "") -> dict:
    return {
        "doc_id": doc_id,
        "bm25_top3": None,
        "bm25_max": None,
        "bge_top3": None,
        "bge_max": None,
        "title": title,
        "rep_snippet": snippet,
    }


def main() -> None:
    if not POOL_PATH.exists():
        raise SystemExit(f"candidate_pool.jsonl not found at {POOL_PATH}")

    lines = POOL_PATH.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Preserve any non-JSON lines as-is
            out_lines.append(line)
            continue

        if obj.get("qid") == "Q12":
            obj["query"] = (
                "Influenza treatment: eGFR 45 — what is the oseltamivir dose (renal adjustment)?"
            )
            obj["candidates"] = [
                build_candidate(
                    "oseltamivir-drug-information",
                    "Oseltamivir: Drug information",
                    "Monograph covers dosing and renal adjustment.",
                ),
                build_candidate(
                    "zanamivir-drug-information",
                    "Zanamivir: Drug information",
                    "Monograph covers dosing and indications.",
                ),
                build_candidate(
                    "peramivir-drug-information",
                    "Peramivir: Drug information",
                    "Monograph covers dosing and renal adjustment.",
                ),
                build_candidate(
                    "amantadine-drug-information",
                    "Amantadine: Drug information",
                    "Monograph includes dosing and renal considerations.",
                ),
                build_candidate(
                    "rimantadine-drug-information",
                    "Rimantadine: Drug information",
                    "Monograph includes dosing and renal considerations.",
                ),
            ]

            out_lines.append(json.dumps(obj, ensure_ascii=False))
        else:
            out_lines.append(line)

    POOL_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Updated Q12 in {POOL_PATH}")


if __name__ == "__main__":
    main()

