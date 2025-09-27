#!/usr/bin/env python3
"""Query the BM25 index and print top matches."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.search_utils import BM25SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval over Lexicomp chunks")
    parser.add_argument("index_path", type=Path, help="BM25 index pickle produced by build_bm25_index.py")
    parser.add_argument("metadata_path", type=Path, help="Chunk metadata JSONL")
    parser.add_argument("query", type=str, help="Free-text query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=200, help="Initial candidate count before rerank")
    parser.add_argument("--must-drug", type=str, default=None, help="Restrict results to doc_ids containing this text")
    parser.add_argument("--debug", action="store_true", help="Print rerank diagnostics")
    args = parser.parse_args()

    engine = BM25SearchEngine.from_files(args.index_path, args.metadata_path)
    results, debug_entries = engine.search(
        args.query,
        top_k=args.top_k,
        retrieve_k=args.retrieve_k,
        must_drug=args.must_drug,
        debug=args.debug,
        include_text=True,
    )

    if not results:
        print("No results")
        return

    if args.debug and debug_entries:
        for entry in debug_entries:
            print(
                "DEBUG:"
                f" {entry.chunk_id} base={entry.base_score:.3f} adj={entry.adjustment:.3f} "
                f"final={entry.final_score:.3f} reasons={','.join(entry.reasons) if entry.reasons else 'n/a'}"
            )

    for hit in results:
        print(
            f"{hit['rank']}. score={hit['score']:.3f} | {hit['chunk_id']} | "
            f"{hit['section_title']} | {hit['drug_title']}"
        )
        print(f"   {hit['snippet']}\n")


if __name__ == "__main__":
    main()
