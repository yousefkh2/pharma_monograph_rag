#!/usr/bin/env python3
"""Build a BM25 index directly from a chunk metadata JSONL file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import BM25Index, default_tokenize


def iter_texts(metadata_path: Path) -> Iterable[str]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield record.get("text", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from chunk metadata JSONL")
    parser.add_argument("metadata", type=Path, help="Path to chunk_metadata.jsonl")
    parser.add_argument("index", type=Path, help="Path to output BM25 pickle")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    documents: List[str] = list(iter_texts(args.metadata))
    if not documents:
        raise SystemExit("No documents found in metadata file")

    index = BM25Index(k1=args.k1, b=args.b, tokenize=default_tokenize)
    index.build(documents)
    index.save(args.index)
    print(f"Indexed {len(documents)} chunks. Avg doc length: {index.avgdl:.2f} tokens")


if __name__ == "__main__":
    main()
