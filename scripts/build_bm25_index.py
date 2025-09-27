#!/usr/bin/env python3
"""Build a BM25 index from parsed Lexicomp chunk data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import BM25Index, default_tokenize


def iter_chunks(parsed_json: Path) -> Iterable[Dict[str, object]]:
    data = json.loads(parsed_json.read_text(encoding="utf-8"))
    for doc in data:
        doc_id = doc.get("doc_id")
        drug_title = doc.get("drug_title")
        for section in doc.get("sections", []):
            section_title = section.get("section_title")
            section_code = section.get("section_code")
            for chunk in section.get("chunks", []):
                chunk_id = f"{doc_id}::S{section.get('order', 0):02d}::C{chunk.get('chunk_index', 0):02d}"
                yield {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "drug_title": drug_title,
                    "section_title": section_title,
                    "section_code": section_code,
                    "text": chunk.get("text", ""),
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index for Lexicomp chunks")
    parser.add_argument("parsed_json", type=Path, help="Path to parsed output JSON (from ingest_sections.py)")
    parser.add_argument("index_path", type=Path, help="Where to store the BM25 index pickle")
    parser.add_argument(
        "metadata_path",
        type=Path,
        help="Path to JSONL file storing chunk metadata (text + attributes)",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    chunks = list(iter_chunks(args.parsed_json))
    documents: List[str] = [chunk["text"] for chunk in chunks]

    index = BM25Index(k1=args.k1, b=args.b, tokenize=default_tokenize)
    index.build(documents)
    index.save(args.index_path)

    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Indexed {len(documents)} chunks")
    print(f"Average document length: {index.avgdl:.2f} tokens")


if __name__ == "__main__":
    main()
