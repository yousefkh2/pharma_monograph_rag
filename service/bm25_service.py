#!/usr/bin/env python3
"""HTTP service exposing BM25 search over Lexicomp chunks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - runtime import guard
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install with: pip install fastapi uvicorn"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.search_utils import BM25SearchEngine, DebugEntry  # noqa: E402


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    section_title: Optional[str]
    section_code: Optional[str]
    drug_title: Optional[str]
    snippet: str
    text: Optional[str]


class DebugHit(BaseModel):
    chunk_id: str
    doc_id: str
    section_title: Optional[str]
    base_score: float
    adjustment: float
    final_score: float
    reasons: List[str]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchHit]
    debug: Optional[List[DebugHit]] = None


def to_debug_hit(entry: DebugEntry) -> DebugHit:
    return DebugHit(
        chunk_id=entry.chunk_id,
        doc_id=entry.doc_id,
        section_title=entry.section_title,
        base_score=entry.base_score,
        adjustment=entry.adjustment,
        final_score=entry.final_score,
        reasons=entry.reasons,
    )


def create_app(engine: BM25SearchEngine) -> FastAPI:
    app = FastAPI(title="Lexicomp BM25 Search", version="1.0.0")
    app.state.engine = engine

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/search", response_model=SearchResponse)
    async def search(
        query: str = Query(..., min_length=1, description="Free-text query"),
        top_k: int = Query(5, ge=1, le=50, description="Number of results to return"),
        retrieve_k: int = Query(200, ge=top_k, le=2000, description="Initial candidate pool"),
        must_drug: Optional[str] = Query(
            None, description="Restrict results to a drug id/alias (substring match)"
        ),
        debug: bool = Query(False, description="Include rerank diagnostics"),
        include_text: bool = Query(False, description="Return full chunk text in addition to snippet"),
    ) -> SearchResponse:
        hits, debug_entries = app.state.engine.search(
            query,
            top_k=top_k,
            retrieve_k=retrieve_k,
            must_drug=must_drug,
            debug=debug,
            include_text=include_text,
        )
        if not hits:
            raise HTTPException(status_code=404, detail="No results")

        results = [
            SearchHit(
                rank=hit["rank"],
                score=hit["score"],
                chunk_id=hit["chunk_id"],
                doc_id=hit["doc_id"],
                section_title=hit.get("section_title"),
                section_code=hit.get("section_code"),
                drug_title=hit.get("drug_title"),
                snippet=hit.get("snippet", ""),
                text=hit.get("text"),
            )
            for hit in hits
        ]

        debug_payload = [to_debug_hit(entry) for entry in debug_entries] if debug else None

        return SearchResponse(query=query, top_k=top_k, results=results, debug=debug_payload)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 search service")
    parser.add_argument("index", type=Path, help="Path to BM25 index pickle")
    parser.add_argument("metadata", type=Path, help="Path to chunk metadata JSONL")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = BM25SearchEngine.from_files(args.index, args.metadata)
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
