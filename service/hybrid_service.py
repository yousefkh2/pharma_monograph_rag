#!/usr/bin/env python3
"""Hybrid BM25 + dense vector search web service with a simple UI."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from nano_vectordb import NanoVectorDB
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import BM25Index, default_tokenize  # noqa: E402
from retrieval.search_utils import BM25SearchEngine, load_metadata  # noqa: E402

try:  # pragma: no cover - optional dependency
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency. Install with: pip install httpx fastapi uvicorn") from exc


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    dense_weight: float = Field(0.6, ge=0.0, le=1.0)
    bm25_candidates: int = Field(200, ge=10, le=2000)
    dense_candidates: int = Field(50, ge=10, le=500)


class SearchHit(BaseModel):
    rank: int
    score: float
    dense_score: float
    bm25_score: float
    chunk_id: str
    doc_id: Optional[str]
    section_title: Optional[str]
    section_code: Optional[str]
    drug_title: Optional[str]
    source_url: Optional[str]
    snippet: str
    text: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchHit]


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------
class HybridSearchEngine:
    def __init__(
        self,
        bm25_engine: BM25SearchEngine,
        metadata: Sequence[dict],
        metadata_map: Dict[str, dict],
        vectordb: NanoVectorDB,
        model: str,
        ollama_host: str,
        request_timeout: float,
    ) -> None:
        self.bm25_engine = bm25_engine
        self.metadata = metadata
        self.metadata_map = metadata_map
        self.vectordb = vectordb
        self.model = model
        self.client = httpx.AsyncClient(base_url=ollama_host, timeout=request_timeout)

    @classmethod
    async def create(
        cls,
        metadata_path: Path,
        bm25_index_path: Path,
        vectordb_path: Path,
        model: str,
        ollama_host: str,
        request_timeout: float,
    ) -> "HybridSearchEngine":
        metadata = load_metadata(metadata_path)
        if not metadata:
            raise RuntimeError(f"No metadata records found at {metadata_path}")

        # Ensure chunk ids are unique
        metadata_map: Dict[str, dict] = {}
        for record in metadata:
            chunk_id = record.get("chunk_id")
            if not chunk_id:
                continue
            metadata_map[chunk_id] = record

        index = BM25Index.load(bm25_index_path, tokenize=default_tokenize)
        bm25_engine = BM25SearchEngine(index, list(metadata))

        sample_record = next((record for record in metadata if record.get("embedding_dim")), None)
        embedding_dim = int(sample_record.get("embedding_dim", 0) if sample_record else 0)
        if not embedding_dim:
            raise RuntimeError("Unable to infer embedding dimension from metadata. Ensure the JSONL includes 'embedding_dim'.")

        vectordb = NanoVectorDB(embedding_dim, storage_file=str(vectordb_path))

        engine = cls(
            bm25_engine=bm25_engine,
            metadata=metadata,
            metadata_map=metadata_map,
            vectordb=vectordb,
            model=model,
            ollama_host=ollama_host,
            request_timeout=request_timeout,
        )
        await engine._ensure_model_available()
        return engine

    async def close(self) -> None:
        await self.client.aclose()

    async def _ensure_model_available(self) -> None:
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()
        payload = resp.json()
        models = {item.get("name") for item in payload.get("models", [])}
        if self.model not in models:
            raise RuntimeError(
                f"Embedding model '{self.model}' is not available on the Ollama host. "
                f"Use 'ollama pull {self.model}' to install it."
            )

    async def _embed_query(self, query: str) -> np.ndarray:
        response = await self.client.post("/api/embeddings", json={"model": self.model, "prompt": query})
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise RuntimeError("Ollama response missing 'embedding'")
        return np.asarray(embedding, dtype=np.float32)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        positive = [value for value in scores.values() if value > 0]
        if not positive:
            return {key: 0.0 for key in scores}
        max_val = max(positive)
        if math.isclose(max_val, 0.0):
            return {key: 0.0 for key in scores}
        return {key: value / max_val if value > 0 else 0.0 for key, value in scores.items()}

    async def search(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        bm25_candidates: int,
        dense_candidates: int,
    ) -> List[SearchHit]:
        query_embedding = await self._embed_query(query)
        dense_results = self.vectordb.query(query_embedding, top_k=max(dense_candidates, top_k))
        dense_scores: Dict[str, float] = {}
        for result in dense_results:
            chunk_id = result.get("__id__")
            score = float(result.get("__metrics__", 0.0) or 0.0)
            if math.isnan(score) or score <= 0:
                continue
            dense_scores[chunk_id] = max(score, 0.0)

        bm25_hits, _ = self.bm25_engine.search(
            query,
            top_k=bm25_candidates,
            retrieve_k=bm25_candidates,
            include_text=False,
        )
        bm25_scores: Dict[str, float] = {}
        for hit in bm25_hits:
            chunk_id = hit.get("chunk_id")
            score = float(hit.get("score", 0.0) or 0.0)
            if score <= 0 or not chunk_id:
                continue
            bm25_scores[chunk_id] = score

        norm_dense = self._normalize_scores(dense_scores)
        norm_bm25 = self._normalize_scores(bm25_scores)

        combined: List[Tuple[str, float, float, float]] = []
        seen: set[str] = set()
        for chunk_id in set(list(norm_dense.keys()) + list(norm_bm25.keys())):
            dense_score = norm_dense.get(chunk_id, 0.0)
            bm25_score = norm_bm25.get(chunk_id, 0.0)
            combined_score = dense_weight * dense_score + (1 - dense_weight) * bm25_score
            combined.append((chunk_id, combined_score, dense_score, bm25_score))
            seen.add(chunk_id)

        combined.sort(key=lambda item: item[1], reverse=True)
        top_results = combined[:top_k]

        hits: List[SearchHit] = []
        for rank, (chunk_id, score, dense_score, bm25_score) in enumerate(top_results, start=1):
            record = self.metadata_map.get(chunk_id)
            if not record:
                continue
            text = record.get("text", "")
            snippet = text.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240].rstrip() + "…"
            hits.append(
                SearchHit(
                    rank=rank,
                    score=score,
                    dense_score=dense_score,
                    bm25_score=bm25_score,
                    chunk_id=chunk_id,
                    doc_id=record.get("doc_id"),
                    section_title=record.get("section_title"),
                    section_code=record.get("section_code"),
                    drug_title=record.get("drug_title"),
                    source_url=record.get("source_url"),
                    snippet=snippet,
                    text=text,
                )
            )
        return hits


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------
def build_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pharmacy Copilot Search</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; color: #1f2933; }
    h1 { font-size: 1.8rem; margin-bottom: 1rem; }
    form { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    input[type="text"] { flex: 1 1 420px; padding: 0.6rem; font-size: 1rem; border-radius: 0.5rem; border: 1px solid #cbd5e1; }
    button { padding: 0.6rem 1.2rem; border-radius: 0.5rem; border: none; background: #2563eb; color: white; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .controls { display: flex; gap: 1rem; align-items: center; font-size: 0.9rem; color: #475569; }
    .controls label { display: flex; align-items: center; gap: 0.35rem; }
    .controls input { padding: 0.3rem 0.4rem; border-radius: 0.35rem; border: 1px solid #cbd5e1; }
    .results { display: grid; gap: 1rem; }
    .card { border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem 1.25rem; background: #f8fafc; }
    .card h2 { margin: 0 0 0.4rem; font-size: 1.05rem; color: #1d4ed8; }
    .meta { font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem; }
    .scores { font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem; }
    .snippet { white-space: pre-wrap; line-height: 1.45; font-size: 0.95rem; color: #1f2933; }
    .toggle { margin-top: 0.6rem; background: transparent; border: none; color: #2563eb; font-weight: 600; cursor: pointer; padding: 0; }
    .toggle:hover { text-decoration: underline; }
    .error { color: #b91c1c; margin-top: 0.75rem; }
  </style>
</head>
<body>
  <h1>Pharmacy Copilot Search</h1>
  <form id="search-form">
    <input type="text" id="query" placeholder="Ask about dosing, interactions, etc." autocomplete="off" required>
    <button type="submit" id="submit-btn">Search</button>
    <div class="controls">
      <label>Top K <input type="number" id="topk" value="5" min="1" max="20"></label>
      <label>Dense weight <input type="number" id="dense-weight" value="0.6" min="0" max="1" step="0.05"></label>
    </div>
  </form>
  <div id="error" class="error"></div>
  <div id="results" class="results"></div>

  <script>
    const form = document.getElementById('search-form');
    const queryInput = document.getElementById('query');
    const topkInput = document.getElementById('topk');
    const denseInput = document.getElementById('dense-weight');
    const submitBtn = document.getElementById('submit-btn');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');

    const formatScore = (value) => Number.isFinite(value) ? value.toFixed(3) : '0.000';
    const escapeHtml = (value) => (value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

    const attachToggleHandlers = () => {
      resultsDiv.querySelectorAll('.snippet').forEach((el) => {
        el.textContent = decodeURIComponent(el.dataset.snippet || '');
      });
      resultsDiv.querySelectorAll('.toggle').forEach((btn) => {
        const snippetEl = btn.previousElementSibling;
        const hasMore = snippetEl.dataset.hasMore === 'true';
        if (!hasMore) {
          btn.style.display = 'none';
          return;
        }
        btn.addEventListener('click', () => {
          const expanded = btn.getAttribute('data-expanded') === 'true';
          if (expanded) {
            snippetEl.textContent = decodeURIComponent(snippetEl.dataset.snippet || '');
            btn.textContent = 'Show more';
            btn.setAttribute('data-expanded', 'false');
          } else {
            snippetEl.textContent = decodeURIComponent(snippetEl.dataset.full || snippetEl.dataset.snippet || '');
            btn.textContent = 'Show less';
            btn.setAttribute('data-expanded', 'true');
          }
        });
      });
    };

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;

      resultsDiv.innerHTML = '';
      errorDiv.textContent = '';
      submitBtn.disabled = true;
      submitBtn.textContent = 'Searching…';

      try {
        const response = await fetch('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            top_k: parseInt(topkInput.value, 10) || 5,
            dense_weight: parseFloat(denseInput.value) || 0.6
          })
        });

        if (!response.ok) {
          const payload = await response.json();
          throw new Error(payload.detail || 'Search failed');
        }

        const payload = await response.json();
        if (!payload.results.length) {
          resultsDiv.innerHTML = '<p>No results found.</p>';
          return;
        }

        resultsDiv.innerHTML = payload.results.map((hit) => {
          const sectionTitle = escapeHtml(hit.section_title || 'Untitled section');
          const drugTitle = escapeHtml(hit.drug_title || hit.doc_id || 'Unknown drug');
          const sourceLink = hit.source_url ? `· <a href="${encodeURI(hit.source_url)}" target="_blank" rel="noopener noreferrer">source</a>` : '';
          const snippetAttr = encodeURIComponent(hit.snippet || '');
          const fullAttr = encodeURIComponent(hit.text || '');
          const needsMore = (hit.text || '').trim().length > (hit.snippet || '').trim().length;

          return `
            <article class="card">
              <h2>#${hit.rank} · ${sectionTitle}</h2>
              <div class="meta">
                <strong>${drugTitle}</strong> ${sourceLink}
              </div>
              <div class="scores">Combined: ${formatScore(hit.score)} · Dense: ${formatScore(hit.dense_score)} · BM25: ${formatScore(hit.bm25_score)}</div>
              <div class="snippet" data-snippet="${snippetAttr}" data-full="${fullAttr}" data-has-more="${needsMore}"></div>
              <button type="button" class="toggle" data-expanded="false">Show more</button>
            </article>
          `;
        }).join('');

        attachToggleHandlers();
      } catch (err) {
        console.error(err);
        errorDiv.textContent = err.message;
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Search';
      }
    });
  </script>
</body>
</html>"""


def create_app(engine: HybridSearchEngine) -> FastAPI:
    app = FastAPI(title="Pharmacy Copilot Hybrid Search", version="1.0.0")
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(build_html())

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        engine: HybridSearchEngine = app.state.engine
        hits = await engine.search(
            query=request.query,
            top_k=request.top_k,
            dense_weight=request.dense_weight,
            bm25_candidates=request.bm25_candidates,
            dense_candidates=request.dense_candidates,
        )
        if not hits:
            raise HTTPException(status_code=404, detail="No results found")
        return SearchResponse(query=request.query, top_k=request.top_k, results=hits)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return app


@asynccontextmanager
async def lifespan(metadata_path: Path, bm25_path: Path, vectordb_path: Path, model: str, host: str, timeout: float):
    engine = await HybridSearchEngine.create(
        metadata_path=metadata_path,
        bm25_index_path=bm25_path,
        vectordb_path=vectordb_path,
        model=model,
        ollama_host=host,
        request_timeout=timeout,
    )
    try:
        yield engine
    finally:
        await engine.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid dense + sparse search service")
    parser.add_argument("--metadata", type=Path, default=Path("vector_store/chunk_metadata.jsonl"))
    parser.add_argument("--bm25", type=Path, default=Path("vector_store/bm25_index.pkl"))
    parser.add_argument("--vectordb", type=Path, default=Path("vector_store/nano_chunks.json"))
    parser.add_argument("--model", default="bge-m3:latest", help="Ollama embedding model")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout for Ollama requests")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Run uvicorn in autoreload mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build engine synchronously before starting uvicorn so startup failures surface early
    engine = asyncio.run(HybridSearchEngine.create(
        metadata_path=args.metadata,
        bm25_index_path=args.bm25,
        vectordb_path=args.vectordb,
        model=args.model,
        ollama_host=args.ollama_host,
        request_timeout=args.timeout,
    ))
    app = create_app(engine)

    config = uvicorn.Config(app, host=args.host, port=args.port, reload=args.reload)
    server = uvicorn.Server(config)

    try:
        asyncio.run(server.serve())
    finally:
        asyncio.run(engine.close())


if __name__ == "__main__":
    main()
