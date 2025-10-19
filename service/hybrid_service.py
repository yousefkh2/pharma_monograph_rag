#!/usr/bin/env python3
"""Hybrid BM25 + dense vector search web service with a simple UI."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv is optional, environment variables may be set another way

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.bm25_index import BM25Index, default_tokenize  # noqa: E402
from retrieval.search_utils import BM25SearchEngine, load_metadata  # noqa: E402
from retrieval.llm_monograph_selector import (
    LLMMonographSelector,
    Monograph,
    MonographCatalog,
    MonographSelection,
)

from eval.llm_client import LLMClient, LLMConfig, DEFAULT_SYSTEM_PROMPT  # noqa: E402

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
    dense_raw: Optional[float] = None
    bm25_raw: Optional[float] = None
    chunk_id: str
    doc_id: Optional[str]
    section_title: Optional[str]
    section_code: Optional[str]
    drug_title: Optional[str]
    source_url: Optional[str]
    snippet: str
    text: str


DEFAULT_SELECTOR_PROVIDER = "openrouter"
DEFAULT_SELECTOR_MODEL = "openai/gpt-5-mini"
DEFAULT_SELECTOR_SYSTEM_PROMPT = (
    "You are a pharmacy monograph selector. Choose exactly one doc_id from the catalog that best answers the question. "
    "Rules: (1) Match the precise drug entity and formulation in the query; avoid combination products unless explicitly requested. "
    "(2) Respect negations/exclusions (e.g., 'NOT amoxicillin-clavulanate' means do not select that doc). "
    "(3) Prefer monographs whose scope matches the patient context (age group, renal/hepatic status, route, indication). "
    "(4) Normalize noisy spelling, abbreviations, and brands to their canonical drugs (AOM, NVAF, BBW, mg/kg/day, etc.). "
    "(5) If multiple docs are related, pick the single most specific one that directly answers the question. "
    'Respond only with JSON of the form {{"doc_id":"<doc_id>","confidence":<0-1>,"reason":"<15-40 words>"}}.'
)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchHit]


def resolve_selector_base_url(provider: str, base_url: Optional[str]) -> str:
    if base_url:
        return base_url
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return "https://api.openai.com/v1"
    if provider_lower == "openrouter":
        return "https://openrouter.ai/api/v1"
    return "http://localhost:11434"


def build_selector_config_from_args(args: argparse.Namespace) -> Optional[LLMConfig]:
    provider = (
        getattr(args, "selector_provider", None)
        or os.getenv("SELECTOR_LLM_PROVIDER")
        or DEFAULT_SELECTOR_PROVIDER
    )
    model = (
        getattr(args, "selector_model", None)
        or os.getenv("SELECTOR_LLM_MODEL")
        or DEFAULT_SELECTOR_MODEL
    )
    if not provider or not model:
        return None

    base_url = resolve_selector_base_url(
        provider,
        getattr(args, "selector_base_url", None) or os.getenv("SELECTOR_LLM_BASE_URL"),
    )

    api_key = (
        getattr(args, "selector_api_key", None)
        or os.getenv("SELECTOR_LLM_API_KEY")
        or os.getenv("EVAL_LLM_API_KEY")
    )
    temperature = getattr(args, "selector_temperature", None)
    if temperature is None:
        temperature = float(os.getenv("SELECTOR_LLM_TEMPERATURE", "0.0"))
    top_p = getattr(args, "selector_top_p", None)
    if top_p is None:
        top_p = float(os.getenv("SELECTOR_LLM_TOP_P", "0.3"))
    max_tokens = getattr(args, "selector_max_tokens", None)
    if max_tokens is None:
        max_tokens = int(os.getenv("SELECTOR_LLM_MAX_TOKENS", "2048"))
    system_prompt = (
        getattr(args, "selector_system_prompt", None)
        or os.getenv("SELECTOR_LLM_SYSTEM_PROMPT")
        or DEFAULT_SELECTOR_SYSTEM_PROMPT
    )

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


def build_answer_config_from_args(args: argparse.Namespace) -> Optional[LLMConfig]:
    provider = getattr(args, "answer_provider", None) or os.getenv("ANSWER_LLM_PROVIDER") or os.getenv("EVAL_LLM_PROVIDER")
    model = getattr(args, "answer_model", None) or os.getenv("ANSWER_LLM_MODEL") or os.getenv("EVAL_LLM_MODEL")
    if not provider or not model:
        return None

    base_url_env = getattr(args, "answer_base_url", None) or os.getenv("ANSWER_LLM_BASE_URL") or os.getenv("EVAL_LLM_BASE_URL")
    base_url = resolve_selector_base_url(provider, base_url_env)

    api_key = getattr(args, "answer_api_key", None) or os.getenv("ANSWER_LLM_API_KEY") or os.getenv("EVAL_LLM_API_KEY")
    temperature = getattr(args, "answer_temperature", None)
    if temperature is None:
        temperature = float(os.getenv("ANSWER_LLM_TEMPERATURE", os.getenv("EVAL_LLM_TEMPERATURE", "0.2")))
    top_p = getattr(args, "answer_top_p", None)
    if top_p is None:
        top_p = float(os.getenv("ANSWER_LLM_TOP_P", os.getenv("EVAL_LLM_TOP_P", "0.3")))
    max_tokens = getattr(args, "answer_max_tokens", None)
    if max_tokens is None:
        max_tokens = int(os.getenv("ANSWER_LLM_MAX_TOKENS", os.getenv("EVAL_LLM_MAX_TOKENS", "512")))
    system_prompt = getattr(args, "answer_system_prompt", None) or os.getenv("ANSWER_LLM_SYSTEM_PROMPT") or os.getenv("EVAL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    dense_weight: float = Field(0.6, ge=0.0, le=1.0)
    bm25_candidates: int = Field(200, ge=10, le=2000)
    dense_candidates: int = Field(50, ge=10, le=500)


class RerankSelection(BaseModel):
    doc_id: Optional[str]
    confidence: Optional[float]
    reason: Optional[str]
    raw_response: str
    prompt: str


class RerankResponse(BaseModel):
    query: str
    selection: RerankSelection
    selected_results: List[SearchHit]
    original_results: List[SearchHit]


class AnswerConfigRequest(BaseModel):
    enabled: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: Optional[float] = None


class AnswerConfigResponse(BaseModel):
    enabled: bool
    provider: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    timeout: float


class MonographChunk(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    section_title: Optional[str]
    section_code: Optional[str]
    drug_title: Optional[str]
    snippet: Optional[str]
    text: str


class AnswerRequest(BaseModel):
    query: str
    doc_id: str


class AnswerResponse(BaseModel):
    doc_id: str
    chunk_count: int
    answer: str
    chunks: List[MonographChunk]


class SelectorConfigRequest(BaseModel):
    enabled: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: Optional[float] = None
    candidates: Optional[int] = None


class SelectorConfigResponse(BaseModel):
    enabled: bool
    provider: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    system_prompt: Optional[str]
    timeout: float
    candidates: int


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
        *,
        catalog: MonographCatalog,
        selector_candidates: int = 30,
        selector_timeout: float = 60.0,
        answer_timeout: float = 120.0,
    ) -> None:
        self.bm25_engine = bm25_engine
        self.metadata = metadata
        self.metadata_map = metadata_map
        self.vectordb = vectordb
        self.model = model
        self.client = httpx.AsyncClient(base_url=ollama_host, timeout=request_timeout)
        self.catalog = catalog
        self.selector_candidates = max(1, selector_candidates)
        self.selector_timeout = selector_timeout
        self.selector_client: Optional[LLMClient] = None
        self.selector: Optional[LLMMonographSelector] = None
        self.selector_config: Optional[LLMConfig] = None
        self.answer_timeout = answer_timeout
        self.answer_client: Optional[LLMClient] = None
        self.answer_config: Optional[LLMConfig] = None

    @classmethod
    async def create(
        cls,
        metadata_path: Path,
        bm25_index_path: Path,
        vectordb_path: Path,
        model: str,
        ollama_host: str,
        request_timeout: float,
        *,
        selector_config: Optional[LLMConfig] = None,
        selector_timeout: float = 60.0,
        selector_candidates: int = 30,
        answer_config: Optional[LLMConfig] = None,
        answer_timeout: float = 120.0,
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

        catalog = MonographCatalog.from_metadata(metadata)

        engine = cls(
            bm25_engine=bm25_engine,
            metadata=metadata,
            metadata_map=metadata_map,
            vectordb=vectordb,
            model=model,
            ollama_host=ollama_host,
            request_timeout=request_timeout,
            catalog=catalog,
            selector_candidates=selector_candidates,
            selector_timeout=selector_timeout,
            answer_timeout=answer_timeout,
        )
        await engine._ensure_model_available()
        if selector_config is not None:
            await engine.update_selector(
                selector_config,
                timeout=selector_timeout,
                candidates=selector_candidates,
            )
        if answer_config is not None:
            await engine.update_answer(
                answer_config,
                timeout=answer_timeout,
            )
        return engine

    async def close(self) -> None:
        await self.client.aclose()
        if self.selector_client is not None:
            await self.selector_client.close()
            self.selector_client = None
            self.selector = None
            self.selector_config = None
        if self.answer_client is not None:
            await self.answer_client.close()
            self.answer_client = None
            self.answer_config = None

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
                    dense_raw=dense_scores.get(chunk_id),
                    bm25_raw=bm25_scores.get(chunk_id),
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

    @property
    def selector_enabled(self) -> bool:
        return self.selector is not None and self.catalog is not None

    async def update_selector(
        self,
        config: Optional[LLMConfig],
        *,
        timeout: Optional[float] = None,
        candidates: Optional[int] = None,
    ) -> None:
        if candidates is not None:
            self.selector_candidates = max(1, candidates)
        if timeout is not None:
            self.selector_timeout = timeout

        if self.selector_client is not None:
            await self.selector_client.close()
            self.selector_client = None
            self.selector = None
            self.selector_config = None

        if config is None:
            return

        client = LLMClient(config, timeout=self.selector_timeout)
        self.selector_client = client
        self.selector = LLMMonographSelector(
            self.catalog,
            client,
            max_catalog_items=self.selector_candidates,
        )
        self.selector_config = config

    async def update_answer(
        self,
        config: Optional[LLMConfig],
        *,
        timeout: Optional[float] = None,
    ) -> None:
        if timeout is not None:
            self.answer_timeout = timeout

        if self.answer_client is not None:
            await self.answer_client.close()
            self.answer_client = None
            self.answer_config = None

        if config is None:
            return

        client = LLMClient(config, timeout=self.answer_timeout)
        self.answer_client = client
        self.answer_config = config

    async def answer_question(self, query: str, doc_id: str) -> Tuple[str, List[Dict[str, object]]]:
        if not self.answer_client:
            raise RuntimeError("Answering LLM is not configured")

        monograph = self.catalog.build_context(doc_id, include_text=True)
        if not monograph:
            raise RuntimeError(f"No chunks found for doc_id '{doc_id}'")

        allowed_chunk_ids = [item.get("chunk_id") for item in monograph if item.get("chunk_id")]

        answer = await self.answer_client.generate_answer(
            question=query,
            contexts=monograph,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        return answer, monograph

    async def rerank(
        self,
        query: str,
        *,
        top_k: int,
        dense_weight: float,
        bm25_candidates: int,
        dense_candidates: int,
    ) -> Tuple[MonographSelection, List[SearchHit], List[SearchHit]]:
        if not self.selector or not self.catalog:
            raise RuntimeError("LLM selector is not configured")

        hits = await self.search(
            query=query,
            top_k=top_k,
            dense_weight=dense_weight,
            bm25_candidates=bm25_candidates,
            dense_candidates=dense_candidates,
        )

        if not hits:
            selection = MonographSelection(doc_id=None, raw_response="", prompt="")
            return selection, [], []

        candidate_monographs: List[Monograph] = []
        snippet_map: Dict[str, str] = {}
        seen: set[str] = set()

        for hit in hits:
            doc_id = hit.doc_id
            if not doc_id or doc_id in seen:
                continue
            mono = self.catalog.get(doc_id)
            if not mono:
                continue
            snippet_map[doc_id] = hit.drug_title or mono.display_name
            candidate_monographs.append(mono)
            seen.add(doc_id)
            if len(candidate_monographs) >= self.selector_candidates:
                break

        if not candidate_monographs:
            tokens = default_tokenize(query)
            candidate_monographs = self.catalog.top_k_by_token_overlap(
                tokens,
                self.selector_candidates,
            )

        if not candidate_monographs:
            candidate_monographs = list(self.catalog)[: self.selector_candidates]

        if not snippet_map:
            snippet_map = {mono.doc_id: mono.display_name for mono in candidate_monographs}

        selection = await self.selector.select(
            query,
            candidates=candidate_monographs,
            snippet_map=snippet_map,
        )

        selected_hits = [hit for hit in hits if hit.doc_id == selection.doc_id]
        if not selected_hits:
            selected_hits = hits[:top_k]

        # Reindex selected hits to provide a clean ranking from 1..N for display
        selected_hits = [hit.copy(update={"rank": idx}) for idx, hit in enumerate(selected_hits, start=1)]

        return selection, hits, selected_hits


# ---------------------------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------------------------
def build_html() -> str:
    return """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Pharmacy Copilot Search</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 960px; color: #1f2933; }
    h1 { font-size: 1.8rem; margin-bottom: 1rem; }
    form { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    input[type='text'] { flex: 1 1 420px; padding: 0.6rem; font-size: 1rem; border-radius: 0.5rem; border: 1px solid #cbd5e1; }
    button { padding: 0.6rem 1.2rem; border-radius: 0.5rem; border: none; background: #2563eb; color: white; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .controls { display: flex; gap: 1rem; align-items: center; font-size: 0.9rem; color: #475569; }
    .controls label { display: flex; align-items: center; gap: 0.35rem; }
    .controls input { padding: 0.3rem 0.4rem; border-radius: 0.35rem; border: 1px solid #cbd5e1; }
    .llm-controls { display: flex; gap: 0.75rem; align-items: center; margin-bottom: 1.25rem; flex-wrap: wrap; }
    .selector-config, .answer-config { border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem 1.25rem; background: #ffffff; margin-bottom: 1.5rem; }
    .selector-header, .answer-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.75rem; }
    .selector-config h2, .answer-config h2 { margin: 0; font-size: 1.2rem; }
    .selector-grid, .answer-grid { display: grid; gap: 0.75rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 0.75rem; }
    .selector-config label, .answer-config label { display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.9rem; color: #475569; }
    .selector-config input, .selector-config select, .selector-config textarea,
    .answer-config input, .answer-config select, .answer-config textarea { padding: 0.45rem 0.5rem; border-radius: 0.45rem; border: 1px solid #cbd5e1; font-size: 0.95rem; }
    .selector-system textarea, .answer-system textarea { resize: vertical; }
    .selector-buttons, .answer-buttons { display: flex; gap: 0.75rem; align-items: center; margin-top: 0.5rem; flex-wrap: wrap; }
    #selector-status, #answer-status { font-size: 0.9rem; color: #475569; }
    .selection { border: 1px dashed #94a3b8; padding: 1rem; border-radius: 0.75rem; background: #f1f5f9; margin-bottom: 1.25rem; }
    .selection h2 { font-size: 1.05rem; margin: 0 0 0.4rem; }
    .results { display: grid; gap: 1rem; }
    .card { border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem 1.25rem; background: #f8fafc; }
    .card h2 { margin: 0 0 0.4rem; font-size: 1.05rem; color: #1d4ed8; }
    .meta { font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem; }
    .scores { font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem; }
    .snippet { white-space: pre-wrap; line-height: 1.45; font-size: 0.95rem; color: #1f2933; }
    .toggle { margin-top: 0.6rem; background: transparent; border: none; color: #2563eb; font-weight: 600; cursor: pointer; padding: 0; }
    .toggle:hover { text-decoration: underline; }
    .error { color: #b91c1c; margin-top: 0.75rem; }
    .answer-output { border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1rem 1.25rem; background: #f8fafc; margin-top: 1.5rem; }
    .answer-output pre { white-space: pre-wrap; background: #0f172a; color: #f8fafc; padding: 0.75rem 1rem; border-radius: 0.5rem; overflow-x: auto; font-size: 0.95rem; }
    .answer-output details { margin-top: 1rem; }
    .answer-output details pre { background: #111827; }
    .answer-output summary { cursor: pointer; font-weight: 600; color: #1d4ed8; }
  </style>
</head>
<body>
  <h1>Pharmacy Copilot Search</h1>
  <form id='search-form'>
    <input type='text' id='query' placeholder='Ask about dosing, interactions, etc.' autocomplete='off' required>
    <button type='submit' id='submit-btn'>Search</button>
    <div class='controls'>
      <label>Top K <input type='number' id='topk' value='5' min='1' max='20'></label>
      <label>Dense weight <input type='number' id='dense-weight' value='0.6' min='0' max='1' step='0.05'></label>
    </div>
  </form>
  <div class='llm-controls'>
    <button type='button' id='rerank-btn' disabled>LLM rerank</button>
    <button type='button' id='answer-btn' disabled>Generate answer</button>
  </div>
  <section class='selector-config'>
    <div class='selector-header'>
      <h2>LLM Selector Settings</h2>
      <span id='selector-status'></span>
    </div>
    <form id='selector-form'>
      <div class='selector-grid'>
        <label>Provider
          <select id='selector-provider'>
            <option value=''>Choose…</option>
            <option value='ollama'>ollama</option>
            <option value='openai'>openai</option>
            <option value='openrouter'>openrouter</option>
          </select>
        </label>
        <label>Model
          <input type='text' id='selector-model' placeholder='e.g. bge-m3 or gpt-4o-mini'>
        </label>
        <label>Base URL
          <input type='url' id='selector-base-url' placeholder='Optional override'>
        </label>
        <label>API key
          <input type='password' id='selector-api-key' placeholder='Optional'>
        </label>
        <label>Temperature
          <input type='number' step='0.05' id='selector-temperature' placeholder='0.0'>
        </label>
        <label>Top p
          <input type='number' step='0.05' id='selector-top-p' placeholder='0.3'>
        </label>
        <label>Max tokens
          <input type='number' id='selector-max-tokens' placeholder='256'>
        </label>
        <label>Timeout (s)
          <input type='number' step='0.5' id='selector-timeout' placeholder='60'>
        </label>
        <label>Candidates
          <input type='number' min='1' max='100' id='selector-candidates' placeholder='30'>
        </label>
      </div>
      <label class='selector-system'>System prompt
        <textarea id='selector-system-prompt' rows='3' placeholder='Leave blank for default prompt'></textarea>
      </label>
      <div class='selector-buttons'>
        <button type='submit' id='selector-apply'>Apply selector settings</button>
        <button type='button' id='selector-disable'>Disable selector</button>
      </div>
      <p id='selector-error' class='error'></p>
    </form>
  </section>
  <section class='answer-config'>
    <div class='answer-header'>
      <h2>Answering LLM Settings</h2>
      <span id='answer-status'></span>
    </div>
    <form id='answer-form'>
      <div class='answer-grid'>
        <label>Provider
          <select id='answer-provider'>
            <option value=''>Choose…</option>
            <option value='ollama'>ollama</option>
            <option value='openai'>openai</option>
            <option value='openrouter'>openrouter</option>
          </select>
        </label>
        <label>Model
          <input type='text' id='answer-model' placeholder='e.g. gpt-4o-mini or pharmacy-copilot'>
        </label>
        <label>Base URL
          <input type='url' id='answer-base-url' placeholder='Optional override'>
        </label>
        <label>API key
          <input type='password' id='answer-api-key' placeholder='Optional'>
        </label>
        <label>Temperature
          <input type='number' step='0.05' id='answer-temperature' placeholder='0.2'>
        </label>
        <label>Top p
          <input type='number' step='0.05' id='answer-top-p' placeholder='0.3'>
        </label>
        <label>Max tokens
          <input type='number' id='answer-max-tokens' placeholder='512'>
        </label>
        <label>Timeout (s)
          <input type='number' step='0.5' id='answer-timeout' placeholder='120'>
        </label>
      </div>
      <label class='answer-system'>System prompt
        <textarea id='answer-system-prompt' rows='3' placeholder='Leave blank for default prompt'></textarea>
      </label>
      <div class='answer-buttons'>
        <button type='submit' id='answer-apply'>Apply answering settings</button>
        <button type='button' id='answer-disable'>Disable answering LLM</button>
      </div>
      <p id='answer-error' class='error'></p>
    </form>
  </section>
  <div id='error' class='error'></div>
  <div id='selection' class='selection' style='display:none;'></div>
  <div id='results' class='results'></div>
  <div id='reranked' class='results'></div>
  <section id='answer-output' class='answer-output' style='display:none;'>
    <h2>LLM Answer</h2>
    <p id='answer-meta'></p>
    <pre id='answer-text'></pre>
    <details id='answer-chunks-container'>
      <summary>Show monograph chunks</summary>
      <div id='answer-chunks'></div>
    </details>
  </section>

  <script>
    const form = document.getElementById('search-form');
    const queryInput = document.getElementById('query');
    const topkInput = document.getElementById('topk');
    const denseInput = document.getElementById('dense-weight');
    const submitBtn = document.getElementById('submit-btn');
    const rerankBtn = document.getElementById('rerank-btn');
    const answerBtn = document.getElementById('answer-btn');
    const selectorStatus = document.getElementById('selector-status');
    const answerStatus = document.getElementById('answer-status');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const selectionDiv = document.getElementById('selection');
    const rerankedDiv = document.getElementById('reranked');
    const answerOutput = document.getElementById('answer-output');
    const answerMeta = document.getElementById('answer-meta');
    const answerText = document.getElementById('answer-text');
    const answerChunksDiv = document.getElementById('answer-chunks');

    const selectorForm = document.getElementById('selector-form');
    const selectorError = document.getElementById('selector-error');
    const selectorApplyBtn = document.getElementById('selector-apply');
    const selectorDisableBtn = document.getElementById('selector-disable');
    const selectorProviderSelect = document.getElementById('selector-provider');
    const selectorModelInput = document.getElementById('selector-model');
    const selectorBaseUrlInput = document.getElementById('selector-base-url');
    const selectorApiKeyInput = document.getElementById('selector-api-key');
    const selectorTemperatureInput = document.getElementById('selector-temperature');
    const selectorTopPInput = document.getElementById('selector-top-p');
    const selectorMaxTokensInput = document.getElementById('selector-max-tokens');
    const selectorTimeoutInput = document.getElementById('selector-timeout');
    const selectorCandidatesInput = document.getElementById('selector-candidates');
    const selectorSystemPromptTextarea = document.getElementById('selector-system-prompt');

    const answerForm = document.getElementById('answer-form');
    const answerError = document.getElementById('answer-error');
    const answerApplyBtn = document.getElementById('answer-apply');
    const answerDisableBtn = document.getElementById('answer-disable');
    const answerProviderSelect = document.getElementById('answer-provider');
    const answerModelInput = document.getElementById('answer-model');
    const answerBaseUrlInput = document.getElementById('answer-base-url');
    const answerApiKeyInput = document.getElementById('answer-api-key');
    const answerTemperatureInput = document.getElementById('answer-temperature');
    const answerTopPInput = document.getElementById('answer-top-p');
    const answerMaxTokensInput = document.getElementById('answer-max-tokens');
    const answerTimeoutInput = document.getElementById('answer-timeout');
    const answerSystemPromptTextarea = document.getElementById('answer-system-prompt');

    let selectorEnabled = false;
    let answerEnabled = false;
    let lastParams = null;
    let lastSelectionDocId = null;
    let lastQuery = null;

    const bm25Candidates = 200;
    const denseCandidates = 50;

    const ensureSelectOption = (selectEl, value) => {
      if (!value) return;
      const exists = Array.from(selectEl.options).some((option) => option.value === value);
      if (!exists) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }
    };

    const formatScore = (value) => Number.isFinite(value) ? value.toFixed(3) : '0.000';
    const escapeHtml = (value) => (value || '').replace(/&/g, '&amp;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

    const attachToggleHandlers = (container) => {
      container.querySelectorAll('.snippet').forEach((el) => {
        el.textContent = decodeURIComponent(el.dataset.snippet || '');
      });
      container.querySelectorAll('.toggle').forEach((btn) => {
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

    const resetSelection = () => {
      selectionDiv.style.display = 'none';
      selectionDiv.innerHTML = '';
      rerankedDiv.innerHTML = '';
      lastSelectionDocId = null;
      updateButtons();
    };

    const resetAnswerOutput = () => {
      answerOutput.style.display = 'none';
      answerMeta.textContent = '';
      answerText.textContent = '';
      answerChunksDiv.innerHTML = '';
    };

    const updateButtons = () => {
      rerankBtn.disabled = !selectorEnabled || !lastParams;
      answerBtn.disabled = !answerEnabled || !lastSelectionDocId || !lastQuery;
    };

    const refreshSelectorState = async () => {
      try {
        const response = await fetch('/selector-config');
        if (!response.ok) {
          throw new Error('Unable to fetch selector configuration');
        }
        const data = await response.json();
        selectorEnabled = Boolean(data.enabled);
        ensureSelectOption(selectorProviderSelect, data.provider);
        selectorProviderSelect.value = data.provider || '';
        selectorModelInput.value = data.model || '';
        selectorBaseUrlInput.value = data.base_url || '';
        selectorTemperatureInput.value = (data.temperature ?? '') === '' ? '' : data.temperature;
        selectorTopPInput.value = (data.top_p ?? '') === '' ? '' : data.top_p;
        selectorMaxTokensInput.value = (data.max_tokens ?? '') === '' ? '' : data.max_tokens;
        selectorTimeoutInput.value = (data.timeout ?? '') === '' ? '' : data.timeout;
        selectorCandidatesInput.value = (data.candidates ?? '') === '' ? '' : data.candidates;
        selectorSystemPromptTextarea.value = data.system_prompt || '';
        selectorStatus.textContent = selectorEnabled ? 'LLM selector ready' : 'LLM selector disabled';
        selectorError.textContent = '';
        selectorDisableBtn.disabled = !selectorEnabled;
      } catch (err) {
        selectorStatus.textContent = 'Selector status unknown';
        selectorEnabled = false;
        selectorDisableBtn.disabled = true;
      } finally {
        updateButtons();
      }
    };

    const refreshAnswerState = async () => {
      try {
        const response = await fetch('/answer-config');
        if (!response.ok) {
          throw new Error('Unable to fetch answer configuration');
        }
        const data = await response.json();
        answerEnabled = Boolean(data.enabled);
        ensureSelectOption(answerProviderSelect, data.provider);
        answerProviderSelect.value = data.provider || '';
        answerModelInput.value = data.model || '';
        answerBaseUrlInput.value = data.base_url || '';
        answerTemperatureInput.value = (data.temperature ?? '') === '' ? '' : data.temperature;
        answerTopPInput.value = (data.top_p ?? '') === '' ? '' : data.top_p;
        answerMaxTokensInput.value = (data.max_tokens ?? '') === '' ? '' : data.max_tokens;
        answerTimeoutInput.value = (data.timeout ?? '') === '' ? '' : data.timeout;
        answerSystemPromptTextarea.value = data.system_prompt || '';
        answerStatus.textContent = answerEnabled ? 'Answering LLM ready' : 'Answering LLM disabled';
        answerError.textContent = '';
        answerDisableBtn.disabled = !answerEnabled;
      } catch (err) {
        answerStatus.textContent = 'Answering status unknown';
        answerEnabled = false;
        answerDisableBtn.disabled = true;
      } finally {
        updateButtons();
      }
    };

    refreshSelectorState();
    refreshAnswerState();

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const query = queryInput.value.trim();
      if (!query) {
        return;
      }

      resultsDiv.innerHTML = '';
      errorDiv.textContent = '';
      submitBtn.disabled = true;
      submitBtn.textContent = 'Searching…';
      lastQuery = query;
      lastSelectionDocId = null;
      updateButtons();
      resetSelection();
      resetAnswerOutput();

      try {
        const response = await fetch('/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            top_k: parseInt(topkInput.value, 10) || 5,
            dense_weight: parseFloat(denseInput.value) || 0.6
          })
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || 'Search failed');
        }

        const payload = await response.json();
        if (!payload.results.length) {
          resultsDiv.innerHTML = '<p>No results found.</p>';
          lastParams = null;
          updateButtons();
          return;
        }

        resultsDiv.innerHTML = payload.results.map((hit) => {
          const sectionTitle = escapeHtml(hit.section_title || 'Untitled section');
          const drugTitle = escapeHtml(hit.drug_title || hit.doc_id || 'Unknown drug');
          const sourceLink = hit.source_url ? `· <a href='${encodeURI(hit.source_url)}' target='_blank' rel='noopener noreferrer'>source</a>` : '';
          const snippetAttr = encodeURIComponent(hit.snippet || '');
          const fullAttr = encodeURIComponent(hit.text || '');
          const needsMore = (hit.text || '').trim().length > (hit.snippet || '').trim().length;

          return `
            <article class='card'>
              <h2>#${hit.rank} · ${sectionTitle}</h2>
              <div class='meta'>
                <strong>${drugTitle}</strong> ${sourceLink}
              </div>
              <div class='scores'>Combined: ${formatScore(hit.score)} · Dense: ${formatScore(hit.dense_score)} (${Number.isFinite(hit.dense_raw) ? formatScore(hit.dense_raw) : '—'}) · BM25: ${formatScore(hit.bm25_score)} (${Number.isFinite(hit.bm25_raw) ? formatScore(hit.bm25_raw) : '—'})</div>
              <div class='snippet' data-snippet='${snippetAttr}' data-full='${fullAttr}' data-has-more='${needsMore}'></div>
              <button type='button' class='toggle' data-expanded='false'>Show more</button>
            </article>
          `;
        }).join('');

        attachToggleHandlers(resultsDiv);
        lastParams = {
          query,
          top_k: parseInt(topkInput.value, 10) || 5,
          dense_weight: parseFloat(denseInput.value) || 0.6,
          bm25_candidates: bm25Candidates,
          dense_candidates: denseCandidates
        };
      } catch (err) {
        console.error(err);
        errorDiv.textContent = err.message;
        lastParams = null;
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Search';
        updateButtons();
      }
    });

    rerankBtn.addEventListener('click', async () => {
      if (!selectorEnabled || !lastParams) {
        return;
      }

      const originalLabel = rerankBtn.textContent;
      rerankBtn.disabled = true;
      rerankBtn.textContent = 'Reranking…';
      errorDiv.textContent = '';
      resetSelection();
      resetAnswerOutput();

      try {
        const response = await fetch('/rerank', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(lastParams)
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || 'Rerank failed');
        }

        const payload = await response.json();
        const docLabel = escapeHtml(payload.selection.doc_id || 'No monograph selected');
        const reason = escapeHtml(payload.selection.reason || 'No reason provided');
        const confidence = (payload.selection.confidence ?? null) !== null
          ? Number(payload.selection.confidence).toFixed(2)
          : '—';

        selectionDiv.innerHTML = `
          <h2>LLM selection: ${docLabel}</h2>
          <p><strong>Confidence:</strong> ${confidence} · <strong>Reason:</strong> ${reason}</p>
        `;
        selectionDiv.style.display = 'block';

        rerankedDiv.innerHTML = payload.selected_results.map((hit) => {
          const sectionTitle = escapeHtml(hit.section_title || 'Untitled section');
          const drugTitle = escapeHtml(hit.drug_title || hit.doc_id || 'Unknown drug');
          const snippetAttr = encodeURIComponent(hit.snippet || '');
          const fullAttr = encodeURIComponent(hit.text || '');
          const needsMore = (hit.text || '').trim().length > (hit.snippet || '').trim().length;

          return `
            <article class='card'>
              <h2>#${hit.rank} · ${sectionTitle}</h2>
              <div class='meta'><strong>${drugTitle}</strong></div>
              <div class='scores'>Combined: ${formatScore(hit.score)} · Dense: ${formatScore(hit.dense_score)} (${Number.isFinite(hit.dense_raw) ? formatScore(hit.dense_raw) : '—'}) · BM25: ${formatScore(hit.bm25_score)} (${Number.isFinite(hit.bm25_raw) ? formatScore(hit.bm25_raw) : '—'})</div>
              <div class='snippet' data-snippet='${snippetAttr}' data-full='${fullAttr}' data-has-more='${needsMore}'></div>
              <button type='button' class='toggle' data-expanded='false'>Show more</button>
            </article>
          `;
        }).join('');

        attachToggleHandlers(rerankedDiv);
        lastSelectionDocId = payload.selection.doc_id || null;
      } catch (err) {
        console.error(err);
        errorDiv.textContent = err.message;
        lastSelectionDocId = null;
      } finally {
        rerankBtn.textContent = originalLabel;
        rerankBtn.disabled = !selectorEnabled || !lastParams;
        updateButtons();
      }
    });

    selectorForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      selectorError.textContent = '';

      const provider = selectorProviderSelect.value.trim();
      const model = selectorModelInput.value.trim();
      if (!provider || !model) {
        selectorError.textContent = 'Provider and model are required to enable the selector.';
        return;
      }

      const payload = { enabled: true, provider, model };
      if (selectorBaseUrlInput.value.trim()) payload.base_url = selectorBaseUrlInput.value.trim();
      if (selectorApiKeyInput.value.trim()) payload.api_key = selectorApiKeyInput.value.trim();
      const temperature = parseFloat(selectorTemperatureInput.value);
      if (!Number.isNaN(temperature)) payload.temperature = temperature;
      const topP = parseFloat(selectorTopPInput.value);
      if (!Number.isNaN(topP)) payload.top_p = topP;
      const maxTokens = parseInt(selectorMaxTokensInput.value, 10);
      if (!Number.isNaN(maxTokens)) payload.max_tokens = maxTokens;
      const timeout = parseFloat(selectorTimeoutInput.value);
      if (!Number.isNaN(timeout)) payload.timeout = timeout;
      const candidates = parseInt(selectorCandidatesInput.value, 10);
      if (!Number.isNaN(candidates)) payload.candidates = candidates;
      if (selectorSystemPromptTextarea.value.trim()) payload.system_prompt = selectorSystemPromptTextarea.value;

      ensureSelectOption(selectorProviderSelect, provider);

      selectorApplyBtn.disabled = true;
      selectorDisableBtn.disabled = true;
      const previousLabel = selectorApplyBtn.textContent;
      selectorApplyBtn.textContent = 'Applying…';

      try {
        const response = await fetch('/selector-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || 'Failed to configure selector');
        }
        await refreshSelectorState();
      } catch (err) {
        selectorError.textContent = err.message;
      } finally {
        selectorApplyBtn.disabled = false;
        selectorDisableBtn.disabled = !selectorEnabled;
        selectorApplyBtn.textContent = previousLabel;
        updateButtons();
      }
    });

    selectorDisableBtn.addEventListener('click', async () => {
      selectorError.textContent = '';
      selectorDisableBtn.disabled = true;
      selectorApplyBtn.disabled = true;
      const previousLabel = selectorDisableBtn.textContent;
      selectorDisableBtn.textContent = 'Disabling…';

      const payload = { enabled: false };
      const timeout = parseFloat(selectorTimeoutInput.value);
      if (!Number.isNaN(timeout)) payload.timeout = timeout;
      const candidates = parseInt(selectorCandidatesInput.value, 10);
      if (!Number.isNaN(candidates)) payload.candidates = candidates;

      try {
        const response = await fetch('/selector-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || 'Failed to disable selector');
        }
        await refreshSelectorState();
        resetSelection();
        resetAnswerOutput();
      } catch (err) {
        selectorError.textContent = err.message;
      } finally {
        selectorDisableBtn.textContent = previousLabel;
        selectorApplyBtn.disabled = false;
        selectorDisableBtn.disabled = !selectorEnabled;
        updateButtons();
      }
    });

    answerForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      answerError.textContent = '';

      const provider = answerProviderSelect.value.trim();
      const model = answerModelInput.value.trim();
      if (!provider || !model) {
        answerError.textContent = 'Provider and model are required to enable answering.';
        return;
      }

      const payload = { enabled: true, provider, model };
      if (answerBaseUrlInput.value.trim()) payload.base_url = answerBaseUrlInput.value.trim();
      if (answerApiKeyInput.value.trim()) payload.api_key = answerApiKeyInput.value.trim();
      const temperature = parseFloat(answerTemperatureInput.value);
      if (!Number.isNaN(temperature)) payload.temperature = temperature;
      const topP = parseFloat(answerTopPInput.value);
      if (!Number.isNaN(topP)) payload.top_p = topP;
      const maxTokens = parseInt(answerMaxTokensInput.value, 10);
      if (!Number.isNaN(maxTokens)) payload.max_tokens = maxTokens;
      const timeout = parseFloat(answerTimeoutInput.value);
      if (!Number.isNaN(timeout)) payload.timeout = timeout;
      if (answerSystemPromptTextarea.value.trim()) payload.system_prompt = answerSystemPromptTextarea.value;

      ensureSelectOption(answerProviderSelect, provider);

      answerApplyBtn.disabled = true;
      answerDisableBtn.disabled = true;
      const previousLabel = answerApplyBtn.textContent;
      answerApplyBtn.textContent = 'Applying…';

      try {
        const response = await fetch('/answer-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || 'Failed to configure answering LLM');
        }
        await refreshAnswerState();
      } catch (err) {
        answerError.textContent = err.message;
      } finally {
        answerApplyBtn.disabled = false;
        answerDisableBtn.disabled = !answerEnabled;
        answerApplyBtn.textContent = previousLabel;
        updateButtons();
      }
    });

    answerDisableBtn.addEventListener('click', async () => {
      answerError.textContent = '';
      answerDisableBtn.disabled = true;
      answerApplyBtn.disabled = true;
      const previousLabel = answerDisableBtn.textContent;
      answerDisableBtn.textContent = 'Disabling…';

      const payload = { enabled: false };
      const timeout = parseFloat(answerTimeoutInput.value);
      if (!Number.isNaN(timeout)) payload.timeout = timeout;

      try {
        const response = await fetch('/answer-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || 'Failed to disable answering LLM');
        }
        await refreshAnswerState();
        resetAnswerOutput();
      } catch (err) {
        answerError.textContent = err.message;
      } finally {
        answerDisableBtn.textContent = previousLabel;
        answerApplyBtn.disabled = false;
        answerDisableBtn.disabled = !answerEnabled;
        updateButtons();
      }
    });

    answerBtn.addEventListener('click', async () => {
      if (!answerEnabled || !lastSelectionDocId || !lastQuery) {
        return;
      }

      const originalLabel = answerBtn.textContent;
      answerBtn.disabled = true;
      answerBtn.textContent = 'Generating…';
      answerError.textContent = '';
      resetAnswerOutput();

      try {
        const response = await fetch('/answer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: lastQuery, doc_id: lastSelectionDocId })
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.detail || 'Failed to generate answer');
        }

        const payload = await response.json();
        answerMeta.textContent = `Monograph ${escapeHtml(payload.doc_id)} · chunks provided: ${payload.chunk_count}`;
        answerText.textContent = payload.answer || '';
        answerChunksDiv.innerHTML = payload.chunks.map((chunk) => `
          <details>
            <summary>#${chunk.rank} · ${escapeHtml(chunk.section_title || 'Untitled section')} (${escapeHtml(chunk.chunk_id)})</summary>
            <pre>${escapeHtml(chunk.text || '')}</pre>
          </details>
        `).join('');
        answerOutput.style.display = 'block';
      } catch (err) {
        console.error(err);
        answerError.textContent = err.message;
      } finally {
        answerBtn.textContent = originalLabel;
        answerBtn.disabled = !answerEnabled || !lastSelectionDocId || !lastQuery;
      }
    });
  </script>
</body>
</html>"""



def create_app(engine: HybridSearchEngine) -> FastAPI:
    app = FastAPI(title="Pharmacy Copilot Hybrid Search", version="1.0.0")
    app.state.engine = engine

    def _build_selector_response() -> SelectorConfigResponse:
        engine_state: HybridSearchEngine = app.state.engine
        config = engine_state.selector_config
        custom_prompt: Optional[str] = None
        if config and config.system_prompt and config.system_prompt != DEFAULT_SYSTEM_PROMPT:
            custom_prompt = config.system_prompt
        return SelectorConfigResponse(
            enabled=engine_state.selector_enabled,
            provider=config.provider if config else None,
            model=config.model if config else None,
            base_url=config.base_url if config else None,
            temperature=config.temperature if config else None,
            top_p=config.top_p if config else None,
            max_tokens=config.max_tokens if config else None,
            system_prompt=custom_prompt,
            timeout=engine_state.selector_timeout,
            candidates=engine_state.selector_candidates,
        )

    def _build_answer_response() -> AnswerConfigResponse:
        engine_state: HybridSearchEngine = app.state.engine
        config = engine_state.answer_config
        custom_prompt: Optional[str] = None
        if config and config.system_prompt and config.system_prompt != DEFAULT_SYSTEM_PROMPT:
            custom_prompt = config.system_prompt
        return AnswerConfigResponse(
            enabled=engine_state.answer_client is not None,
            provider=config.provider if config else None,
            model=config.model if config else None,
            base_url=config.base_url if config else None,
            temperature=config.temperature if config else None,
            top_p=config.top_p if config else None,
            max_tokens=config.max_tokens if config else None,
            system_prompt=custom_prompt,
            timeout=engine_state.answer_timeout,
        )

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

    @app.get("/selector-enabled")
    async def selector_enabled() -> JSONResponse:
        engine: HybridSearchEngine = app.state.engine
        return JSONResponse({"enabled": engine.selector_enabled})

    @app.get("/selector-config", response_model=SelectorConfigResponse)
    async def get_selector_config() -> SelectorConfigResponse:
        return _build_selector_response()

    @app.post("/selector-config", response_model=SelectorConfigResponse)
    async def set_selector_config(request: SelectorConfigRequest) -> SelectorConfigResponse:
        engine_state: HybridSearchEngine = app.state.engine
        try:
            if not request.enabled:
                await engine_state.update_selector(
                    None,
                    timeout=request.timeout if request.timeout is not None else engine_state.selector_timeout,
                    candidates=request.candidates if request.candidates is not None else engine_state.selector_candidates,
                )
                return _build_selector_response()

            provider = request.provider or os.getenv("SELECTOR_LLM_PROVIDER")
            model = request.model or os.getenv("SELECTOR_LLM_MODEL")
            if not provider or not model:
                raise HTTPException(status_code=400, detail="provider and model are required when enabling the selector")

            base_url = resolve_selector_base_url(
                provider,
                request.base_url or os.getenv("SELECTOR_LLM_BASE_URL"),
            )
            api_key = request.api_key or os.getenv("SELECTOR_LLM_API_KEY")
            temperature = request.temperature if request.temperature is not None else float(os.getenv("SELECTOR_LLM_TEMPERATURE", "0.0"))
            top_p = request.top_p if request.top_p is not None else float(os.getenv("SELECTOR_LLM_TOP_P", "0.3"))
            max_tokens = request.max_tokens if request.max_tokens is not None else int(os.getenv("SELECTOR_LLM_MAX_TOKENS", "256"))
            system_prompt = request.system_prompt or os.getenv("SELECTOR_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

            config = LLMConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            await engine_state.update_selector(
                config,
                timeout=request.timeout if request.timeout is not None else engine_state.selector_timeout,
                candidates=request.candidates if request.candidates is not None else engine_state.selector_candidates,
            )
            return _build_selector_response()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to configure selector: {exc}") from exc

    @app.get("/answer-config", response_model=AnswerConfigResponse)
    async def get_answer_config() -> AnswerConfigResponse:
        return _build_answer_response()

    @app.post("/answer-config", response_model=AnswerConfigResponse)
    async def set_answer_config(request: AnswerConfigRequest) -> AnswerConfigResponse:
        engine_state: HybridSearchEngine = app.state.engine
        try:
            if not request.enabled:
                await engine_state.update_answer(
                    None,
                    timeout=request.timeout if request.timeout is not None else engine_state.answer_timeout,
                )
                return _build_answer_response()

            provider = request.provider or os.getenv("ANSWER_LLM_PROVIDER") or os.getenv("EVAL_LLM_PROVIDER")
            model = request.model or os.getenv("ANSWER_LLM_MODEL") or os.getenv("EVAL_LLM_MODEL")
            if not provider or not model:
                raise HTTPException(status_code=400, detail="provider and model are required when enabling the answer LLM")

            base_url = resolve_selector_base_url(
                provider,
                request.base_url or os.getenv("ANSWER_LLM_BASE_URL") or os.getenv("EVAL_LLM_BASE_URL"),
            )
            api_key = request.api_key or os.getenv("ANSWER_LLM_API_KEY") or os.getenv("EVAL_LLM_API_KEY")
            temperature = request.temperature if request.temperature is not None else float(
                os.getenv("ANSWER_LLM_TEMPERATURE", os.getenv("EVAL_LLM_TEMPERATURE", "0.2"))
            )
            top_p = request.top_p if request.top_p is not None else float(
                os.getenv("ANSWER_LLM_TOP_P", os.getenv("EVAL_LLM_TOP_P", "0.3"))
            )
            max_tokens = request.max_tokens if request.max_tokens is not None else int(
                os.getenv("ANSWER_LLM_MAX_TOKENS", os.getenv("EVAL_LLM_MAX_TOKENS", "512"))
            )
            system_prompt = request.system_prompt or os.getenv("ANSWER_LLM_SYSTEM_PROMPT") or os.getenv("EVAL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

            config = LLMConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            await engine_state.update_answer(
                config,
                timeout=request.timeout if request.timeout is not None else engine_state.answer_timeout,
            )
            return _build_answer_response()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to configure answer LLM: {exc}") from exc

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(request: RerankRequest) -> RerankResponse:
        engine: HybridSearchEngine = app.state.engine
        if not engine.selector_enabled:
            raise HTTPException(status_code=503, detail="LLM selector not configured")
        selection, original_hits, selected_hits = await engine.rerank(
            request.query,
            top_k=request.top_k,
            dense_weight=request.dense_weight,
            bm25_candidates=request.bm25_candidates,
            dense_candidates=request.dense_candidates,
        )

        selection_payload = RerankSelection(
            doc_id=selection.doc_id,
            confidence=selection.confidence,
            reason=selection.reason,
            raw_response=selection.raw_response,
            prompt=selection.prompt,
        )
        return RerankResponse(
            query=request.query,
            selection=selection_payload,
            selected_results=selected_hits,
            original_results=original_hits,
        )

    @app.post("/answer", response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse:
        engine_state: HybridSearchEngine = app.state.engine
        if not engine_state.answer_client:
            raise HTTPException(status_code=503, detail="Answering LLM not configured")
        try:
            answer_text, monograph = await engine_state.answer_question(request.query, request.doc_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks: List[MonographChunk] = []
        for idx, item in enumerate(monograph):
            chunks.append(
                MonographChunk(
                    rank=int(item.get("rank") or idx + 1),
                    chunk_id=item.get("chunk_id") or f"{request.doc_id}::chunk::{idx+1}",
                    doc_id=item.get("doc_id") or request.doc_id,
                    section_title=item.get("section_title"),
                    section_code=item.get("section_code"),
                    drug_title=item.get("drug_title"),
                    snippet=item.get("snippet"),
                    text=item.get("text") or "",
                )
            )

        return AnswerResponse(
            doc_id=request.doc_id,
            chunk_count=len(chunks),
            answer=answer_text,
            chunks=chunks,
        )

    return app


@asynccontextmanager
async def lifespan(
    metadata_path: Path,
    bm25_path: Path,
    vectordb_path: Path,
    model: str,
    host: str,
    timeout: float,
    *,
    selector_config: Optional[LLMConfig] = None,
    selector_timeout: float = 60.0,
    selector_candidates: int = 30,
    answer_config: Optional[LLMConfig] = None,
    answer_timeout: float = 120.0,
):
    engine = await HybridSearchEngine.create(
        metadata_path=metadata_path,
        bm25_index_path=bm25_path,
        vectordb_path=vectordb_path,
        model=model,
        ollama_host=host,
        request_timeout=timeout,
        selector_config=selector_config,
        selector_timeout=selector_timeout,
        selector_candidates=selector_candidates,
        answer_config=answer_config,
        answer_timeout=answer_timeout,
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
    parser.add_argument("--selector-provider")
    parser.add_argument("--selector-model")
    parser.add_argument("--selector-base-url")
    parser.add_argument("--selector-api-key")
    parser.add_argument("--selector-temperature", type=float)
    parser.add_argument("--selector-top-p", type=float)
    parser.add_argument("--selector-max-tokens", type=int)
    parser.add_argument("--selector-system-prompt")
    parser.add_argument("--selector-timeout", type=float, default=float(os.getenv("SELECTOR_LLM_TIMEOUT", "60")))
    parser.add_argument("--selector-candidates", type=int, default=int(os.getenv("SELECTOR_LLM_CANDIDATES", "30")))
    parser.add_argument("--answer-provider")
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--answer-temperature", type=float)
    parser.add_argument("--answer-top-p", type=float)
    parser.add_argument("--answer-max-tokens", type=int)
    parser.add_argument("--answer-system-prompt")
    parser.add_argument("--answer-timeout", type=float, default=float(os.getenv("ANSWER_LLM_TIMEOUT", os.getenv("EVAL_LLM_TIMEOUT", "120"))))
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    selector_config = build_selector_config_from_args(args)
    answer_config = build_answer_config_from_args(args)

    engine = await HybridSearchEngine.create(
        metadata_path=args.metadata,
        bm25_index_path=args.bm25,
        vectordb_path=args.vectordb,
        model=args.model,
        ollama_host=args.ollama_host,
        request_timeout=args.timeout,
        selector_config=selector_config,
        selector_timeout=args.selector_timeout,
        selector_candidates=args.selector_candidates,
        answer_config=answer_config,
        answer_timeout=args.answer_timeout,
    )
    app = create_app(engine)

    config = uvicorn.Config(app, host=args.host, port=args.port, reload=args.reload)
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        await engine.close()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
