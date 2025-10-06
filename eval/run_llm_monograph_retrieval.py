#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.dataset_schema import EvalDataset
from eval.retrieval_metrics import RetrievalResult, aggregate_retrieval_metrics, evaluate_retrieval, format_retrieval_metrics
from eval.serialization import to_serializable
from eval.llm_client import DEFAULT_SYSTEM_PROMPT, LLMClient, LLMConfig
from retrieval.bm25_index import BM25Index, default_tokenize
from retrieval.llm_monograph_selector import LLMMonographSelector, MonographCatalog, Monograph
from retrieval.search_utils import BM25SearchEngine, load_metadata


def _build_selector_config(args) -> LLMConfig:
    provider = args.selector_provider or os.getenv("SELECTOR_LLM_PROVIDER") or "ollama"
    model = args.selector_model or os.getenv("SELECTOR_LLM_MODEL") or "pharmacy-copilot"
    base_url = args.selector_base_url or os.getenv("SELECTOR_LLM_BASE_URL")
    if not base_url:
        base_url = "https://api.openai.com/v1" if provider == "openai" else (
            "https://openrouter.ai/api/v1" if provider == "openrouter" else "http://localhost:11434"
        )
    api_key = args.selector_api_key or os.getenv("SELECTOR_LLM_API_KEY")
    temperature = args.selector_temperature if args.selector_temperature is not None else float(os.getenv("SELECTOR_LLM_TEMPERATURE", "0.0"))
    top_p = args.selector_top_p if args.selector_top_p is not None else float(os.getenv("SELECTOR_LLM_TOP_P", "0.3"))
    max_tokens = args.selector_max_tokens if args.selector_max_tokens is not None else int(os.getenv("SELECTOR_LLM_MAX_TOKENS", "256"))
    system_prompt = args.selector_system_prompt or os.getenv(
        "SELECTOR_LLM_SYSTEM_PROMPT",
        "You are an expert Lexicomp librarian who selects the most relevant monograph by doc_id.",
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


def _build_answer_config(args) -> Optional[LLMConfig]:
    if not args.generate_answers:
        return None
    provider = args.answer_provider or os.getenv("EVAL_LLM_PROVIDER")
    model = args.answer_model or os.getenv("EVAL_LLM_MODEL")
    if not provider or not model:
        raise ValueError("Answer generation requested but provider/model not supplied")
    base_url = args.answer_base_url or os.getenv("EVAL_LLM_BASE_URL")
    if not base_url:
        base_url = "https://api.openai.com/v1" if provider == "openai" else (
            "https://openrouter.ai/api/v1" if provider == "openrouter" else "http://localhost:11434"
        )
    api_key = args.answer_api_key or os.getenv("EVAL_LLM_API_KEY")
    temperature = args.answer_temperature if args.answer_temperature is not None else float(os.getenv("EVAL_LLM_TEMPERATURE", "0.2"))
    top_p = args.answer_top_p if args.answer_top_p is not None else float(os.getenv("EVAL_LLM_TOP_P", "0.3"))
    max_tokens = args.answer_max_tokens if args.answer_max_tokens is not None else int(os.getenv("EVAL_LLM_MAX_TOKENS", "256"))
    system_prompt = args.answer_system_prompt or os.getenv("EVAL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
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
    return config


async def evaluate_dataset(args) -> dict:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    dataset = EvalDataset.load(args.dataset)

    metadata_path = args.metadata if args.metadata else Path("indexes/chunk_metadata.jsonl")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata_records = load_metadata(metadata_path)
    catalog = MonographCatalog.from_metadata(metadata_records)

    bm25_engine: Optional[BM25SearchEngine] = None
    if args.bm25_index:
        index_path = args.bm25_index
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
        bm25_index = BM25Index.load(index_path, tokenize=default_tokenize)
        bm25_engine = BM25SearchEngine(bm25_index, metadata_records)


    selector_config = _build_selector_config(args)
    answer_config = _build_answer_config(args)

    results: List[dict] = []
    retrieval_metrics_list = []

    async with AsyncExitStack() as stack:
        selector_client = LLMClient(selector_config, timeout=args.selector_timeout)
        await stack.enter_async_context(selector_client)
        selector = LLMMonographSelector(
            catalog,
            selector_client,
            max_catalog_items=args.selector_candidates,
        )

        answer_client: Optional[LLMClient] = None
        if answer_config:
            answer_client = LLMClient(answer_config, timeout=args.answer_timeout)
            await stack.enter_async_context(answer_client)

        for question in dataset.questions:
            candidate_monographs: List[Monograph] = []
            snippet_map: Dict[str, str] = {}

            if bm25_engine:
                hits, _ = bm25_engine.search(
                    question.question,
                    top_k=args.selector_candidates * 2,
                    retrieve_k=args.selector_candidates * 4,
                    include_text=False,
                )
                for hit in hits:
                    doc_id = hit.get("doc_id")
                    if not doc_id or doc_id in snippet_map:
                        continue
                    mono = catalog.get(doc_id)
                    if not mono:
                        continue
                    title = hit.get("drug_title") or mono.display_name
                    snippet_map[doc_id] = title.strip()
                    candidate_monographs.append(mono)
                    if len(candidate_monographs) >= args.selector_candidates:
                        break

            if not candidate_monographs:
                tokens = default_tokenize(question.question)
                candidate_monographs = catalog.top_k_by_token_overlap(
                    tokens,
                    args.selector_candidates,
                )

            if not candidate_monographs:
                candidate_monographs = list(catalog)[: args.selector_candidates]

            if not snippet_map:
                snippet_map = {mono.doc_id: mono.display_name for mono in candidate_monographs}

            selection = await selector.select(
                question.question,
                candidates=candidate_monographs,
                snippet_map=snippet_map,
            )
            contexts = catalog.build_context(selection.doc_id, include_text=not args.strip_text) if selection.doc_id else []
            if selection.doc_id and args.top_chunk_limit:
                contexts = contexts[: args.top_chunk_limit]

            retrieval_results: List[RetrievalResult] = []
            for idx, ctx in enumerate(contexts, start=1):
                retrieval_results.append(
                    RetrievalResult(
                        chunk_id=ctx.get("chunk_id", f"{selection.doc_id}::chunk::{idx}"),
                        score=1.0 / idx,
                        rank=idx,
                    )
                )

            metrics = evaluate_retrieval(question, retrieval_results)
            retrieval_metrics_list.append(metrics)

            generated_answer: Optional[str] = None
            if answer_client and contexts:
                generated_answer = await answer_client.generate_answer(
                    question=question.question,
                    contexts=contexts,
                    key_points=question.answer_key_points,
                    allowed_chunk_ids=[ctx["chunk_id"] for ctx in contexts if ctx.get("chunk_id")],
                )

            results.append(
                {
                    "question_id": question.id,
                    "question": question.question,
                    "selection": to_serializable(selection),
                    "retrieval_metrics": to_serializable(metrics),
                    "retrieval_results": [to_serializable(item) for item in retrieval_results],
                    "contexts": contexts,
                    "generated_answer": generated_answer,
                }
            )

    aggregated = aggregate_retrieval_metrics(retrieval_metrics_list)
    return {
        "dataset": {
            "name": dataset.name,
            "description": dataset.description,
            "version": dataset.version,
            "question_count": len(dataset.questions),
        },
        "selector": {
            "provider": selector_config.provider,
            "model": selector_config.model,
        },
        "aggregated_retrieval_metrics": to_serializable(aggregated),
        "results": results,
        "report": format_retrieval_metrics(aggregated, "LLM Monograph Selector"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using an LLM monograph selector")
    parser.add_argument("dataset", type=Path, help="Evaluation dataset JSON")
    parser.add_argument("--metadata", type=Path, default=Path("indexes/chunk_metadata.jsonl"), help="Chunk metadata JSONL")
    parser.add_argument("--bm25-index", type=Path, help="Path to BM25 index for lexical pre-filtering")
    parser.add_argument("--output", type=Path, help="Where to write JSON results")
    parser.add_argument("--top-chunk-limit", type=int, help="Limit number of chunks per monograph")
    parser.add_argument("--strip-text", action="store_true", help="Exclude full chunk text from saved contexts")
    parser.add_argument("--generate-answers", action="store_true", help="Generate answers with an LLM using retrieved monograph")
    parser.add_argument("--selector-candidates", type=int, default=30, help="Number of candidate monographs to include in selector prompt")

    parser.add_argument("--selector-provider")
    parser.add_argument("--selector-model")
    parser.add_argument("--selector-base-url")
    parser.add_argument("--selector-api-key")
    parser.add_argument("--selector-temperature", type=float)
    parser.add_argument("--selector-top-p", type=float)
    parser.add_argument("--selector-max-tokens", type=int)
    parser.add_argument("--selector-system-prompt")
    parser.add_argument("--selector-timeout", type=float, default=float(os.getenv("SELECTOR_LLM_TIMEOUT", "60")))

    parser.add_argument("--answer-provider")
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-api-key")
    parser.add_argument("--answer-temperature", type=float)
    parser.add_argument("--answer-top-p", type=float)
    parser.add_argument("--answer-max-tokens", type=int)
    parser.add_argument("--answer-system-prompt")
    parser.add_argument("--answer-timeout", type=float, default=float(os.getenv("EVAL_LLM_TIMEOUT", "120")))

    args = parser.parse_args()

    try:
        payload = await evaluate_dataset(args)
    except Exception as exc:
        print(f"Error during evaluation: {exc}")
        return 1

    print("\n" + payload["report"] + "\n")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
