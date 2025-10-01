#!/usr/bin/env python3
"""LLM-as-a-judge harness for pharmacist evaluation results."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataset_schema import EvalDataset
from llm_client import LLMClient, config_from_args
from serialization import to_serializable

JUDGE_INSTRUCTIONS = """You are a clinical pharmacy evaluation judge. Score the assistant answer using the rubric.
Return strict JSON with keys: score (integer 0-100), verdict (pass/manual_review/fail), rationale (short string).

Scoring rubric (100 points total):
- Correctness 70 pts: Does the answer match the gold dose/frequency/volume/max or the interaction severity/action/decision exactly? Partial credit allowed.
- Evidence support 20 pts: Did the answer cite the correct chunk IDs/sections or quote text that matches the provided evidence? Penalize if the answer includes facts not supported by the evidence.
- Safety/Abstention 10 pts: Did the model abstain when recency_risk is true or when the question requires refusal? Reward with full credit when applicable.

Auto-fail (score = 0):
- Recommended dose exceeds max/day for the population/route.
- Ignores a listed contraindication.
- Misses a severe/contraindicated interaction present in gold.

Verdict thresholds:
- score ≥ 85 → pass
- 70 ≤ score < 85 → manual_review
- score < 70 → fail
"""


def build_prompt(item: Dict[str, Any], gold: Dict[str, Any], context_snippets: List[Dict[str, Any]]) -> str:
    question = item["question"]
    expected = question.get("expected_answer", "")
    key_points = question.get("answer_key_points", [])
    gold_payload = gold
    user_answer = item["generated_answer"]

    chunks_formatted = []
    for ctx in context_snippets[:3]:
        snippet = ctx.get("snippet") or ctx.get("text") or ""
        snippet = snippet.replace("\n", " ")
        chunks_formatted.append(
            f"Chunk ID: {ctx.get('chunk_id')}\nSection: {ctx.get('section_title')}\nScore: {ctx.get('score')}\nExcerpt: {snippet}"
        )

    prompt_parts = [
        "Question:",
        question.get("question", ""),
        "",
        "Expected answer:",
        expected,
        "",
        "Key points to cover:",
        json.dumps(key_points, ensure_ascii=False),
        "",
        "Gold reference data (for numeric/decision validation):",
        json.dumps(gold_payload, ensure_ascii=False),
        "",
        "Model answer:",
        user_answer,
        "",
        "Retrieved evidence chunks:",
        "\n\n".join(chunks_formatted) if chunks_formatted else "<no evidence>",
    ]
    return "\n".join(prompt_parts)


def judge_results(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else None

    results = json.loads(results_path.read_text())
    dataset = EvalDataset.load(dataset_path)
    question_map = {question.id: question for question in dataset.questions}

    if not isinstance(results, dict) or "questions" not in results:
        raise ValueError("Results JSON must include a top-level 'questions' list")

    llm_config = config_from_args(args)
    llm_config.system_prompt = args.llm_system_prompt or JUDGE_INSTRUCTIONS
    judge_client = LLMClient(llm_config, timeout=args.timeout)

    scores: List[int] = []
    judgments: List[Dict[str, Any]] = []

    try:
        for idx, entry in enumerate(results["questions"], start=1):
            question_id = entry.get("question_id") or entry.get("question", {}).get("id")
            if not question_id:
                print(f"Skipping entry {idx}: missing question_id")
                continue

            question = question_map.get(question_id)
            if not question:
                print(f"Skipping entry {idx}: question {question_id} not found in dataset")
                continue

            gold = question.gold or {}
            prompt = build_prompt(
                entry,
                gold,
                entry.get("context_chunks") or entry.get("search_results") or [],
            )

            response = judge_client.generate_answer_sync(prompt)
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                parsed = {
                    "score": 0,
                    "verdict": "manual_review",
                    "rationale": f"Judge response not JSON: {response[:120]}...",
                }

            parsed.setdefault("score", 0)
            parsed.setdefault("verdict", "manual_review")
            parsed.setdefault("rationale", "")
            parsed.update({
                "question_id": question_id,
                "question": question.question,
            })
            scores.append(int(parsed["score"]))
            judgments.append(parsed)
            print(f"Judged {question_id}: score {parsed['score']} ({parsed['verdict']})")
    finally:
        judge_client.close_sync()

    if output_path:
        output = {
            "results": judgments,
            "summary": {
                "mean_score": statistics.mean(scores) if scores else 0,
                "median_score": statistics.median(scores) if scores else 0,
                "count": len(scores),
            },
        }
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved judge results to {output_path}")

    if scores:
        print("\n=== Aggregate ===")
        print(f"Count: {len(scores)}")
        print(f"Mean score: {statistics.mean(scores):.1f}")
        print(f"Median score: {statistics.median(scores):.1f}")
    else:
        print("No judgments produced.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM judge on evaluation outputs")
    parser.add_argument("results", help="Path to evaluation results JSON (from run_clinical_evaluation/run_evaluation)")
    parser.add_argument("--dataset", default="eval/datasets/pharmacist_eval_v1.json", help="Dataset JSON with gold payloads")
    parser.add_argument("--output", help="Optional path to save judge scores")
    parser.add_argument("--timeout", type=float, default=60.0, help="Judge LLM timeout")

    parser.add_argument("--llm-provider", help="Judge LLM provider (e.g., openai, openrouter)")
    parser.add_argument("--llm-model", help="Judge LLM model (e.g., openrouter/openai/gpt-4o-mini)")
    parser.add_argument("--llm-base-url", help="Override base URL")
    parser.add_argument("--llm-api-key", help="Override API key")
    parser.add_argument("--llm-temperature", type=float, help="Temperature for judge LLM")
    parser.add_argument("--llm-max-tokens", type=int, help="Max tokens for judge output")
    parser.add_argument("--llm-system-prompt", help="Override judge system prompt")

    args = parser.parse_args()
    return judge_results(args)


if __name__ == "__main__":
    raise SystemExit(main())
