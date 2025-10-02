#!/usr/bin/env python3
"""CLI wrapper for the LLM judge."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.judge_results import execute_judge


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM judge on evaluation outputs")
    parser.add_argument("results", help="Path to evaluation results JSON")
    parser.add_argument("--dataset", default="eval/datasets/pharmacist_eval_v1.json")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--llm-provider", default=os.getenv("EVAL_LLM_PROVIDER"))
    parser.add_argument("--llm-model", default=os.getenv("EVAL_LLM_MODEL"))
    parser.add_argument("--llm-base-url", default=os.getenv("EVAL_LLM_BASE_URL"))
    parser.add_argument("--llm-api-key", default=os.getenv("EVAL_LLM_API_KEY"))

    args = parser.parse_args()

    return execute_judge(
        Path(args.results),
        Path(args.dataset),
        Path(args.output) if args.output else None,
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
