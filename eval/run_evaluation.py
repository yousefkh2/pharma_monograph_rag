#!/usr/bin/env python3
"""End-to-end evaluation harness for pharmacy copilot system."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.dataset_schema import EvalDataset, EvalQuestion
from eval.retrieval_metrics import RetrievalResult, evaluate_retrieval, aggregate_retrieval_metrics, format_retrieval_metrics
from eval.qa_metrics import QAEvaluator, aggregate_qa_metrics, format_qa_metrics
from eval.failure_analysis import FailureAnalyzer, generate_failure_report
from eval.serialization import to_serializable
from eval.llm_client import LLMClient, config_from_args


class PharmacyCopilotEvaluator:
    """End-to-end evaluator for the pharmacy copilot system."""
    
    def __init__(
        self,
        service_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        top_k_retrieval: int = 10,
        llm_client: Optional[LLMClient] = None,
    ):
        self.service_url = service_url
        self.timeout = timeout
        self.top_k_retrieval = top_k_retrieval
        self.llm_client = llm_client
        
        # Initialize evaluators
        self.qa_evaluator = QAEvaluator()
        self.failure_analyzer = FailureAnalyzer()
        
        # HTTP client for service calls
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _call_search_service(self, query: str) -> Dict:
        """Call the hybrid search service."""
        try:
            response = await self.client.post(
                f"{self.service_url}/search",
                json={
                    "query": query,
                    "top_k": self.top_k_retrieval,
                    "dense_weight": 0.6,  # Default hybrid weights
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling search service: {e}")
            return {"results": []}
    
    def _generate_answer_stub(self, query: str, search_results: List[Dict]) -> str:
        """Generate answer from search results (STUB IMPLEMENTATION).
        
        This is a placeholder that concatenates snippets. In the real system,
        this would be replaced with an LLM-based answer generation.
        """
        if not search_results:
            return "I don't have enough information to answer that question."
        
        # Simple concatenation of top snippets (stub behavior)
        snippets = []
        for result in search_results[:3]:  # Use top 3 results
            snippet = result.get("snippet", "")
            if snippet and snippet not in snippets:
                snippets.append(snippet)
        
        if not snippets:
            return "I don't have enough information to answer that question."
        
        # Basic answer generation (replace with LLM)
        answer = "Based on the available information: " + " ".join(snippets)
        
        # Truncate if too long
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        return answer
    
    async def _generate_answer(self, question: EvalQuestion, search_results: List[Dict]) -> str:
        if self.llm_client:
            return await self.llm_client.generate_answer(
                question=question.question,
                contexts=search_results,
                key_points=question.answer_key_points,
            )
        return self._generate_answer_stub(question.question, search_results)

    async def evaluate_question(self, question: EvalQuestion) -> Dict:
        """Evaluate a single question end-to-end."""
        print(f"Evaluating question {question.id}: {question.question[:50]}...")
        
        # Call search service
        search_response = await self._call_search_service(question.question)
        search_results = search_response.get("results", [])
        
        # Convert to RetrievalResult format
        retrieved = [
            RetrievalResult(
                chunk_id=result.get("chunk_id", f"chunk_{idx+1}"),
                score=result.get("score", 0.0),
                rank=result.get("rank", idx + 1),
            )
            for idx, result in enumerate(search_results)
        ]
        
        generated_answer = await self._generate_answer(question, search_results)
        
        # Evaluate retrieval
        retrieval_metrics = evaluate_retrieval(question, retrieved)
        
        # Evaluate QA
        qa_metrics = self.qa_evaluator.evaluate_answer(question, generated_answer)
        
        # Analyze failures
        failure_case = self.failure_analyzer.analyze_failure(
            question, retrieved, generated_answer, retrieval_metrics, qa_metrics
        )
        
        return {
            "question_id": question.id,
            "question": question.question,
            "generated_answer": generated_answer,
            "expected_answer": question.expected_answer,
            "retrieval_metrics": retrieval_metrics,
            "qa_metrics": qa_metrics,
            "retrieval_results": retrieved,
            "context_chunks": search_results[:6],
            "failure_case": failure_case if failure_case.failure_types else None,
            "question_dataclass": question,
        }
    
    async def evaluate_dataset(self, dataset: EvalDataset) -> Dict:
        """Evaluate the entire dataset."""
        print(f"Evaluating dataset: {dataset.name} ({len(dataset.questions)} questions)")
        
        results = []
        retrieval_metrics_list = []
        qa_metrics_list = []
        failure_cases = []
        
        start_time = time.time()
        
        for question in dataset.questions:
            result = await self.evaluate_question(question)
            results.append(result)
            
            retrieval_metrics_list.append(result["retrieval_metrics"])
            qa_metrics_list.append(result["qa_metrics"])
            
            if result["failure_case"]:
                failure_cases.append(result["failure_case"])
        
        end_time = time.time()
        
        # Aggregate metrics
        aggregated_retrieval = aggregate_retrieval_metrics(retrieval_metrics_list)
        aggregated_qa = aggregate_qa_metrics(qa_metrics_list)
        
        return {
            "dataset_name": dataset.name,
            "total_questions": len(dataset.questions),
            "evaluation_time": end_time - start_time,
            "aggregated_retrieval_metrics": aggregated_retrieval,
            "aggregated_qa_metrics": aggregated_qa,
            "failure_cases": failure_cases,
            "detailed_results": results,
        }
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate a comprehensive evaluation report."""
        lines = [
            f"# Pharmacy Copilot Evaluation Report",
            f"",
            f"**Dataset**: {evaluation_results['dataset_name']}",
            f"**Questions**: {evaluation_results['total_questions']}",
            f"**Evaluation Time**: {evaluation_results['evaluation_time']:.2f} seconds",
            f"**Average Time per Question**: {evaluation_results['evaluation_time'] / evaluation_results['total_questions']:.2f} seconds",
            f"",
        ]
        
        # Add retrieval metrics
        retrieval_report = format_retrieval_metrics(
            evaluation_results['aggregated_retrieval_metrics'],
            "Aggregated Retrieval Metrics"
        )
        lines.append(retrieval_report)
        lines.append("")
        
        # Add QA metrics
        qa_report = format_qa_metrics(
            evaluation_results['aggregated_qa_metrics'],
            "Aggregated QA Metrics"
        )
        lines.append(qa_report)
        lines.append("")
        
        # Add failure analysis
        failure_report = generate_failure_report(evaluation_results['failure_cases'])
        lines.append(failure_report)
        
        return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate pharmacy copilot system")
    parser.add_argument("dataset", type=Path, help="Path to evaluation dataset JSON file")
    parser.add_argument("--service-url", default="http://localhost:8001", help="URL of the search service")
    parser.add_argument("--output", type=Path, help="(Deprecated) Alias for --json-output")
    parser.add_argument("--json-output", type=Path, help="Output file for detailed results (JSON)")
    parser.add_argument("--report", type=Path, help="Output file for human-readable report (Markdown)")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "30.0")), help="Request timeout in seconds")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("EVAL_TOP_K", "10")), help="Number of chunks to retrieve")
    parser.add_argument("--use-llm", action="store_true", help="Generate answers with a configured LLM instead of snippet concatenation")
    parser.add_argument("--llm-provider", help="LLM provider identifier (ollama, openai, etc.)")
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--llm-base-url", help="Base URL for the LLM API endpoint")
    parser.add_argument("--llm-api-key", help="API key for hosted LLM providers")
    parser.add_argument("--llm-temperature", type=float, help="Sampling temperature for LLM generation")
    parser.add_argument("--llm-max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--llm-system-prompt", help="Override default system prompt for the LLM")

    args = parser.parse_args()

    # Load dataset
    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
    
    try:
        dataset = EvalDataset.load(args.dataset)
        print(f"Loaded dataset: {dataset.name} with {len(dataset.questions)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Check if service is available
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{args.service_url}/health", timeout=5.0)
            response.raise_for_status()
            print(f"✅ Service is available at {args.service_url}")
    except Exception as e:
        print(f"❌ Service is not available at {args.service_url}: {e}")
        return 1
    
    if args.output and not args.json_output:
        print("⚠️ --output is deprecated; please use --json-output going forward.")
    json_output_path = args.json_output or args.output

    async with AsyncExitStack() as stack:
        llm_client: Optional[LLMClient] = None
        if args.use_llm:
            llm_config = config_from_args(args)
            llm_client = LLMClient(llm_config, timeout=args.timeout)
            await stack.enter_async_context(llm_client)
            print(f"🤖 Using LLM provider '{llm_config.provider}' with model '{llm_config.model}'")

        evaluator = PharmacyCopilotEvaluator(
            service_url=args.service_url,
            timeout=args.timeout,
            top_k_retrieval=args.top_k,
            llm_client=llm_client,
        )
        await stack.enter_async_context(evaluator)

        evaluation_results = await evaluator.evaluate_dataset(dataset)

        report = evaluator.generate_report(evaluation_results)
        print("\n" + "="*80)
        print(report)
        print("="*80)

        if json_output_path:
            payload = {
                "dataset": {
                    "name": dataset.name,
                    "description": dataset.description,
                    "version": dataset.version,
                },
                "metadata": {
                    "question_count": len(dataset.questions),
                    "evaluation_time": evaluation_results["evaluation_time"],
                },
                "aggregated_metrics": {
                    "retrieval": to_serializable(evaluation_results["aggregated_retrieval_metrics"]),
                    "qa": to_serializable(evaluation_results["aggregated_qa_metrics"]),
                },
                "failure_cases": [to_serializable(case) for case in evaluation_results["failure_cases"]],
                "questions": [],
            }

            for question, result in zip(dataset.questions, evaluation_results["detailed_results"]):
                question_payload = result.get("question_dataclass", question)
                payload["questions"].append(
                    {
                        "question_id": question.id,
                        "question": to_serializable(question_payload),
                        "generated_answer": result["generated_answer"],
                        "expected_answer": result["expected_answer"],
                        "retrieval_metrics": to_serializable(result["retrieval_metrics"]),
                        "qa_metrics": to_serializable(result["qa_metrics"]),
                        "retrieval_results": [to_serializable(r) for r in result["retrieval_results"]],
                        "context_chunks": result.get("context_chunks", []),
                        "failure_analysis": to_serializable(result["failure_case"]) if result["failure_case"] else None,
                    }
                )

            json_output_path.parent.mkdir(parents=True, exist_ok=True)
            with json_output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"💾 Detailed results saved to: {json_output_path}")

        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("w", encoding="utf-8") as f:
                f.write(report)
            print(f"📄 Report saved to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
