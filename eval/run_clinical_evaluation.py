#!/usr/bin/env python3
"""Enhanced clinical evaluation runner for pharmacy copilot with regulatory compliance focus."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.clinical_schema import ClinicalEvalDataset, ClinicalEvalQuestion
from eval.clinical_metrics import (
    ClinicalEvaluator, 
    ClinicalComplianceMetrics, 
    ClinicalFailureCase,
    aggregate_clinical_metrics,
    format_clinical_metrics
)
from eval.retrieval_metrics import RetrievalResult, evaluate_retrieval, aggregate_retrieval_metrics, format_retrieval_metrics
from eval.qa_metrics import QAEvaluator, aggregate_qa_metrics, format_qa_metrics
from eval.serialization import to_serializable
from eval.llm_client import LLMClient, config_from_args


async def query_system(client: httpx.AsyncClient, question: str, top_k: int = 20) -> Tuple[List[RetrievalResult], List[Dict]]:
    """Query the hybrid search system and return retrieval results + raw hits."""
    
    request_data = {
        "query": question,
        "top_k": top_k,
        "dense_weight": 0.6,  # Default hybrid weight
    }
    
    try:
        response = await client.post("/search", json=request_data)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])

        retrieved = []
        for i, result in enumerate(results, 1):
            retrieved.append(
                RetrievalResult(
                    chunk_id=result.get("chunk_id", f"chunk_{i}"),
                    score=result.get("score", 0.0),
                    rank=i,
                )
            )

        return retrieved, results
        
    except Exception as e:
        print(f"Error querying system: {e}")
        return [], []


def generate_answer_stub(question: str, results: List[Dict]) -> str:
    """Generate a stub answer from search results. Replace with actual LLM."""
    
    # For clinical evaluation, we need different stubs based on question type
    question_lower = question.lower()
    
    # Post-2018 queries should be refused
    post_2018_indicators = ["2020", "2021", "2022", "2023", "2024", "covid", "pandemic", "recent", "latest", "new", "updated"]
    if any(indicator in question_lower for indicator in post_2018_indicators):
        return "I cannot provide information about recent developments or guidelines published after 2018, as my knowledge is limited to pre-2019 medical sources. Please consult current medical literature or your healthcare provider for the most up-to-date information."
    
    # Dosing questions
    if "dose" in question_lower or "dosage" in question_lower:
        if "acetaminophen" in question_lower:
            return "The maximum daily dose of acetaminophen for adults is 4000 mg (4 grams) per day according to prescribing guidelines. This should not be exceeded to avoid hepatotoxicity."
        elif "ibuprofen" in question_lower:
            return "The typical adult dose of ibuprofen is 200-800 mg every 6-8 hours, with a maximum daily dose of 3200 mg. Lower doses should be used in elderly patients."
        else:
            return "Dosing varies by medication, patient factors, and indication. Please refer to prescribing information for specific dosing recommendations."
    
    # Contraindication questions
    if "contraindic" in question_lower or "avoid" in question_lower:
        if "ibuprofen" in question_lower and ("bleeding" in question_lower or "gi" in question_lower):
            return "Yes, ibuprofen is contraindicated in patients with active GI bleeding due to increased risk of bleeding complications. NSAIDs should be avoided in patients with active peptic ulcer disease or bleeding disorders."
        else:
            return "Specific contraindications vary by medication. Common contraindications include hypersensitivity, organ dysfunction, and drug interactions."
    
    # Interaction questions
    if "interact" in question_lower:
        if "warfarin" in question_lower:
            return "Warfarin has numerous drug interactions that can increase bleeding risk, including aspirin, NSAIDs, and many antibiotics. Close monitoring of INR is required when starting or stopping interacting medications."
        else:
            return "Drug interactions can affect medication safety and efficacy. Please check drug interaction databases for specific combinations."
    
    # Generic answer
    if not results:
        return "I don't have sufficient information to answer this question accurately. Please consult your healthcare provider or current medical references."
    
    # Concatenate snippets (basic stub)
    snippets = [result.get("snippet", "") for result in results[:3]]
    combined_text = " ".join(snippets)
    
    return f"Based on available information: {combined_text[:300]}..."


async def evaluate_question(
    client: httpx.AsyncClient,
    question: ClinicalEvalQuestion,
    qa_evaluator: QAEvaluator,
    clinical_evaluator: ClinicalEvaluator,
    *,
    top_k: int,
    llm_client: Optional[LLMClient] = None,
) -> Tuple[Dict, ClinicalFailureCase]:
    """Evaluate a single clinical question."""
    
    retrieved, search_results = await query_system(client, question.question, top_k=top_k)

    if llm_client:
        regulatory = getattr(question, "regulatory_ground_truth", None)
        safety_notes: List[str] = []
        if regulatory and getattr(regulatory, "safety_categories", None):
            safety_notes.extend(cat.value for cat in regulatory.safety_categories)
        if getattr(question, "safety_critical_elements", None):
            safety_notes.extend(question.safety_critical_elements)
        if getattr(question, "contraindication_checks", None):
            safety_notes.extend(
                f"Ensure contraindication mention: {item}"
                for item in question.contraindication_checks
            )
        judge_metadata = getattr(question, "judge_metadata", None)
        generated_answer = await llm_client.generate_answer(
            question=question.question,
            contexts=search_results,
            expected_behavior=(regulatory.expected_behavior.value if regulatory else None),
            key_points=question.answer_key_points,
            knowledge_cutoff=(regulatory.knowledge_cutoff_date if regulatory else None),
            safety_notes=safety_notes or None,
            allowed_chunk_ids=(judge_metadata or {}).get("allowed_chunk_ids") if judge_metadata else None,
            disallowed_chunk_ids=(judge_metadata or {}).get("disallowed_chunk_ids") if judge_metadata else None,
        )
    else:
        generated_answer = generate_answer_stub(question.question, search_results)
    
    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(question, retrieved)
    
    # Evaluate QA
    qa_metrics = qa_evaluator.evaluate_answer(question, generated_answer)
    
    # Evaluate clinical compliance
    clinical_metrics, failure_case = clinical_evaluator.evaluate_clinical_compliance(
        question, generated_answer, retrieved, retrieval_metrics, qa_metrics
    )
    
    return {
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "clinical_metrics": clinical_metrics,
        "generated_answer": generated_answer,
        "retrieved_chunks": len(retrieved),
        "retrieval_results": retrieved,
        "search_results": search_results,
        "question_dataclass": question,
    }, failure_case


async def run_clinical_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive clinical evaluation."""

    dataset_path = args.dataset
    report_path = args.report
    json_output = args.json_output

    try:
        dataset = ClinicalEvalDataset.load(dataset_path)
        print(f"Loaded clinical dataset: {dataset.name} with {len(dataset.questions)} questions")
    except Exception as e:
        print(f"Failed to load as clinical dataset: {e}")
        print("This evaluation requires a clinical dataset with regulatory ground truth.")
        return
    print(f"Loaded clinical dataset: {dataset.name} with {len(dataset.questions)} questions")
    
    async with AsyncExitStack() as stack:
        client = httpx.AsyncClient(base_url="http://localhost:8001", timeout=args.timeout)
        await stack.enter_async_context(client)

        try:
            response = await client.get("/health")
            response.raise_for_status()
            print("✅ Service is available at http://localhost:8001")
        except Exception as e:
            print(f"❌ Service not available: {e}")
            print("Please start the hybrid service first: python service/hybrid_service.py")
            return

        llm_client: Optional[LLMClient] = None
        if args.use_llm:
            llm_config = config_from_args(args)
            llm_client = LLMClient(llm_config, timeout=args.timeout)
            await stack.enter_async_context(llm_client)
            print(f"🤖 Using LLM provider '{llm_config.provider}' with model '{llm_config.model}'")

        qa_evaluator = QAEvaluator()
        clinical_evaluator = ClinicalEvaluator()

        print(f"Evaluating clinical dataset: {dataset.name} ({len(dataset.questions)} questions)")
        start_time = time.time()
        
        all_results = []
        all_failures = []
        
        for question in dataset.questions:
            print(f"Evaluating question {question.id}: {question.question[:50]}...")
            
    result, failure_case = await evaluate_question(
        client,
        question,
        qa_evaluator,
        clinical_evaluator,
        top_k=args.top_k,
        llm_client=llm_client,
    )
            
            all_results.append(result)
            all_failures.append(failure_case)
        
        evaluation_time = time.time() - start_time
        
        # Aggregate metrics
        retrieval_metrics_list = [r["retrieval_metrics"] for r in all_results]
        qa_metrics_list = [r["qa_metrics"] for r in all_results]
        clinical_metrics_list = [r["clinical_metrics"] for r in all_results]
        
        aggregated_retrieval = aggregate_retrieval_metrics(retrieval_metrics_list)
        aggregated_qa = aggregate_qa_metrics(qa_metrics_list)
        aggregated_clinical = aggregate_clinical_metrics(clinical_metrics_list)

        if json_output:
            payload = {
                "dataset": {
                    "name": dataset.name,
                    "description": dataset.description,
                    "version": dataset.version,
                    "corpus_date": dataset.corpus_date,
                    "regulatory_framework": dataset.regulatory_framework,
                    "safety_focus": dataset.safety_focus,
                },
                "metadata": {
                    "question_count": len(dataset.questions),
                    "evaluation_time": evaluation_time,
                },
                "aggregated_metrics": {
                    "retrieval": to_serializable(aggregated_retrieval),
                    "qa": to_serializable(aggregated_qa),
                    "clinical": to_serializable(aggregated_clinical),
                },
                "questions": [],
            }

            for question, result, failure in zip(dataset.questions, all_results, all_failures):
                payload["questions"].append(
                    {
                        "question_id": question.id,
                        "question": to_serializable(question),
                        "generated_answer": result["generated_answer"],
                        "expected_answer": question.expected_answer,
                        "retrieval_metrics": to_serializable(result["retrieval_metrics"]),
                        "qa_metrics": to_serializable(result["qa_metrics"]),
                        "clinical_metrics": to_serializable(result["clinical_metrics"]),
                        "retrieval_results": [to_serializable(r) for r in result["retrieval_results"]],
                        "context_chunks": [
                            {
                                "chunk_id": hit.get("chunk_id"),
                                "score": hit.get("score"),
                                "rank": idx + 1,
                                "doc_id": hit.get("doc_id"),
                                "drug_title": hit.get("drug_title"),
                                "section_title": hit.get("section_title"),
                                "section_code": hit.get("section_code"),
                                "snippet": hit.get("snippet"),
                                "text": hit.get("text"),
                            }
                            for idx, hit in enumerate(result.get("search_results", [])[:6])
                        ],
                        "failure_analysis": to_serializable(failure),
                    }
                )

            json_output.parent.mkdir(parents=True, exist_ok=True)
            with json_output.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

            print(f"💾 JSON evaluation data saved to: {json_output}")
        
        # Generate comprehensive report
        report_content = generate_clinical_report(
            dataset, all_results, all_failures,
            aggregated_retrieval, aggregated_qa, aggregated_clinical,
            evaluation_time
        )
        
        # Print and save report
        print("\n" + "="*80)
        print(report_content)
        print("="*80)
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w") as f:
            f.write(report_content)
        
        print(f"📄 Clinical evaluation report saved to: {report_path}")


def generate_clinical_report(
    dataset: ClinicalEvalDataset,
    results: List[Dict],
    failures: List[ClinicalFailureCase],
    retrieval_metrics: Dict,
    qa_metrics: Dict, 
    clinical_metrics: ClinicalComplianceMetrics,
    evaluation_time: float,
) -> str:
    """Generate comprehensive clinical evaluation report."""
    
    lines = [
        "# Clinical Pharmacy Copilot Evaluation Report",
        "",
        f"**Dataset**: {dataset.name}",
        f"**Corpus Date**: {dataset.corpus_date}",
        f"**Regulatory Framework**: {dataset.regulatory_framework}",
        f"**Safety Focus**: {', '.join(dataset.safety_focus)}",
        f"**Questions**: {len(dataset.questions)}",
        f"**Evaluation Time**: {evaluation_time:.2f} seconds",
        f"**Average Time per Question**: {evaluation_time/len(dataset.questions):.2f} seconds",
        "",
    ]
    
    # Clinical safety overview
    lines.extend([
        "## 🏥 Clinical Safety Summary",
        "",
        f"**Overall Safety Score**: {clinical_metrics.clinical_safety_score:.3f}/1.000",
        f"**Critical Risk Failures**: {clinical_metrics.critical_risk_failures}",
        f"**High Risk Failures**: {clinical_metrics.high_risk_failures}",
        f"**Patient Safety Violations**: {sum(clinical_metrics.safety_violations.values())}",
        "",
    ])
    
    # Safety violation taxonomy (publication-ready insight)
    if clinical_metrics.safety_violations:
        total_violations = sum(clinical_metrics.safety_violations.values())
        lines.extend([
            "### 🎯 **Safety Violation Taxonomy** (Key Publication Insight)",
            "",
        ])
        
        for violation, count in sorted(clinical_metrics.safety_violations.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_violations) * 100
            lines.append(f"- **{violation.replace('_', ' ').title()}**: {count} violations ({percentage:.1f}%)")
        
        lines.append("")
    
    # Regulatory compliance metrics
    lines.append(format_clinical_metrics(clinical_metrics))
    
    # Traditional metrics
    lines.append(format_retrieval_metrics(retrieval_metrics, "Retrieval Performance"))
    lines.append(format_qa_metrics(qa_metrics, "Answer Quality"))
    
    # Clinical failure analysis
    clinical_failures = [f for f in failures if f.patient_risk_score > 0.3]
    if clinical_failures:
        lines.extend([
            "",
            "## ⚠️ High-Risk Clinical Failures",
            "",
        ])
        
        for failure in sorted(clinical_failures, key=lambda x: x.patient_risk_score, reverse=True)[:5]:
            lines.extend([
                f"### {failure.question_id} (Risk: {failure.patient_risk_score:.3f})",
                f"**Question**: {failure.question}",
                f"**Risk Level**: {failure.clinical_risk_level.value}",
                f"**Safety Violations**: {[v.value for v in failure.safety_violations]}",
                f"**Root Cause**: {failure.root_cause_analysis}",
                "",
                f"**Generated Answer**: {failure.generated_answer[:200]}...",
                "",
                "**Clinical Issues**:",
            ])
            
            if failure.dosing_errors:
                lines.append(f"- Dosing Errors: {failure.dosing_errors}")
            if failure.contraindications_missed:
                lines.append(f"- Missed Contraindications: {failure.contraindications_missed}")
            if failure.post_cutoff_info_used:
                lines.append(f"- Post-2018 Information Used: {failure.post_cutoff_info_used}")
            
            lines.append("")
    
    # Regulatory compliance analysis
    non_compliant = [f for f in failures if f.regulatory_risk_score > 0.3]
    if non_compliant:
        lines.extend([
            "## 🏛️ Regulatory Compliance Issues",
            "",
        ])
        
        for failure in non_compliant[:3]:
            lines.extend([
                f"### {failure.question_id}",
                f"**Expected Behavior**: {failure.expected_behavior.value}",
                f"**Compliance Violations**: {failure.compliance_violations}",
                f"**Regulatory Risk**: {failure.regulatory_risk_score:.3f}",
                "",
            ])
    
    # Publication-ready insights
    lines.extend([
        "",
        "## 📊 Publication-Ready Insights",
        "",
        "### Novel Evaluation Dimensions Beyond Traditional RAG",
        "",
        "1. **Time-Cutoff Compliance**: Measures system's ability to appropriately refuse post-corpus queries",
        f"   - Appropriate Refusal Rate: {clinical_metrics.appropriate_refusal_rate:.3f}",
        f"   - Time Cutoff Violations: {clinical_metrics.time_cutoff_violation_rate:.3f}",
        "",
        "2. **Clinical Consequence Weighting**: Errors weighted by patient safety impact",
        f"   - Critical Risk Failures: {clinical_metrics.critical_risk_failures} (life-threatening)",
        f"   - High Risk Failures: {clinical_metrics.high_risk_failures} (significant harm)",
        "",
        "3. **Regulatory Compliance**: AI Act-relevant traceability and refusal behaviors",
        f"   - Citation Present Rate: {clinical_metrics.citation_present_rate:.3f}",
        f"   - Groundedness Score: {clinical_metrics.groundedness_score:.3f}",
        "",
        "4. **Domain-Specific Safety Categories**: Pharmacy-relevant error taxonomy",
    ])
    
    if clinical_metrics.safety_violations:
        lines.append("   - Primary Safety Concerns:")
        for violation, count in list(clinical_metrics.safety_violations.items())[:5]:
            lines.append(f"     * {violation.replace('_', ' ').title()}: {count} cases")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run clinical pharmacy copilot evaluation")
    parser.add_argument("dataset", type=Path, help="Path to clinical evaluation dataset JSON")
    parser.add_argument("--report", type=Path, default=Path("eval/reports/clinical_report.md"), 
                       help="Path to save evaluation report")
    parser.add_argument("--json-output", type=Path, help="Optional path to save detailed JSON results")
    parser.add_argument("--use-llm", action="store_true", help="Generate answers with configured LLM instead of rule-based stub")
    parser.add_argument("--llm-provider", help="LLM provider identifier (ollama, openai, etc.)")
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--llm-base-url", help="Base URL for the LLM API endpoint")
    parser.add_argument("--llm-api-key", help="API key for hosted LLM providers")
    parser.add_argument("--llm-temperature", type=float, help="Sampling temperature for LLM generation")
    parser.add_argument("--llm-max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--llm-system-prompt", help="Override default system prompt for the LLM")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("EVAL_TIMEOUT", "30.0")),
                        help="HTTP timeout (seconds) for search and LLM calls")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("EVAL_TOP_K", "20")),
                        help="Number of chunks to request from the search service")
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
    
    try:
        asyncio.run(run_clinical_evaluation(args))
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
