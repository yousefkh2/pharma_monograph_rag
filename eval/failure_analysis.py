"""Failure taxonomy and analysis for pharmacy copilot evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from dataset_schema import EvalQuestion
from retrieval_metrics import RetrievalResult, RetrievalMetrics
from qa_metrics import QAMetrics


class FailureType(str, Enum):
    """Categories of system failures."""
    # Retrieval failures
    RETRIEVAL_MISS = "retrieval_miss"           # Failed to retrieve relevant chunks
    RETRIEVAL_IRRELEVANT = "retrieval_irrelevant"  # Retrieved irrelevant chunks
    RETRIEVAL_PARTIAL = "retrieval_partial"     # Retrieved some but not all essential chunks
    
    # QA failures  
    QA_NO_ANSWER = "qa_no_answer"              # System said "I don't know"
    QA_INCORRECT = "qa_incorrect"              # Wrong answer despite good retrieval
    QA_INCOMPLETE = "qa_incomplete"            # Missing key information
    QA_HALLUCINATION = "qa_hallucination"      # Made up information not in retrieved chunks
    
    # End-to-end failures
    E2E_LOW_QUALITY = "e2e_low_quality"        # Low overall quality scores
    E2E_FACTUAL_ERROR = "e2e_factual_error"   # Factually incorrect information


@dataclass
class FailureCase:
    """A specific failure instance."""
    question_id: str
    question: str
    failure_types: List[FailureType]
    
    # Retrieval context
    retrieved_chunks: List[RetrievalResult]
    relevant_chunks_found: Set[str]
    essential_chunks_missed: Set[str]
    
    # QA context
    generated_answer: str
    expected_answer: str
    key_points_missing: List[str]
    
    # Metrics
    retrieval_metrics: RetrievalMetrics
    qa_metrics: QAMetrics
    
    # Analysis
    root_cause: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    notes: Optional[str] = None


class FailureAnalyzer:
    """Analyzes evaluation results to identify and categorize failures."""
    
    def __init__(
        self,
        recall_threshold: float = 0.8,
        precision_threshold: float = 0.6,
        semantic_similarity_threshold: float = 0.7,
        key_points_threshold: float = 0.8,
    ):
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.key_points_threshold = key_points_threshold
    
    def analyze_failure(
        self,
        question: EvalQuestion,
        retrieved: List[RetrievalResult],
        generated_answer: str,
        retrieval_metrics: RetrievalMetrics,
        qa_metrics: QAMetrics,
    ) -> FailureCase:
        """Analyze a single question's results and identify failure modes."""
        
        failure_types = []
        
        # Identify retrieval failures
        retrieval_failures = self._analyze_retrieval_failures(question, retrieved, retrieval_metrics)
        failure_types.extend(retrieval_failures)
        
        # Identify QA failures
        qa_failures = self._analyze_qa_failures(question, generated_answer, qa_metrics)
        failure_types.extend(qa_failures)
        
        # Identify end-to-end failures
        e2e_failures = self._analyze_e2e_failures(retrieval_metrics, qa_metrics)
        failure_types.extend(e2e_failures)
        
        # Extract contextual information
        retrieved_chunk_ids = {r.chunk_id for r in retrieved}
        relevant_chunk_ids = {c.chunk_id for c in question.relevant_chunks}
        essential_chunk_ids = {c.chunk_id for c in question.relevant_chunks if c.is_essential}
        
        relevant_found = retrieved_chunk_ids & relevant_chunk_ids
        essential_missed = essential_chunk_ids - retrieved_chunk_ids
        
        # Determine severity and root cause
        severity = self._determine_severity(failure_types, retrieval_metrics, qa_metrics)
        root_cause = self._determine_root_cause(failure_types, retrieval_metrics, qa_metrics)
        
        return FailureCase(
            question_id=question.id,
            question=question.question,
            failure_types=failure_types,
            retrieved_chunks=retrieved,
            relevant_chunks_found=relevant_found,
            essential_chunks_missed=essential_missed,
            generated_answer=generated_answer,
            expected_answer=question.expected_answer,
            key_points_missing=qa_metrics.key_points_missing,
            retrieval_metrics=retrieval_metrics,
            qa_metrics=qa_metrics,
            root_cause=root_cause,
            severity=severity,
        )
    
    def _analyze_retrieval_failures(
        self,
        question: EvalQuestion,
        retrieved: List[RetrievalResult],
        metrics: RetrievalMetrics,
    ) -> List[FailureType]:
        """Identify retrieval-specific failures."""
        failures = []
        
        # Check overall recall
        recall_5 = metrics.recall_at_k.get(5, 0.0)
        if recall_5 < self.recall_threshold:
            failures.append(FailureType.RETRIEVAL_MISS)
        
        # Check precision
        precision_5 = metrics.precision_at_k.get(5, 0.0)
        if precision_5 < self.precision_threshold:
            failures.append(FailureType.RETRIEVAL_IRRELEVANT)
        
        # Check essential chunks
        essential_recall_5 = metrics.essential_recall_at_k.get(5, 0.0)
        if essential_recall_5 < 1.0 and recall_5 > 0:
            failures.append(FailureType.RETRIEVAL_PARTIAL)
        
        return failures
    
    def _analyze_qa_failures(
        self,
        question: EvalQuestion,
        generated_answer: str,
        metrics: QAMetrics,
    ) -> List[FailureType]:
        """Identify QA-specific failures."""
        failures = []
        
        # Check if system provided an answer
        if not metrics.has_answer:
            failures.append(FailureType.QA_NO_ANSWER)
            return failures  # Other QA metrics don't apply if no answer
        
        # Check semantic similarity
        if metrics.semantic_similarity < self.semantic_similarity_threshold:
            failures.append(FailureType.QA_INCORRECT)
        
        # Check key points coverage
        if metrics.key_points_covered < self.key_points_threshold:
            failures.append(FailureType.QA_INCOMPLETE)
        
        # TODO: Add hallucination detection
        # This would require checking if the answer contains information
        # not present in the retrieved chunks
        
        return failures
    
    def _analyze_e2e_failures(
        self,
        retrieval_metrics: RetrievalMetrics,
        qa_metrics: QAMetrics,
    ) -> List[FailureType]:
        """Identify end-to-end failures."""
        failures = []
        
        # Overall quality check
        if (qa_metrics.semantic_similarity < 0.6 and 
            qa_metrics.key_points_covered < 0.5):
            failures.append(FailureType.E2E_LOW_QUALITY)
        
        # TODO: Add factual error detection using LLM-as-a-judge
        
        return failures
    
    def _determine_severity(
        self,
        failure_types: List[FailureType],
        retrieval_metrics: RetrievalMetrics,
        qa_metrics: QAMetrics,
    ) -> str:
        """Determine failure severity."""
        if not failure_types:
            return "none"
        
        # Critical failures
        if (FailureType.QA_NO_ANSWER in failure_types or
            FailureType.E2E_FACTUAL_ERROR in failure_types):
            return "critical"
        
        # High severity
        if (FailureType.RETRIEVAL_MISS in failure_types or
            FailureType.QA_INCORRECT in failure_types):
            return "high"
        
        # Medium severity  
        if (FailureType.RETRIEVAL_PARTIAL in failure_types or
            FailureType.QA_INCOMPLETE in failure_types):
            return "medium"
        
        # Low severity
        return "low"
    
    def _determine_root_cause(
        self,
        failure_types: List[FailureType],
        retrieval_metrics: RetrievalMetrics,
        qa_metrics: QAMetrics,
    ) -> Optional[str]:
        """Determine likely root cause of failures."""
        retrieval_failures = [f for f in failure_types if f.value.startswith('retrieval')]
        qa_failures = [f for f in failure_types if f.value.startswith('qa')]
        
        if retrieval_failures and not qa_failures:
            return "retrieval_system"
        elif qa_failures and not retrieval_failures:
            return "qa_system"  
        elif retrieval_failures and qa_failures:
            return "both_systems"
        elif FailureType.E2E_LOW_QUALITY in failure_types:
            return "integration_issue"
        else:
            return "unknown"


def generate_failure_report(failures: List[FailureCase]) -> str:
    """Generate a comprehensive failure analysis report."""
    if not failures:
        return "🎉 No failures detected!"
    
    lines = ["# Failure Analysis Report\n"]
    
    # Summary statistics
    total_failures = len(failures)
    severity_counts = {}
    failure_type_counts = {}
    root_cause_counts = {}
    
    for failure in failures:
        severity_counts[failure.severity] = severity_counts.get(failure.severity, 0) + 1
        root_cause_counts[failure.root_cause] = root_cause_counts.get(failure.root_cause, 0) + 1
        
        for failure_type in failure.failure_types:
            failure_type_counts[failure_type.value] = failure_type_counts.get(failure_type.value, 0) + 1
    
    lines.append(f"## Summary")
    lines.append(f"- **Total Failures**: {total_failures}")
    lines.append(f"- **By Severity**: {dict(sorted(severity_counts.items()))}")
    lines.append(f"- **By Root Cause**: {dict(sorted(root_cause_counts.items()))}")
    lines.append("")
    
    # Failure type breakdown
    lines.append("## Failure Types")
    for failure_type, count in sorted(failure_type_counts.items()):
        percentage = (count / total_failures) * 100
        lines.append(f"- **{failure_type}**: {count} ({percentage:.1f}%)")
    lines.append("")
    
    # Detailed cases (show worst cases)
    critical_failures = [f for f in failures if f.severity == "critical"]
    high_failures = [f for f in failures if f.severity == "high"]
    
    if critical_failures:
        lines.append("## Critical Failures")
        for failure in critical_failures[:5]:  # Show top 5
            lines.append(f"### {failure.question_id}")
            lines.append(f"**Question**: {failure.question}")
            lines.append(f"**Failure Types**: {[f.value for f in failure.failure_types]}")
            lines.append(f"**Root Cause**: {failure.root_cause}")
            lines.append(f"**Generated**: {failure.generated_answer[:100]}...")
            lines.append(f"**Expected**: {failure.expected_answer[:100]}...")
            lines.append("")
    
    if high_failures:
        lines.append("## High Severity Failures")
        for failure in high_failures[:3]:  # Show top 3
            lines.append(f"### {failure.question_id}")
            lines.append(f"**Question**: {failure.question}")
            lines.append(f"**Failure Types**: {[f.value for f in failure.failure_types]}")
            lines.append(f"**Root Cause**: {failure.root_cause}")
            lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test failure analysis
    print("Failure taxonomy framework implemented!")
    print("Available failure types:", [f.value for f in FailureType])