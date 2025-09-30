"""Retrieval evaluation metrics for pharmacy copilot."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from dataset_schema import EvalQuestion, GroundTruthChunk


@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    chunk_id: str
    score: float
    rank: int  # 1-indexed


@dataclass 
class RetrievalMetrics:
    """Computed retrieval evaluation metrics."""
    # Core metrics
    recall_at_k: Dict[int, float]  # {k: recall@k}
    precision_at_k: Dict[int, float]  # {k: precision@k}
    
    # Ranking metrics
    mean_reciprocal_rank: float
    average_precision: float
    ndcg_at_k: Dict[int, float]  # {k: NDCG@k}
    
    # Additional insights
    essential_recall_at_k: Dict[int, float]  # Recall for essential chunks only
    first_relevant_rank: float  # Average rank of first relevant result
    
    # Raw counts for debugging
    total_relevant: int
    total_essential: int


def compute_recall_at_k(retrieved: List[RetrievalResult], relevant: Set[str], k: int) -> float:
    """Compute recall@k: fraction of relevant items retrieved in top-k."""
    if not relevant:
        return 0.0
    
    retrieved_top_k = {r.chunk_id for r in retrieved[:k]}
    return len(retrieved_top_k & relevant) / len(relevant)


def compute_precision_at_k(retrieved: List[RetrievalResult], relevant: Set[str], k: int) -> float:
    """Compute precision@k: fraction of retrieved items that are relevant."""
    if k == 0:
        return 0.0
    
    retrieved_top_k = [r.chunk_id for r in retrieved[:k]]
    relevant_retrieved = sum(1 for chunk_id in retrieved_top_k if chunk_id in relevant)
    return relevant_retrieved / min(k, len(retrieved_top_k))


def compute_reciprocal_rank(retrieved: List[RetrievalResult], relevant: Set[str]) -> float:
    """Compute reciprocal rank: 1/rank of first relevant item."""
    for result in retrieved:
        if result.chunk_id in relevant:
            return 1.0 / result.rank
    return 0.0


def compute_average_precision(retrieved: List[RetrievalResult], relevant: Set[str]) -> float:
    """Compute average precision: average of precision@k for each relevant item retrieved."""
    if not relevant:
        return 0.0
    
    relevant_retrieved = 0
    precision_sum = 0.0
    
    for i, result in enumerate(retrieved):
        if result.chunk_id in relevant:
            relevant_retrieved += 1
            precision_at_i = relevant_retrieved / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant) if relevant else 0.0


def compute_dcg(retrieved: List[RetrievalResult], relevance_scores: Dict[str, float], k: int) -> float:
    """Compute Discounted Cumulative Gain."""
    dcg = 0.0
    for i, result in enumerate(retrieved[:k]):
        if result.chunk_id in relevance_scores:
            relevance = relevance_scores[result.chunk_id]
            # DCG formula: sum(rel_i / log2(i + 2)) for i in 0..k-1
            dcg += relevance / math.log2(i + 2)
    return dcg


def compute_ndcg_at_k(retrieved: List[RetrievalResult], relevance_scores: Dict[str, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    if not relevance_scores:
        return 0.0
    
    # Compute DCG@k
    dcg = compute_dcg(retrieved, relevance_scores, k)
    
    # Compute IDCG@k (ideal DCG)
    ideal_ranking = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ideal_results = [RetrievalResult(chunk_id, score, i+1) for i, (chunk_id, score) in enumerate(ideal_ranking)]
    idcg = compute_dcg(ideal_results, relevance_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(question: EvalQuestion, retrieved: List[RetrievalResult]) -> RetrievalMetrics:
    """Evaluate retrieval results for a single question."""
    
    # Extract ground truth sets
    all_relevant = {chunk.chunk_id for chunk in question.relevant_chunks}
    essential_relevant = {chunk.chunk_id for chunk in question.relevant_chunks if chunk.is_essential}
    relevance_scores = {chunk.chunk_id: chunk.relevance_score for chunk in question.relevant_chunks}
    
    # Define k values to evaluate
    k_values = [1, 3, 5, 10, 20]
    
    # Compute recall@k and precision@k
    recall_at_k = {}
    precision_at_k = {}
    essential_recall_at_k = {}
    ndcg_at_k = {}
    
    for k in k_values:
        recall_at_k[k] = compute_recall_at_k(retrieved, all_relevant, k)
        precision_at_k[k] = compute_precision_at_k(retrieved, all_relevant, k)
        essential_recall_at_k[k] = compute_recall_at_k(retrieved, essential_relevant, k)
        ndcg_at_k[k] = compute_ndcg_at_k(retrieved, relevance_scores, k)
    
    # Compute ranking metrics
    mrr = compute_reciprocal_rank(retrieved, all_relevant)
    avg_precision = compute_average_precision(retrieved, all_relevant)
    
    # Find first relevant rank
    first_relevant_rank = float('inf')
    for result in retrieved:
        if result.chunk_id in all_relevant:
            first_relevant_rank = result.rank
            break
    
    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mean_reciprocal_rank=mrr,
        average_precision=avg_precision,
        ndcg_at_k=ndcg_at_k,
        essential_recall_at_k=essential_recall_at_k,
        first_relevant_rank=first_relevant_rank if first_relevant_rank != float('inf') else 0.0,
        total_relevant=len(all_relevant),
        total_essential=len(essential_relevant),
    )


def aggregate_retrieval_metrics(metrics_list: List[RetrievalMetrics]) -> RetrievalMetrics:
    """Aggregate retrieval metrics across multiple questions."""
    if not metrics_list:
        return RetrievalMetrics(
            recall_at_k={}, precision_at_k={}, mean_reciprocal_rank=0.0,
            average_precision=0.0, ndcg_at_k={}, essential_recall_at_k={},
            first_relevant_rank=0.0, total_relevant=0, total_essential=0
        )
    
    n = len(metrics_list)
    
    # Get all k values
    k_values = set()
    for metrics in metrics_list:
        k_values.update(metrics.recall_at_k.keys())
    k_values = sorted(k_values)
    
    # Aggregate metrics
    recall_at_k = {k: sum(m.recall_at_k.get(k, 0.0) for m in metrics_list) / n for k in k_values}
    precision_at_k = {k: sum(m.precision_at_k.get(k, 0.0) for m in metrics_list) / n for k in k_values}
    essential_recall_at_k = {k: sum(m.essential_recall_at_k.get(k, 0.0) for m in metrics_list) / n for k in k_values}
    ndcg_at_k = {k: sum(m.ndcg_at_k.get(k, 0.0) for m in metrics_list) / n for k in k_values}
    
    mrr = sum(m.mean_reciprocal_rank for m in metrics_list) / n
    avg_precision = sum(m.average_precision for m in metrics_list) / n
    first_relevant_rank = sum(m.first_relevant_rank for m in metrics_list) / n
    
    total_relevant = sum(m.total_relevant for m in metrics_list)
    total_essential = sum(m.total_essential for m in metrics_list)
    
    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mean_reciprocal_rank=mrr,
        average_precision=avg_precision,
        ndcg_at_k=ndcg_at_k,
        essential_recall_at_k=essential_recall_at_k,
        first_relevant_rank=first_relevant_rank,
        total_relevant=total_relevant,
        total_essential=total_essential,
    )


def format_retrieval_metrics(metrics: RetrievalMetrics, title: str = "Retrieval Metrics") -> str:
    """Format retrieval metrics for display."""
    lines = [f"\n=== {title} ==="]
    
    # Recall and Precision
    lines.append("\n📊 Recall@k:")
    for k in sorted(metrics.recall_at_k.keys()):
        lines.append(f"  R@{k}: {metrics.recall_at_k[k]:.3f}")
    
    lines.append("\n📊 Precision@k:")
    for k in sorted(metrics.precision_at_k.keys()):
        lines.append(f"  P@{k}: {metrics.precision_at_k[k]:.3f}")
    
    lines.append("\n📊 Essential Recall@k:")
    for k in sorted(metrics.essential_recall_at_k.keys()):
        lines.append(f"  ER@{k}: {metrics.essential_recall_at_k[k]:.3f}")
    
    # Ranking metrics
    lines.append("\n🎯 Ranking Metrics:")
    lines.append(f"  MRR: {metrics.mean_reciprocal_rank:.3f}")
    lines.append(f"  MAP: {metrics.average_precision:.3f}")
    lines.append(f"  First Relevant Rank: {metrics.first_relevant_rank:.1f}")
    
    lines.append("\n📈 NDCG@k:")
    for k in sorted(metrics.ndcg_at_k.keys()):
        lines.append(f"  NDCG@{k}: {metrics.ndcg_at_k[k]:.3f}")
    
    # Summary stats
    lines.append("\n📋 Summary:")
    lines.append(f"  Total Relevant: {metrics.total_relevant}")
    lines.append(f"  Total Essential: {metrics.total_essential}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test with dummy data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.dataset_schema import EvalQuestion, GroundTruthChunk, QuestionType, DifficultyLevel
    
    # Create test question
    question = EvalQuestion(
        id="test",
        question="Test question",
        question_type=QuestionType.DOSAGE,
        difficulty=DifficultyLevel.EASY,
        expected_answer="Test answer",
        answer_key_points=["test"],
        relevant_chunks=[
            GroundTruthChunk("chunk1", 1.0, True),
            GroundTruthChunk("chunk2", 0.8, False),
            GroundTruthChunk("chunk3", 0.6, False),
        ],
        drug_names=["test"],
        tags=["test"],
    )
    
    # Create test retrieval results
    retrieved = [
        RetrievalResult("chunk1", 0.95, 1),  # Relevant, rank 1
        RetrievalResult("chunk4", 0.90, 2),  # Not relevant, rank 2  
        RetrievalResult("chunk2", 0.85, 3),  # Relevant, rank 3
        RetrievalResult("chunk5", 0.80, 4),  # Not relevant, rank 4
        RetrievalResult("chunk3", 0.75, 5),  # Relevant, rank 5
    ]
    
    # Evaluate
    metrics = evaluate_retrieval(question, retrieved)
    print(format_retrieval_metrics(metrics, "Test Retrieval Metrics"))