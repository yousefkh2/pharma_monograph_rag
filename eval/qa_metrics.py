"""End-to-end QA evaluation metrics for pharmacy copilot."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from dataset_schema import EvalQuestion


@dataclass
class QAMetrics:
    """End-to-end QA evaluation metrics."""
    # Lexical overlap metrics
    bleu_score: float
    rouge_1_f1: float
    rouge_2_f1: float  
    rouge_l_f1: float
    
    # Semantic similarity
    semantic_similarity: float  # Cosine similarity of embeddings
    
    # Content coverage
    key_points_covered: float  # Fraction of key points found in answer
    key_points_found: List[str]  # Which key points were found
    key_points_missing: List[str]  # Which key points were missing
    
    # Answer characteristics
    answer_length: int  # Number of words in generated answer
    has_answer: bool    # Whether system provided an answer vs "I don't know"
    
    # LLM-as-a-judge (placeholder for now)
    factual_accuracy: Optional[float] = None  # 0-1 score from LLM judge
    helpfulness: Optional[float] = None       # 0-1 score from LLM judge


class QAEvaluator:
    """Evaluates generated answers against ground truth."""
    
    def __init__(self):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing dependencies. Install with: pip install rouge-score nltk sentence-transformers")
        
        # Initialize NLTK data
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
            except:
                pass
            nltk.download('punkt_tab', quiet=True)
            
        # Initialize models
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        self.smoothing = SmoothingFunction().method1
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for comparison
        return text.lower()
    
    def _compute_bleu(self, reference: str, candidate: str) -> float:
        """Compute BLEU score between reference and candidate."""
        ref_tokens = nltk.word_tokenize(self._preprocess_text(reference))
        cand_tokens = nltk.word_tokenize(self._preprocess_text(candidate))
        
        if not cand_tokens:
            return 0.0
            
        # Use BLEU-4 with smoothing
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
    
    def _compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute ROUGE scores."""
        if not candidate.strip():
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
            
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1_f1': scores['rouge1'].fmeasure,
            'rouge2_f1': scores['rouge2'].fmeasure,
            'rougeL_f1': scores['rougeL'].fmeasure,
        }
    
    def _compute_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if not candidate.strip():
            return 0.0
            
        embeddings = self.similarity_model.encode([reference, candidate])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def _check_key_points_coverage(self, answer: str, key_points: List[str]) -> Tuple[float, List[str], List[str]]:
        """Check which key points are covered in the answer."""
        if not key_points:
            return 1.0, [], []
            
        answer_lower = self._preprocess_text(answer)
        found = []
        missing = []
        
        for key_point in key_points:
            key_point_lower = self._preprocess_text(key_point)
            
            # Simple substring matching (could be improved with fuzzy matching)
            if key_point_lower in answer_lower:
                found.append(key_point)
            else:
                # Try word-by-word matching for more flexibility
                key_words = set(nltk.word_tokenize(key_point_lower))
                answer_words = set(nltk.word_tokenize(answer_lower))
                
                # Consider found if most key words are present
                overlap = len(key_words & answer_words)
                if overlap >= len(key_words) * 0.7:  # 70% of words must match
                    found.append(key_point)
                else:
                    missing.append(key_point)
        
        coverage = len(found) / len(key_points) if key_points else 1.0
        return coverage, found, missing
    
    def _detect_no_answer(self, answer: str) -> bool:
        """Detect if the system said it doesn't know the answer."""
        answer_lower = answer.lower()
        no_answer_phrases = [
            "i don't know", "i do not know", "i'm not sure", "i am not sure",
            "cannot determine", "can't determine", "no information", "insufficient information",
            "unable to answer", "don't have enough information"
        ]
        return any(phrase in answer_lower for phrase in no_answer_phrases)
    
    def evaluate_answer(self, question: EvalQuestion, generated_answer: str) -> QAMetrics:
        """Evaluate a generated answer against the ground truth."""
        
        # Preprocess inputs
        expected = question.expected_answer
        generated = generated_answer.strip()
        
        # Check if answer was provided
        has_answer = bool(generated) and not self._detect_no_answer(generated)
        
        if not has_answer:
            # Return zero metrics for non-answers
            return QAMetrics(
                bleu_score=0.0,
                rouge_1_f1=0.0,
                rouge_2_f1=0.0,
                rouge_l_f1=0.0,
                semantic_similarity=0.0,
                key_points_covered=0.0,
                key_points_found=[],
                key_points_missing=question.answer_key_points,
                answer_length=len(generated.split()) if generated else 0,
                has_answer=False,
            )
        
        # Compute lexical metrics
        bleu = self._compute_bleu(expected, generated)
        rouge_scores = self._compute_rouge(expected, generated)
        
        # Compute semantic similarity
        semantic_sim = self._compute_semantic_similarity(expected, generated) 
        
        # Check key points coverage
        coverage, found, missing = self._check_key_points_coverage(generated, question.answer_key_points)
        
        return QAMetrics(
            bleu_score=bleu,
            rouge_1_f1=rouge_scores['rouge1_f1'],
            rouge_2_f1=rouge_scores['rouge2_f1'],
            rouge_l_f1=rouge_scores['rougeL_f1'],
            semantic_similarity=semantic_sim,
            key_points_covered=coverage,
            key_points_found=found,
            key_points_missing=missing,
            answer_length=len(generated.split()),
            has_answer=True,
        )


def aggregate_qa_metrics(metrics_list: List[QAMetrics]) -> QAMetrics:
    """Aggregate QA metrics across multiple questions."""
    if not metrics_list:
        return QAMetrics(
            bleu_score=0.0, rouge_1_f1=0.0, rouge_2_f1=0.0, rouge_l_f1=0.0,
            semantic_similarity=0.0, key_points_covered=0.0,
            key_points_found=[], key_points_missing=[], answer_length=0, has_answer=False
        )
    
    n = len(metrics_list)
    
    # Average numerical metrics
    avg_bleu = sum(m.bleu_score for m in metrics_list) / n
    avg_rouge_1 = sum(m.rouge_1_f1 for m in metrics_list) / n
    avg_rouge_2 = sum(m.rouge_2_f1 for m in metrics_list) / n
    avg_rouge_l = sum(m.rouge_l_f1 for m in metrics_list) / n
    avg_semantic_sim = sum(m.semantic_similarity for m in metrics_list) / n
    avg_key_points = sum(m.key_points_covered for m in metrics_list) / n
    avg_answer_length = sum(m.answer_length for m in metrics_list) / n
    
    # Count-based metrics
    answer_rate = sum(1 for m in metrics_list if m.has_answer) / n
    
    return QAMetrics(
        bleu_score=avg_bleu,
        rouge_1_f1=avg_rouge_1,
        rouge_2_f1=avg_rouge_2,
        rouge_l_f1=avg_rouge_l,
        semantic_similarity=avg_semantic_sim,
        key_points_covered=avg_key_points,
        key_points_found=[],  # Individual lists don't aggregate meaningfully
        key_points_missing=[],
        answer_length=int(avg_answer_length),
        has_answer=answer_rate > 0.5,  # Majority
    )


def format_qa_metrics(metrics: QAMetrics, title: str = "QA Metrics") -> str:
    """Format QA metrics for display."""
    lines = [f"\n=== {title} ==="]
    
    # Answer characteristics
    lines.append(f"\n📝 Answer Quality:")
    lines.append(f"  Has Answer: {metrics.has_answer}")
    lines.append(f"  Answer Length: {metrics.answer_length} words")
    
    # Lexical overlap
    lines.append(f"\n📊 Lexical Overlap:")
    lines.append(f"  BLEU: {metrics.bleu_score:.3f}")
    lines.append(f"  ROUGE-1 F1: {metrics.rouge_1_f1:.3f}")
    lines.append(f"  ROUGE-2 F1: {metrics.rouge_2_f1:.3f}")
    lines.append(f"  ROUGE-L F1: {metrics.rouge_l_f1:.3f}")
    
    # Semantic similarity
    lines.append(f"\n🧠 Semantic Similarity: {metrics.semantic_similarity:.3f}")
    
    # Key points coverage
    lines.append(f"\n🎯 Key Points Coverage: {metrics.key_points_covered:.3f}")
    if metrics.key_points_found:
        lines.append(f"  Found: {metrics.key_points_found}")
    if metrics.key_points_missing:
        lines.append(f"  Missing: {metrics.key_points_missing}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the QA evaluator
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval.dataset_schema import EvalQuestion, QuestionType, DifficultyLevel
    
    # Create test question
    question = EvalQuestion(
        id="test",
        question="What is the maximum daily dose of acetaminophen?",
        question_type=QuestionType.DOSAGE,
        difficulty=DifficultyLevel.EASY,
        expected_answer="The maximum daily dose of acetaminophen for adults is 4000 mg per day.",
        answer_key_points=["4000 mg", "maximum daily dose", "adults"],
        relevant_chunks=[],
        drug_names=["acetaminophen"],
        tags=["dosage"],
    )
    
    # Test answers
    test_answers = [
        "The maximum daily dose of acetaminophen for adults is 4000 mg per day to avoid liver toxicity.",
        "Adults should not exceed 4 grams of acetaminophen daily.",
        "I don't know the maximum dose.",
        "Acetaminophen is a pain reliever but I'm not sure about dosing.",
    ]
    
    evaluator = QAEvaluator()
    
    for i, answer in enumerate(test_answers, 1):
        print(f"\n--- Test Answer {i} ---")
        print(f"Answer: {answer}")
        metrics = evaluator.evaluate_answer(question, answer)
        print(format_qa_metrics(metrics, f"Metrics for Answer {i}"))