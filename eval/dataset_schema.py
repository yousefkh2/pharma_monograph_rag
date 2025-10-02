"""Evaluation dataset schema and types for pharmacy copilot."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path
import json


class QuestionType(str, Enum):
    """Categories of pharmacy questions for targeted evaluation."""
    DOSAGE = "dosage"                    # "What's the standard dose of acetaminophen?"
    CONTRAINDICATIONS = "contraindications"  # "When should I avoid ibuprofen?"
    INTERACTIONS = "interactions"        # "Does warfarin interact with aspirin?"
    INTERACTION = "interaction"          # "Does warfarin interact with aspirin?" (singular)
    SIDE_EFFECTS = "side_effects"       # "What are common side effects of metformin?"
    MECHANISM = "mechanism"             # "How does lisinopril work?"
    ADMINISTRATION = "administration"    # "How should I take this medication?"
    MONITORING = "monitoring"           # "What labs should be monitored on statins?"
    PREGNANCY = "pregnancy"             # "Is this safe during pregnancy?"
    GENERAL_INFO = "general_info"       # "What is acetaminophen used for?"
    GUIDELINE = "guideline"             # "What are the guidelines for this medication?"


class DifficultyLevel(str, Enum):
    """Difficulty levels for questions."""
    EASY = "easy"        # Single fact, common drug
    MEDIUM = "medium"    # Multiple facts, less common drug, or requires inference
    HARD = "hard"        # Complex interactions, rare scenarios, multi-step reasoning


@dataclass
class GroundTruthChunk:
    """Reference to a chunk that should be retrieved for a question."""
    chunk_id: str
    relevance_score: float  # 0.0-1.0, how relevant is this chunk
    is_essential: bool      # Must be retrieved for correct answer
    note: Optional[str] = None  # Why this chunk is relevant


@dataclass
class EvalQuestion:
    """A single evaluation question with ground truth."""
    id: str
    question: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    
    # Expected answer
    expected_answer: str
    answer_key_points: List[str]  # Key facts that must be in the answer
    
    # Retrieval ground truth
    relevant_chunks: List[GroundTruthChunk]

    # Metadata
    drug_names: List[str]  # Primary drugs mentioned
    tags: List[str]        # Additional categorization
    source_note: Optional[str] = None  # Where this question came from

    # Reasoning/explanation
    reasoning: Optional[str] = None  # Why this is the correct answer
    gold: Optional[Dict[str, object]] = None  # Rich evaluable structure for LLM judge
    oracle_context: Optional[List[Dict[str, object]]] = None
    judge_metadata: Optional[Dict[str, object]] = None


@dataclass
class EvalDataset:
    """Collection of evaluation questions."""
    name: str
    description: str
    version: str
    questions: List[EvalQuestion]
    
    def save(self, path: Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "question_type": q.question_type.value,
                    "difficulty": q.difficulty.value,
                    "expected_answer": q.expected_answer,
                    "answer_key_points": q.answer_key_points,
                    "relevant_chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "relevance_score": chunk.relevance_score,
                            "is_essential": chunk.is_essential,
                            "note": chunk.note,
                        }
                        for chunk in q.relevant_chunks
                    ],
                "drug_names": q.drug_names,
                "tags": q.tags,
                "source_note": q.source_note,
                "reasoning": q.reasoning,
                "gold": q.gold,
                "oracle_context": q.oracle_context,
                "judge_metadata": q.judge_metadata,
            }
            for q in self.questions
        ]
    }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset from JSON file."""
        with path.open("r") as f:
            data = json.load(f)
        
        questions = []
        for q_data in data["questions"]:
            relevant_chunks = [
                GroundTruthChunk(
                    chunk_id=chunk["chunk_id"],
                    relevance_score=chunk["relevance_score"],
                    is_essential=chunk["is_essential"],
                    note=chunk.get("note"),
                )
                for chunk in q_data["relevant_chunks"]
            ]
            
            question = EvalQuestion(
                id=q_data["id"],
                question=q_data["question"],
                question_type=QuestionType(q_data["question_type"]),
                difficulty=DifficultyLevel(q_data["difficulty"]),
                expected_answer=q_data["expected_answer"],
                answer_key_points=q_data["answer_key_points"],
                relevant_chunks=relevant_chunks,
                drug_names=q_data["drug_names"],
                tags=q_data["tags"],
                source_note=q_data.get("source_note"),
                reasoning=q_data.get("reasoning"),
                gold=q_data.get("gold"),
                oracle_context=q_data.get("oracle_context"),
                judge_metadata=q_data.get("judge_metadata"),
            )
            questions.append(question)
        
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            questions=questions,
        )


def create_sample_questions() -> List[EvalQuestion]:
    """Create a small set of sample evaluation questions."""
    
    questions = [
        EvalQuestion(
            id="q001",
            question="What is the maximum daily dose of acetaminophen for adults?",
            question_type=QuestionType.DOSAGE,
            difficulty=DifficultyLevel.EASY,
            expected_answer="The maximum daily dose of acetaminophen for adults is 4000 mg (4 grams) per day.",
            answer_key_points=[
                "4000 mg maximum daily dose",
                "4 grams maximum daily dose", 
                "For adults"
            ],
            relevant_chunks=[
                # These would be real chunk IDs from your corpus
                GroundTruthChunk("chunk_acetaminophen_dosage_001", 1.0, True, "Primary dosage information"),
                GroundTruthChunk("chunk_acetaminophen_safety_001", 0.8, False, "Safety warnings about max dose"),
            ],
            drug_names=["acetaminophen", "paracetamol"],
            tags=["common", "otc", "dosage"],
            source_note="Standard pharmacy reference"
        ),
        
        EvalQuestion(
            id="q002", 
            question="What are the contraindications for ibuprofen?",
            question_type=QuestionType.CONTRAINDICATIONS,
            difficulty=DifficultyLevel.MEDIUM,
            expected_answer="Ibuprofen is contraindicated in patients with active GI bleeding, severe heart failure, severe kidney disease, and in the third trimester of pregnancy.",
            answer_key_points=[
                "Active GI bleeding",
                "Severe heart failure", 
                "Severe kidney disease",
                "Third trimester pregnancy"
            ],
            relevant_chunks=[
                GroundTruthChunk("chunk_ibuprofen_contraind_001", 1.0, True, "Main contraindications"),
                GroundTruthChunk("chunk_ibuprofen_pregnancy_001", 0.9, True, "Pregnancy warnings"),
                GroundTruthChunk("chunk_nsaid_warnings_001", 0.7, False, "General NSAID warnings"),
            ],
            drug_names=["ibuprofen"],
            tags=["nsaid", "contraindications", "safety"],
            source_note="FDA prescribing information"
        ),
        
        EvalQuestion(
            id="q003",
            question="How does warfarin interact with aspirin?",
            question_type=QuestionType.INTERACTIONS,
            difficulty=DifficultyLevel.HARD,
            expected_answer="Warfarin and aspirin both increase bleeding risk. When used together, they significantly increase the risk of serious bleeding complications and require careful monitoring of INR levels.",
            answer_key_points=[
                "Both increase bleeding risk",
                "Significant bleeding risk when combined", 
                "Requires INR monitoring",
                "Serious bleeding complications"
            ],
            relevant_chunks=[
                GroundTruthChunk("chunk_warfarin_interactions_001", 1.0, True, "Warfarin drug interactions"),
                GroundTruthChunk("chunk_aspirin_bleeding_001", 0.9, True, "Aspirin bleeding risks"),
                GroundTruthChunk("chunk_anticoag_monitoring_001", 0.8, False, "Anticoagulation monitoring"),
            ],
            drug_names=["warfarin", "aspirin"],
            tags=["interactions", "bleeding", "monitoring", "anticoagulation"],
            source_note="Clinical pharmacology reference"
        ),
    ]
    
    return questions


if __name__ == "__main__":
    # Create sample dataset
    sample_questions = create_sample_questions()
    dataset = EvalDataset(
        name="pharmacy_copilot_eval_v1",
        description="Initial evaluation dataset for pharmacy copilot system",
        version="1.0.0",
        questions=sample_questions
    )
    
    # Save to file
    output_path = Path("eval/datasets/sample_eval_v1.json")
    dataset.save(output_path)
    print(f"Saved sample evaluation dataset with {len(sample_questions)} questions to {output_path}")
    
    # Test loading
    loaded = EvalDataset.load(output_path)
    print(f"Successfully loaded dataset: {loaded.name} with {len(loaded.questions)} questions")
