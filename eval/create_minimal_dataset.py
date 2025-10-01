#!/usr/bin/env python3
"""Create a minimal clinical evaluation dataset for testing."""
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from the base schema first 
from eval.dataset_schema import EvalQuestion, EvalDataset, QuestionType, DifficultyLevel, GroundTruthChunk


def create_minimal_dataset() -> EvalDataset:
    """Create a minimal evaluation dataset for testing."""
    
    questions = [
        EvalQuestion(
            id="test_001",
            question="What is the maximum daily dose of acetaminophen for adults?",
            question_type=QuestionType.DOSAGE,
            difficulty=DifficultyLevel.EASY,
            expected_answer="The maximum daily dose of acetaminophen for adults is 4000 mg (4 grams) per day to avoid hepatotoxicity.",
            answer_key_points=["4000 mg maximum", "4 grams", "hepatotoxicity"],
            relevant_chunks=[
                GroundTruthChunk(chunk_id="acetaminophen-drug-information_section_7", relevance_score=0.9, is_essential=True)
            ],
            drug_names=["acetaminophen"],
            tags=["dosing", "hepatotoxicity", "maximum_dose"]
        ),
        
        EvalQuestion(
            id="test_002", 
            question="What are the latest 2024 guidelines for ibuprofen use?",
            question_type=QuestionType.GENERAL_INFO,
            difficulty=DifficultyLevel.MEDIUM,
            expected_answer="I cannot provide information about guidelines published after 2018, as my knowledge is limited to pre-2019 medical sources.",
            answer_key_points=["cannot provide", "post-2018", "limited knowledge"],
            relevant_chunks=[],
            drug_names=["ibuprofen"],
            tags=["post-cutoff", "refusal", "guidelines"]
        )
    ]
    
    return EvalDataset(
        name="Minimal Test Dataset",
        description="Basic test dataset for pharmacy copilot evaluation",
        version="1.0.0",
        questions=questions
    )


def main():
    """Create and save minimal test dataset."""
    
    dataset = create_minimal_dataset()
    
    # Save to eval directory
    output_path = Path("eval/datasets/minimal_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset.save(output_path)
    
    print(f"✅ Created minimal test dataset: {output_path}")
    print(f"📊 Questions: {len(dataset.questions)}")
    
    for q in dataset.questions:
        print(f"  • {q.id}: {q.question[:50]}...")
    
    print(f"\n🚀 Test with:")
    print(f"python eval/run_evaluation.py {output_path}")


if __name__ == "__main__":
    main()