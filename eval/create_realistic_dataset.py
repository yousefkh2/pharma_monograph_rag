#!/usr/bin/env python3
"""Helper script to create realistic evaluation questions with real chunk IDs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.dataset_schema import EvalQuestion, GroundTruthChunk, QuestionType, DifficultyLevel, EvalDataset


def load_chunk_metadata(metadata_path: Path) -> List[dict]:
    """Load chunk metadata to find real chunk IDs."""
    chunks = []
    with metadata_path.open("r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def find_chunks_by_keywords(chunks: List[dict], keywords: List[str]) -> List[dict]:
    """Find chunks containing specific keywords."""
    matching_chunks = []
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        if any(keyword.lower() in text for keyword in keywords):
            matching_chunks.append(chunk)
    return matching_chunks


def create_realistic_questions(chunks: List[dict]) -> List[EvalQuestion]:
    """Create realistic questions using actual chunk data."""
    
    questions = []
    
    # Find acetaminophen chunks for dosage question
    acetaminophen_chunks = find_chunks_by_keywords(chunks, ["acetaminophen", "dosage", "maximum"])
    if acetaminophen_chunks:
        questions.append(EvalQuestion(
            id="q001_real",
            question="What is the maximum daily dose of acetaminophen for adults?",
            question_type=QuestionType.DOSAGE,
            difficulty=DifficultyLevel.EASY,
            expected_answer="The maximum daily dose of acetaminophen for adults is 4000 mg (4 grams) per day.",
            answer_key_points=["4000 mg", "maximum daily dose", "adults"],
            relevant_chunks=[
                GroundTruthChunk(chunk["chunk_id"], 1.0, True, "Primary dosage information") 
                for chunk in acetaminophen_chunks[:3]  # Take first 3 matching chunks
            ],
            drug_names=["acetaminophen", "paracetamol"],
            tags=["dosage", "otc"],
            source_note="Based on real chunk data"
        ))
    
    # Find ibuprofen contraindication chunks
    ibuprofen_chunks = find_chunks_by_keywords(chunks, ["ibuprofen", "contraindication", "avoid"])
    if ibuprofen_chunks:
        questions.append(EvalQuestion(
            id="q002_real",
            question="What are the contraindications for ibuprofen?",
            question_type=QuestionType.CONTRAINDICATIONS,
            difficulty=DifficultyLevel.MEDIUM,
            expected_answer="Ibuprofen is contraindicated in patients with active GI bleeding, severe heart failure, severe kidney disease, and in the third trimester of pregnancy.",
            answer_key_points=["active GI bleeding", "severe heart failure", "severe kidney disease", "third trimester pregnancy"],
            relevant_chunks=[
                GroundTruthChunk(chunk["chunk_id"], 1.0, True, "Contraindication information")
                for chunk in ibuprofen_chunks[:2]
            ],
            drug_names=["ibuprofen"],
            tags=["contraindications", "nsaid"],
            source_note="Based on real chunk data"
        ))
    
    # Find warfarin interaction chunks
    warfarin_chunks = find_chunks_by_keywords(chunks, ["warfarin", "interaction", "bleeding"])
    if warfarin_chunks:
        questions.append(EvalQuestion(
            id="q003_real",
            question="What drugs interact with warfarin to increase bleeding risk?",
            question_type=QuestionType.INTERACTIONS,
            difficulty=DifficultyLevel.HARD,
            expected_answer="Drugs that interact with warfarin to increase bleeding risk include aspirin, NSAIDs, heparin, and other anticoagulants.",
            answer_key_points=["aspirin", "NSAIDs", "bleeding risk", "anticoagulants"],
            relevant_chunks=[
                GroundTruthChunk(chunk["chunk_id"], 1.0, True, "Warfarin interactions")
                for chunk in warfarin_chunks[:3]
            ],
            drug_names=["warfarin"],
            tags=["interactions", "bleeding", "anticoagulation"],
            source_note="Based on real chunk data"
        ))
    
    return questions


def main():
    metadata_path = Path("vector_store/chunk_metadata.jsonl")
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please run the ingestion pipeline first to generate chunk metadata.")
        return 1
    
    print("Loading chunk metadata...")
    chunks = load_chunk_metadata(metadata_path)
    print(f"Loaded {len(chunks)} chunks")
    
    print("Creating realistic evaluation questions...")
    questions = create_realistic_questions(chunks)
    
    if not questions:
        print("No suitable chunks found for creating questions.")
        print("The metadata may not contain the expected drug information.")
        return 1
    
    # Create dataset
    dataset = EvalDataset(
        name="pharmacy_copilot_realistic_eval_v1",
        description="Realistic evaluation dataset with actual chunk IDs from the corpus",
        version="1.0.0",
        questions=questions
    )
    
    # Save dataset
    output_path = Path("eval/datasets/realistic_eval_v1.json")
    dataset.save(output_path)
    
    print(f"✅ Created realistic evaluation dataset with {len(questions)} questions")
    print(f"💾 Saved to: {output_path}")
    
    # Show summary
    for question in questions:
        print(f"\n📝 {question.id}: {question.question}")
        print(f"   Type: {question.question_type.value}, Difficulty: {question.difficulty.value}")
        print(f"   Relevant chunks: {len(question.relevant_chunks)}")
        if question.relevant_chunks:
            print(f"   Example chunk ID: {question.relevant_chunks[0].chunk_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())