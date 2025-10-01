#!/usr/bin/env python3
"""
Fix evaluation dataset chunk IDs to match the actual F-format chunk IDs used by the search service.
The search service uses vector_store/chunk_metadata.jsonl which has F######::## format chunks,
not the S##::C## format chunks from indexes/chunk_metadata.jsonl.
"""
import json
from pathlib import Path
from typing import Dict, List, Set
import re

def load_f_format_metadata() -> List[Dict]:
    """Load the F-format metadata that the search service actually uses."""
    metadata_path = Path("vector_store/chunk_metadata.jsonl")
    metadata = []
    
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    
    return metadata

def create_drug_mapping() -> Dict[str, str]:
    """Map evaluation dataset drug names to actual drug file names."""
    return {
        "lexi-acetaminophen-2018": "acetaminophen-paracetamol-drug-information",
        "lexi-amoxicillin-2018": "amoxicillin-drug-information",
        "lexi-ibuprofen-2018": "ibuprofen-drug-information",
        "lexi-metformin-2018": "metformin-drug-information",
        # Add other drugs as we find them in the F-format data
    }

def find_matching_f_chunks(drug_name: str, section_hint: str, metadata: List[Dict]) -> List[str]:
    """Find F-format chunks that match the drug and section."""
    if not drug_name:
        return []
    
    matching_chunks = []
    
    # Search for chunks from this drug
    for record in metadata:
        chunk_id = record.get("chunk_id", "")
        if not chunk_id.startswith(drug_name):
            continue
            
        section_title = record.get("section_title", "").lower()
        text = record.get("text", "").lower()
        
        # Match based on section hints
        section_matches = False
        
        if section_hint == "dosage" or section_hint == "dosing":
            section_matches = (
                "dosing" in section_title or
                "dosage" in section_title or
                "dose" in text[:200]  # Check first part of text
            )
        elif section_hint == "peds-dosing" or section_hint == "peds":
            section_matches = (
                "pediatric" in section_title or
                "dosing: pediatric" in section_title
            )
        elif section_hint == "max":
            section_matches = (
                "dosing" in section_title and
                ("maximum" in text[:500] or "max" in text[:500])
            )
        elif section_hint == "renal":
            section_matches = (
                "renal" in section_title or
                "renal impairment" in section_title
            )
        
        if section_matches:
            matching_chunks.append(chunk_id)
    
    return matching_chunks

def analyze_current_mappings():
    """Analyze what chunks are available in F-format."""
    print("=== F-Format Mapping Analysis ===")
    
    metadata = load_f_format_metadata()
    drug_mapping = create_drug_mapping()
    
    # Get unique eval chunk IDs from the original dataset
    eval_path = Path("eval/datasets/pharmacist_eval_v1.json")
    eval_chunk_ids = set()
    
    with eval_path.open("r") as f:
        data = json.load(f)
        for question in data.get("questions", []):
            for chunk in question.get("relevant_chunks", []):
                eval_chunk_ids.add(chunk["chunk_id"])
    
    print(f"Unique eval chunk IDs: {len(eval_chunk_ids)}")
    print()
    
    for eval_chunk_id in sorted(eval_chunk_ids):
        # Parse eval chunk ID: lexi-amoxicillin-2018#dosage
        if "#" not in eval_chunk_id:
            print(f"❌ Invalid format: {eval_chunk_id}")
            continue
            
        drug_part, section_part = eval_chunk_id.split("#", 1)
        mapped_drug = drug_mapping.get(drug_part)
        
        if not mapped_drug:
            print(f"❌ Unknown drug: {eval_chunk_id}")
            continue
            
        # Find matching F-format chunks
        matches = find_matching_f_chunks(mapped_drug, section_part, metadata)
        
        if matches:
            print(f"✅ {eval_chunk_id}:")
            for i, match in enumerate(matches[:5]):  # Show first 5 matches
                print(f"    {match}")
            if len(matches) > 5:
                print(f"    ... and {len(matches) - 5} more")
        else:
            print(f"❌ No matches: {eval_chunk_id}")
        print()

def fix_dataset():
    """Fix the evaluation dataset with F-format chunk IDs."""
    print("=== Fixing Dataset ===")
    
    metadata = load_f_format_metadata()
    drug_mapping = create_drug_mapping()
    
    # Load original dataset
    input_path = Path("eval/datasets/pharmacist_eval_v1.json")
    output_path = Path("eval/datasets/pharmacist_eval_v1_fixed_F.json")
    
    with input_path.open("r") as f:
        dataset = json.load(f)
    
    total_refs = 0
    mapped_refs = 0
    
    for question in dataset.get("questions", []):
        question_id = question.get("id", "unknown")
        print(f"\nFixing question {question_id}...")
        
        new_chunks = []
        for chunk_info in question.get("relevant_chunks", []):
            eval_chunk_id = chunk_info["chunk_id"]
            total_refs += 1
            
            if "#" not in eval_chunk_id:
                print(f"  ❌ Invalid format: {eval_chunk_id}")
                continue
                
            drug_part, section_part = eval_chunk_id.split("#", 1)
            mapped_drug = drug_mapping.get(drug_part)
            
            if not mapped_drug:
                print(f"  ❌ Unknown drug: {eval_chunk_id}")
                continue
            
            # Find matching chunks
            matches = find_matching_f_chunks(mapped_drug, section_part, metadata)
            
            if matches:
                # Add the first match as primary
                new_chunks.append({
                    "chunk_id": matches[0],
                    "relevance_score": chunk_info.get("relevance_score", 1.0),
                    "is_essential": chunk_info.get("is_essential", False)
                })
                print(f"  ✅ {eval_chunk_id} → {matches[0]}")
                mapped_refs += 1
                
                # Add additional matches with lower relevance
                for additional_match in matches[1:3]:  # Add up to 2 more
                    new_chunks.append({
                        "chunk_id": additional_match,
                        "relevance_score": min(0.8, chunk_info.get("relevance_score", 1.0)),
                        "is_essential": False
                    })
                    print(f"  ➕ Additional: {additional_match}")
            else:
                print(f"  ❌ No match for {eval_chunk_id}")
        
        question["relevant_chunks"] = new_chunks
    
    # Save fixed dataset
    with output_path.open("w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Summary ===")
    print(f"Total chunk references: {total_refs}")
    print(f"Successfully mapped: {mapped_refs}")  
    print(f"Success rate: {100 * mapped_refs / total_refs if total_refs > 0 else 0:.1f}%")
    print(f"Fixed dataset saved to: {output_path}")
    print(f"✅ Done! Use the fixed dataset:")
    print(f"   {output_path}")

if __name__ == "__main__":
    print("Step 1: Analyzing current F-format mappings...")
    analyze_current_mappings()
    
    print("\n" + "="*60)
    print("Step 2: Fixing dataset...")
    fix_dataset()