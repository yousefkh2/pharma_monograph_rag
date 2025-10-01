#!/usr/bin/env python3
"""
Fix chunk ID mismatches in evaluation dataset by mapping conceptual references
to actual system chunk IDs.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict

def load_system_metadata(metadata_path: str) -> Dict[str, dict]:
    """Load actual system chunk metadata."""
    chunks = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line.strip())
                chunks[chunk['chunk_id']] = chunk
    return chunks

def create_drug_mapping() -> Dict[str, str]:
    """Create mapping from eval dataset drug names to actual doc_ids."""
    return {
        'lexi-amoxicillin-2018': 'amoxicillin-drug-information',
        'lexi-acetaminophen-2018': 'acetaminophen-paracetamol-drug-information',
        'lexi-ibuprofen-2018': 'ibuprofen-drug-information',
        'lexi-apixaban-2018': 'apixaban-drug-information',
        'lexi-warfarin-2018': 'warfarin-drug-information',
        'lexi-aspirin-2018': 'aspirin-drug-information',
        'lexi-clopidogrel-2018': 'clopidogrel-drug-information',
        'lexi-atorvastatin-2018': 'atorvastatin-drug-information',
        'lexi-metformin-2018': 'metformin-drug-information',
        'lexi-lisinopril-2018': 'lisinopril-drug-information',
        'lexi-albuterol-2018': 'albuterol-drug-information',
        'lexi-prednisone-2018': 'prednisone-drug-information',
    }

def create_section_mapping() -> Dict[str, List[str]]:
    """Create mapping from eval section names to actual section titles."""
    return {
        'dosage': ['Dosing: Adult', 'Dosing: Geriatric', 'Dosing: Pediatric', 'Dosing'],
        'peds': ['Dosing: Pediatric', 'Pediatric Considerations'],
        'peds-dosing': ['Dosing: Pediatric'],
        'max': ['Maximum Daily Dose', 'Dosing: Adult', 'Administration'],
        'interactions': ['Drug Interactions'],
        'contraindications': ['Contraindications'],
        'warnings': ['Warnings/Precautions'],
        'admin': ['Administration'],
        'monitoring': ['Monitoring Parameters'],
        'pregnancy': ['Pregnancy Implications'],
        'renal': ['Dosing: Renal Impairment'],
        'hepatic': ['Dosing: Hepatic Impairment'],
    }

def find_matching_chunks(eval_chunk_id: str, system_chunks: Dict[str, dict]) -> List[str]:
    """Find system chunks that match an eval dataset chunk reference."""
    
    # Parse eval chunk ID: "lexi-drug-2018#section"
    if '#' not in eval_chunk_id:
        print(f"Warning: Invalid eval chunk ID format: {eval_chunk_id}")
        return []
    
    drug_part, section_part = eval_chunk_id.split('#', 1)
    
    # Map drug name
    drug_mapping = create_drug_mapping()
    if drug_part not in drug_mapping:
        print(f"Warning: Unknown drug in eval dataset: {drug_part}")
        return []
    
    target_doc_id = drug_mapping[drug_part]
    
    # Map section name
    section_mapping = create_section_mapping()
    if section_part not in section_mapping:
        print(f"Warning: Unknown section in eval dataset: {section_part}")
        return []
    
    target_sections = section_mapping[section_part]
    
    # Find matching chunks
    matches = []
    for chunk_id, chunk_data in system_chunks.items():
        if chunk_data['doc_id'] == target_doc_id:
            section_title = chunk_data.get('section_title', '')
            if any(target_sec.lower() in section_title.lower() for target_sec in target_sections):
                matches.append(chunk_id)
    
    return matches

def analyze_current_mappings(eval_dataset_path: str, system_metadata_path: str):
    """Analyze what mappings are possible with current data."""
    
    # Load data
    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)
    system_chunks = load_system_metadata(system_metadata_path)
    
    print("=== Current Mapping Analysis ===")
    
    # Collect all eval chunk IDs
    eval_chunk_ids = set()
    for question in eval_data['questions']:
        for chunk in question['relevant_chunks']:
            eval_chunk_ids.add(chunk['chunk_id'])
    
    print(f"Unique eval chunk IDs: {len(eval_chunk_ids)}")
    
    # Try to map each one
    mapping_results = {}
    for eval_chunk_id in sorted(eval_chunk_ids):
        matches = find_matching_chunks(eval_chunk_id, system_chunks)
        mapping_results[eval_chunk_id] = matches
        
        print(f"\n{eval_chunk_id}:")
        if matches:
            print(f"  ✅ Found {len(matches)} matches:")
            for match in matches[:3]:  # Show first 3
                chunk_data = system_chunks[match]
                print(f"    {match} ({chunk_data.get('section_title', 'No title')})")
            if len(matches) > 3:
                print(f"    ... and {len(matches) - 3} more")
        else:
            print(f"  ❌ No matches found")
    
    return mapping_results

def fix_dataset(eval_dataset_path: str, system_metadata_path: str, output_path: str):
    """Fix the evaluation dataset with correct chunk IDs."""
    
    # Load data
    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)
    system_chunks = load_system_metadata(system_metadata_path)
    
    print("=== Fixing Dataset ===")
    
    fixed_count = 0
    total_chunks = 0
    
    for question in eval_data['questions']:
        print(f"\nFixing question {question['id']}...")
        
        new_relevant_chunks = []
        for chunk_ref in question['relevant_chunks']:
            total_chunks += 1
            eval_chunk_id = chunk_ref['chunk_id']
            
            # Find matching system chunks
            matches = find_matching_chunks(eval_chunk_id, system_chunks)
            
            if matches:
                # Use the first match (could be improved with more sophisticated selection)
                new_chunk_id = matches[0]
                new_chunk_ref = {
                    'chunk_id': new_chunk_id,
                    'relevance_score': chunk_ref['relevance_score'],
                    'is_essential': chunk_ref['is_essential']
                }
                if 'note' in chunk_ref:
                    new_chunk_ref['note'] = chunk_ref['note']
                
                new_relevant_chunks.append(new_chunk_ref)
                fixed_count += 1
                
                print(f"  ✅ {eval_chunk_id} → {new_chunk_id}")
                
                # If multiple matches, add them too (with lower relevance)
                for additional_match in matches[1:3]:  # Add up to 2 more
                    additional_ref = {
                        'chunk_id': additional_match,
                        'relevance_score': max(0.5, chunk_ref['relevance_score'] - 0.2),
                        'is_essential': False
                    }
                    new_relevant_chunks.append(additional_ref)
                    print(f"  ➕ Additional: {additional_match}")
            else:
                print(f"  ❌ No match for {eval_chunk_id}")
                # Keep original for manual review
                new_relevant_chunks.append(chunk_ref)
        
        question['relevant_chunks'] = new_relevant_chunks
    
    # Save fixed dataset
    with open(output_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total chunk references: {total_chunks}")
    print(f"Successfully mapped: {fixed_count}")
    print(f"Success rate: {fixed_count/total_chunks*100:.1f}%")
    print(f"Fixed dataset saved to: {output_path}")

def main():
    eval_dataset_path = "/Users/yusufkhattab3/last/eval/datasets/pharmacist_eval_v1.json"
    system_metadata_path = "/Users/yusufkhattab3/last/indexes/chunk_metadata.jsonl"
    output_path = "/Users/yusufkhattab3/last/eval/datasets/pharmacist_eval_v1_fixed.json"
    
    print("Step 1: Analyzing current mappings...")
    mapping_results = analyze_current_mappings(eval_dataset_path, system_metadata_path)
    
    print("\n" + "="*60)
    print("Step 2: Fixing dataset...")
    fix_dataset(eval_dataset_path, system_metadata_path, output_path)
    
    print(f"\n✅ Done! Use the fixed dataset:")
    print(f"   {output_path}")

if __name__ == "__main__":
    main()