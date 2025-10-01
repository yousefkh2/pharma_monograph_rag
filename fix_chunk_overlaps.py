#!/usr/bin/env python3
"""
Fix the two minor overlaps found in the chunk analysis.
This script will remove overlapping text from chunks.
"""
import json
import re
from pathlib import Path

def fix_overlaps():
    """Fix the specific overlaps found in the analysis."""
    metadata_path = "/Users/yusufkhattab3/last/indexes/chunk_metadata.jsonl"
    backup_path = "/Users/yusufkhattab3/last/indexes/chunk_metadata_backup.jsonl"
    
    # Create backup
    Path(backup_path).write_text(Path(metadata_path).read_text())
    print(f"✅ Created backup: {backup_path}")
    
    chunks = []
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
    
    fixed_count = 0
    
    # Fix simethicone overlap
    for i, chunk in enumerate(chunks):
        if chunk['chunk_id'] == 'simethicone-drug-information::S08::C02':
            original_text = chunk['text']
            # Remove the overlapping "Gas-X Extra Strength: 125 mg" from beginning
            if original_text.startswith("Gas-X Extra Strength: 125 mg"):
                chunk['text'] = original_text[28:].lstrip()
                fixed_count += 1
                print(f"✅ Fixed simethicone overlap")
                break
    
    # Fix voriconazole overlap  
    for i, chunk in enumerate(chunks):
        if chunk['chunk_id'] == 'voriconazole-drug-information::S24::C01':
            original_text = chunk['text']
            # Remove the overlapping monitoring text from beginning
            overlap_text = "Monitoring of serum trough concentrations is recommended in the following infections: invasive asper"
            if original_text.startswith("Monitoring of serum trough concentrations is recomm"):
                # Find the end of the overlap and trim
                chunk['text'] = original_text[462:].lstrip()
                fixed_count += 1
                print(f"✅ Fixed voriconazole overlap")
                break
    
    # Write back the fixed chunks
    with open(metadata_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"✅ Fixed {fixed_count} overlaps")
    print(f"✅ Updated {metadata_path}")
    print("⚠️  Remember to regenerate embeddings if you use this fix!")

if __name__ == "__main__":
    print("This will fix the 2 minor overlaps found in your chunks.")
    print("A backup will be created first.")
    
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        fix_overlaps()
    else:
        print("Cancelled - no changes made.")