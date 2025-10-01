#!/usr/bin/env python3
"""
Analyze chunks for overlaps and provide statistics about chunking strategy.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import re

def load_chunks(metadata_path: str) -> List[dict]:
    """Load chunk metadata from JSONL file."""
    chunks = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
    return chunks

def analyze_text_overlaps(chunks: List[dict], min_overlap_chars: int = 50) -> Dict:
    """Analyze text overlaps between chunks."""
    doc_chunks = defaultdict(list)
    
    # Group chunks by document
    for chunk in chunks:
        doc_chunks[chunk['doc_id']].append(chunk)
    
    overlap_stats = {
        'total_docs': len(doc_chunks),
        'docs_with_overlaps': 0,
        'total_overlaps': 0,
        'overlap_details': [],
        'overlap_lengths': []
    }
    
    for doc_id, doc_chunk_list in doc_chunks.items():
        # Sort chunks by section and chunk number
        doc_chunk_list.sort(key=lambda x: (
            int(re.search(r'S(\d+)', x['chunk_id']).group(1)),
            int(re.search(r'C(\d+)', x['chunk_id']).group(1))
        ))
        
        doc_overlaps = 0
        for i in range(len(doc_chunk_list) - 1):
            current_chunk = doc_chunk_list[i]
            next_chunk = doc_chunk_list[i + 1]
            
            current_text = current_chunk['text']
            next_text = next_chunk['text']
            
            # Find longest common substring at the end of current and start of next
            overlap = find_overlap(current_text, next_text)
            
            if len(overlap) >= min_overlap_chars:
                doc_overlaps += 1
                overlap_stats['total_overlaps'] += 1
                overlap_stats['overlap_lengths'].append(len(overlap))
                overlap_stats['overlap_details'].append({
                    'doc_id': doc_id,
                    'chunk1': current_chunk['chunk_id'],
                    'chunk2': next_chunk['chunk_id'],
                    'overlap_length': len(overlap),
                    'overlap_text': overlap[:100] + '...' if len(overlap) > 100 else overlap
                })
        
        if doc_overlaps > 0:
            overlap_stats['docs_with_overlaps'] += 1
    
    return overlap_stats

def find_overlap(text1: str, text2: str) -> str:
    """Find the longest overlap between the end of text1 and start of text2."""
    max_overlap = ""
    text1_clean = text1.strip()
    text2_clean = text2.strip()
    
    # Check for overlaps of different lengths
    for i in range(1, min(len(text1_clean), len(text2_clean)) + 1):
        suffix = text1_clean[-i:]
        prefix = text2_clean[:i]
        
        if suffix == prefix and len(suffix) > len(max_overlap):
            max_overlap = suffix
    
    return max_overlap

def analyze_chunk_boundaries(chunks: List[dict]) -> Dict:
    """Analyze how chunks are divided (by section, content, etc.)."""
    section_stats = Counter()
    chunk_size_stats = []
    doc_chunk_counts = Counter()
    
    for chunk in chunks:
        section_stats[chunk.get('section_title', 'Unknown')] += 1
        chunk_size_stats.append(len(chunk['text']))
        doc_chunk_counts[chunk['doc_id']] += 1
    
    return {
        'total_chunks': len(chunks),
        'avg_chunk_size': sum(chunk_size_stats) / len(chunk_size_stats),
        'min_chunk_size': min(chunk_size_stats),
        'max_chunk_size': max(chunk_size_stats),
        'avg_chunks_per_doc': sum(doc_chunk_counts.values()) / len(doc_chunk_counts),
        'top_sections': section_stats.most_common(10),
        'chunk_size_distribution': {
            'small_chunks_(<200_chars)': sum(1 for size in chunk_size_stats if size < 200),
            'medium_chunks_(200-1000_chars)': sum(1 for size in chunk_size_stats if 200 <= size < 1000),
            'large_chunks_(1000+_chars)': sum(1 for size in chunk_size_stats if size >= 1000)
        }
    }

def detect_chunking_strategy(chunks: List[dict]) -> str:
    """Try to detect what chunking strategy was used."""
    sample_chunks = chunks[:100]  # Sample for analysis
    
    # Check if chunks follow section boundaries
    section_boundary_chunks = 0
    for chunk in sample_chunks:
        if chunk.get('section_title') and chunk['text'].strip().startswith(chunk['section_title']):
            section_boundary_chunks += 1
    
    section_boundary_ratio = section_boundary_chunks / len(sample_chunks)
    
    # Check chunk ID patterns
    chunk_ids = [chunk['chunk_id'] for chunk in sample_chunks]
    has_sequential_chunks = any('C01' in cid or 'C02' in cid for cid in chunk_ids)
    
    if section_boundary_ratio > 0.8:
        return "Section-based chunking (chunks align with document sections)"
    elif has_sequential_chunks:
        return "Fixed-size or sliding window chunking (sequential chunks within sections)"
    else:
        return "Custom chunking strategy"

def main():
    metadata_path = "/Users/yusufkhattab3/last/indexes/chunk_metadata.jsonl"
    
    if not Path(metadata_path).exists():
        print(f"Error: {metadata_path} not found")
        sys.exit(1)
    
    print("Loading chunks...")
    chunks = load_chunks(metadata_path)
    print(f"Loaded {len(chunks)} chunks")
    
    print("\n" + "="*60)
    print("CHUNKING STRATEGY ANALYSIS")
    print("="*60)
    
    strategy = detect_chunking_strategy(chunks)
    print(f"Detected strategy: {strategy}")
    
    print("\n" + "="*60)
    print("CHUNK STATISTICS")
    print("="*60)
    
    boundary_stats = analyze_chunk_boundaries(chunks)
    print(f"Total chunks: {boundary_stats['total_chunks']:,}")
    print(f"Average chunk size: {boundary_stats['avg_chunk_size']:.1f} characters")
    print(f"Size range: {boundary_stats['min_chunk_size']} - {boundary_stats['max_chunk_size']} characters")
    print(f"Average chunks per document: {boundary_stats['avg_chunks_per_doc']:.1f}")
    
    print(f"\nChunk size distribution:")
    for size_range, count in boundary_stats['chunk_size_distribution'].items():
        percentage = (count / boundary_stats['total_chunks']) * 100
        print(f"  {size_range}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nTop sections by chunk count:")
    for section, count in boundary_stats['top_sections']:
        print(f"  {section}: {count:,} chunks")
    
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS")
    print("="*60)
    
    print("Analyzing text overlaps (this may take a moment)...")
    overlap_stats = analyze_text_overlaps(chunks, min_overlap_chars=20)
    
    print(f"Documents analyzed: {overlap_stats['total_docs']:,}")
    print(f"Documents with overlaps: {overlap_stats['docs_with_overlaps']:,}")
    print(f"Total overlapping chunk pairs: {overlap_stats['total_overlaps']:,}")
    
    if overlap_stats['total_overlaps'] > 0:
        avg_overlap_length = sum(overlap_stats['overlap_lengths']) / len(overlap_stats['overlap_lengths'])
        print(f"Average overlap length: {avg_overlap_length:.1f} characters")
        print(f"Max overlap length: {max(overlap_stats['overlap_lengths'])} characters")
        
        print(f"\nSample overlaps (showing first 5):")
        for i, detail in enumerate(overlap_stats['overlap_details'][:5]):
            print(f"  {i+1}. {detail['doc_id']}")
            print(f"     {detail['chunk1']} -> {detail['chunk2']}")
            print(f"     Overlap ({detail['overlap_length']} chars): \"{detail['overlap_text']}\"")
            print()
    else:
        print("✅ No significant overlaps detected between adjacent chunks")
    
    # Calculate overlap percentage
    if overlap_stats['total_docs'] > 0:
        overlap_percentage = (overlap_stats['docs_with_overlaps'] / overlap_stats['total_docs']) * 100
        print(f"\nOverlap summary: {overlap_percentage:.1f}% of documents have overlapping chunks")
        
        if overlap_percentage > 50:
            print("⚠️  High overlap detected - you may be using sliding window chunking")
        elif overlap_percentage > 10:
            print("ℹ️  Moderate overlap detected - some chunks may have intentional overlaps")
        else:
            print("✅ Low/no overlap detected - likely using clean section-based chunking")

if __name__ == "__main__":
    main()