#!/usr/bin/env python3
"""Fix Q27 and Q28 in selector1.jsonl by re-running them"""
import json
import httpx
from collections import defaultdict

# Load existing results
with open('eval/runs_selector/selector1.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Queries to fix
queries_to_fix = {
    'Q27': "Asthma rescue — dosing and spacer/technique notes for albuterol HFA; maximum puffs/day.",
    'Q28': "Bacterial conjunctivitis — dosing and contact lens precautions for ofloxacin ophthalmic solution."
}

client = httpx.Client(base_url='http://localhost:8001', timeout=180.0)

for i, result in enumerate(results):
    if result['qid'] not in queries_to_fix:
        continue
        
    query = queries_to_fix[result['qid']]
    print(f"Processing {result['qid']}: {query[:70]}...")
    
    try:
        resp = client.post('/rerank', json={
            'query': query,
            'top_k': 20,
            'dense_weight': 0.6,
            'bm25_candidates': 200,
            'dense_candidates': 50,
        })
        resp.raise_for_status()
        data = resp.json()
        
        # Aggregate scores per monograph
        per_doc = defaultdict(list)
        for hit in data.get('original_results', []):
            doc_id = hit.get('doc_id')
            if doc_id:
                per_doc[doc_id].append(float(hit.get('score', 0)))
        
        # Average top 3 scores per doc
        mono_scores = {}
        for doc_id, scores in per_doc.items():
            scores.sort(reverse=True)
            top3 = scores[:3]
            mono_scores[doc_id] = sum(top3) / len(top3) if top3 else 0.0
        
        # Build results, sorted by score
        ranked_docs = sorted(mono_scores.items(), key=lambda kv: kv[1], reverse=True)
        new_results = [{'doc_id': d, 'title': '', 'score': s} for d, s in ranked_docs]
        
        # Put LLM-selected doc first (force rank 1)
        sel_doc = data.get('selection', {}).get('doc_id')
        if sel_doc:
            existing = next((r for r in new_results if r['doc_id'] == sel_doc), None)
            if existing:
                new_results.remove(existing)
            else:
                existing = {'doc_id': sel_doc, 'title': '', 'score': float('inf')}
            new_results.insert(0, existing)
        
        results[i]['results'] = new_results
        print(f"  ✓ Got {len(new_results)} results")
        print(f"  ✓ LLM selected: {sel_doc}")
        print(f"  ✓ Top 5 monographs: {[r['doc_id'] for r in new_results[:5]]}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

client.close()

# Save updated results
with open('eval/runs_selector/selector1.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"\n✓ Updated selector1.jsonl")
