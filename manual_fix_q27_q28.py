#!/usr/bin/env python3
"""Manually fix Q27 and Q28 based on successful curl test results"""
import json

# Load existing results
with open('eval/runs_selector/selector1.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# From your earlier successful curl tests:
# Q27: Selected "ipratropium-and-albuterol-drug-information" with confidence 0.4
# Q28: Selected "ofloxacin-ophthalmic-drug-information" with confidence 0.95

manual_fixes = {
    'Q27': {
        'selected_doc': 'ipratropium-and-albuterol-drug-information',
        'score': 0.4
    },
    'Q28': {
        'selected_doc': 'ofloxacin-ophthalmic-drug-information',
        'score': 0.95
    }
}

for i, result in enumerate(results):
    if result['qid'] in manual_fixes:
        fix = manual_fixes[result['qid']]
        # Create a minimal result with just the selected document
        # (The actual aggregated scores would require re-running the full query)
        results[i]['results'] = [
            {
                'doc_id': fix['selected_doc'],
                'title': '',
                'score': fix['score']
            }
        ]
        print(f"Fixed {result['qid']}: {fix['selected_doc']} (score: {fix['score']})")

# Save updated results
with open('eval/runs_selector/selector1.jsonl', 'w') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print("\n✓ Updated selector1.jsonl with manual fixes")
print("Note: Q27 and Q28 only contain the LLM-selected document, not full rankings")
