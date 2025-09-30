# Pharmacy Copilot Evaluation Report

**Dataset**: pharmacy_copilot_eval_v1
**Questions**: 3
**Evaluation Time**: 13.78 seconds
**Average Time per Question**: 4.59 seconds


=== Aggregated Retrieval Metrics ===

📊 Recall@k:
  R@1: 0.000
  R@3: 0.000
  R@5: 0.000
  R@10: 0.000
  R@20: 0.000

📊 Precision@k:
  P@1: 0.000
  P@3: 0.000
  P@5: 0.000
  P@10: 0.000
  P@20: 0.000

📊 Essential Recall@k:
  ER@1: 0.000
  ER@3: 0.000
  ER@5: 0.000
  ER@10: 0.000
  ER@20: 0.000

🎯 Ranking Metrics:
  MRR: 0.000
  MAP: 0.000
  First Relevant Rank: 0.0

📈 NDCG@k:
  NDCG@1: 0.000
  NDCG@3: 0.000
  NDCG@5: 0.000
  NDCG@10: 0.000
  NDCG@20: 0.000

📋 Summary:
  Total Relevant: 8
  Total Essential: 5


=== Aggregated QA Metrics ===

📝 Answer Quality:
  Has Answer: True
  Answer Length: 68 words

📊 Lexical Overlap:
  BLEU: 0.015
  ROUGE-1 F1: 0.131
  ROUGE-2 F1: 0.037
  ROUGE-L F1: 0.109

🧠 Semantic Similarity: 0.506

🎯 Key Points Coverage: 0.000

# Failure Analysis Report

## Summary
- **Total Failures**: 3
- **By Severity**: {'high': 3}
- **By Root Cause**: {'both_systems': 3}

## Failure Types
- **e2e_low_quality**: 2 (66.7%)
- **qa_incomplete**: 3 (100.0%)
- **qa_incorrect**: 3 (100.0%)
- **retrieval_irrelevant**: 3 (100.0%)
- **retrieval_miss**: 3 (100.0%)

## High Severity Failures
### q001
**Question**: What is the maximum daily dose of acetaminophen for adults?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete']
**Root Cause**: both_systems

### q002
**Question**: What are the contraindications for ibuprofen?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete', 'e2e_low_quality']
**Root Cause**: both_systems

### q003
**Question**: How does warfarin interact with aspirin?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete', 'e2e_low_quality']
**Root Cause**: both_systems
