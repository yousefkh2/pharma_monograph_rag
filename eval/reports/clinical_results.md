# Pharmacy Copilot Evaluation Report

**Dataset**: Working Clinical Test Dataset
**Questions**: 2
**Evaluation Time**: 0.48 seconds
**Average Time per Question**: 0.24 seconds


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
  Total Relevant: 1
  Total Essential: 1


=== Aggregated QA Metrics ===

📝 Answer Quality:
  Has Answer: True
  Answer Length: 68 words

📊 Lexical Overlap:
  BLEU: 0.019
  ROUGE-1 F1: 0.125
  ROUGE-2 F1: 0.040
  ROUGE-L F1: 0.106

🧠 Semantic Similarity: 0.419

🎯 Key Points Coverage: 0.000

# Failure Analysis Report

## Summary
- **Total Failures**: 2
- **By Severity**: {'high': 2}
- **By Root Cause**: {'both_systems': 2}

## Failure Types
- **e2e_low_quality**: 2 (100.0%)
- **qa_incomplete**: 2 (100.0%)
- **qa_incorrect**: 2 (100.0%)
- **retrieval_irrelevant**: 2 (100.0%)
- **retrieval_miss**: 2 (100.0%)

## High Severity Failures
### clinical_001
**Question**: What is the maximum daily dose of acetaminophen for adults?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete', 'e2e_low_quality']
**Root Cause**: both_systems

### clinical_002
**Question**: What are the latest 2024 FDA guidelines for acetaminophen?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete', 'e2e_low_quality']
**Root Cause**: both_systems
