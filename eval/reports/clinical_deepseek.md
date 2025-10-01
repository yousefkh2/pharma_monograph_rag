# Pharmacy Copilot Evaluation Report

**Dataset**: Working Clinical Test Dataset
**Questions**: 2
**Evaluation Time**: 17.38 seconds
**Average Time per Question**: 8.69 seconds


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
  Answer Length: 66 words

📊 Lexical Overlap:
  BLEU: 0.105
  ROUGE-1 F1: 0.314
  ROUGE-2 F1: 0.151
  ROUGE-L F1: 0.223

🧠 Semantic Similarity: 0.685

🎯 Key Points Coverage: 0.500

# Failure Analysis Report

## Summary
- **Total Failures**: 2
- **By Severity**: {'high': 2}
- **By Root Cause**: {'both_systems': 2}

## Failure Types
- **qa_incomplete**: 2 (100.0%)
- **qa_incorrect**: 1 (50.0%)
- **retrieval_irrelevant**: 2 (100.0%)
- **retrieval_miss**: 2 (100.0%)

## High Severity Failures
### clinical_001
**Question**: What is the maximum daily dose of acetaminophen for adults?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incomplete']
**Root Cause**: both_systems

### clinical_002
**Question**: What are the latest 2024 FDA guidelines for acetaminophen?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incorrect', 'qa_incomplete']
**Root Cause**: both_systems
