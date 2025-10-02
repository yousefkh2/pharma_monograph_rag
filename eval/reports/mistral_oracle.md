# Pharmacy Copilot Evaluation Report

**Dataset**: pharmacy_copilot_pharmacist_eval_v1
**Questions**: 13
**Evaluation Time**: 45.53 seconds
**Average Time per Question**: 3.50 seconds


=== Aggregated Retrieval Metrics ===

📊 Recall@k:
  R@1: 0.654
  R@3: 0.923
  R@5: 0.923
  R@10: 0.923
  R@20: 0.923

📊 Precision@k:
  P@1: 0.923
  P@3: 0.590
  P@5: 0.590
  P@10: 0.590
  P@20: 0.590

📊 Essential Recall@k:
  ER@1: 0.923
  ER@3: 0.923
  ER@5: 0.923
  ER@10: 0.923
  ER@20: 0.923

🎯 Ranking Metrics:
  MRR: 0.923
  MAP: 0.923
  First Relevant Rank: 0.9

📈 NDCG@k:
  NDCG@1: 0.923
  NDCG@3: 0.923
  NDCG@5: 0.923
  NDCG@10: 0.923
  NDCG@20: 0.923

📋 Summary:
  Total Relevant: 19
  Total Essential: 12


=== Aggregated QA Metrics ===

📝 Answer Quality:
  Has Answer: True
  Answer Length: 29 words

📊 Lexical Overlap:
  BLEU: 0.025
  ROUGE-1 F1: 0.323
  ROUGE-2 F1: 0.143
  ROUGE-L F1: 0.224

🧠 Semantic Similarity: 0.536

🎯 Key Points Coverage: 0.171

# Failure Analysis Report

## Summary
- **Total Failures**: 13
- **By Severity**: {'high': 10, 'medium': 3}
- **By Root Cause**: {'both_systems': 5, 'qa_system': 8}

## Failure Types
- **e2e_low_quality**: 6 (46.2%)
- **qa_incomplete**: 13 (100.0%)
- **qa_incorrect**: 10 (76.9%)
- **retrieval_irrelevant**: 5 (38.5%)
- **retrieval_miss**: 1 (7.7%)

## High Severity Failures
### q002
**Question**: 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?
**Failure Types**: ['qa_incorrect', 'qa_incomplete']
**Root Cause**: qa_system

### q003
**Question**: 12 kg child with fever: ibuprofen dose and volume using 100 mg/5 mL?
**Failure Types**: ['qa_incorrect', 'qa_incomplete']
**Root Cause**: qa_system

### q004
**Question**: Elderly patient with nonvalvular AF: 82 years, 58 kg, SCr 1.6 mg/dL. What apixaban dose?
**Failure Types**: ['qa_incorrect', 'qa_incomplete']
**Root Cause**: qa_system
