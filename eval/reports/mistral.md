# Pharmacy Copilot Evaluation Report

**Dataset**: pharmacy_copilot_pharmacist_eval_v1
**Questions**: 13
**Evaluation Time**: 40.81 seconds
**Average Time per Question**: 3.14 seconds


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
  Total Relevant: 19
  Total Essential: 12


=== Aggregated QA Metrics ===

📝 Answer Quality:
  Has Answer: True
  Answer Length: 93 words

📊 Lexical Overlap:
  BLEU: 0.018
  ROUGE-1 F1: 0.200
  ROUGE-2 F1: 0.074
  ROUGE-L F1: 0.165

🧠 Semantic Similarity: 0.570

🎯 Key Points Coverage: 0.208

# Failure Analysis Report

## Summary
- **Total Failures**: 13
- **By Severity**: {'critical': 1, 'high': 12}
- **By Root Cause**: {'both_systems': 13}

## Failure Types
- **e2e_low_quality**: 5 (38.5%)
- **qa_incomplete**: 12 (92.3%)
- **qa_incorrect**: 6 (46.2%)
- **qa_no_answer**: 1 (7.7%)
- **retrieval_irrelevant**: 13 (100.0%)
- **retrieval_miss**: 13 (100.0%)

## Critical Failures
### q004
**Question**: Elderly patient with nonvalvular AF: 82 years, 58 kg, SCr 1.6 mg/dL. What apixaban dose?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_no_answer', 'e2e_low_quality']
**Root Cause**: both_systems
**Generated**: ...
**Expected**: Apixaban 2.5 mg BID (meets 2 of 3: age ≥80, weight ≤60 kg, SCr ≥1.5). Avoid with strong CYP3A4/P-gp ...

## High Severity Failures
### q001
**Question**: 5-year-old (18 kg) with acute otitis media. What is the amoxicillin dose and volume using 400 mg/5 mL suspension?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incomplete']
**Root Cause**: both_systems

### q002
**Question**: 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incomplete']
**Root Cause**: both_systems

### q003
**Question**: 12 kg child with fever: ibuprofen dose and volume using 100 mg/5 mL?
**Failure Types**: ['retrieval_miss', 'retrieval_irrelevant', 'qa_incomplete']
**Root Cause**: both_systems
