# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** 2025-10-02 04:22:50

## 📊 Executive Summary

- **Total Questions:** 13
- **Mean Score:** 73.8/100
- **Median Score:** 80.0/100
- **Recall@5:** 92.3%

---

## Question Q001 - Score: 70/100

### ❓ Question
> 5-year-old (18 kg) with acute otitis media. What is the amoxicillin dose and volume using 400 mg/5 mL suspension?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 810,
  "volume_per_dose_mL": 10.1,
  "frequency": "q12h",
  "max_daily_mg": 4000,
  "rounding": "volume rounded to 10 mL",
  "calc_rule": "80 mg/kg/day \u00d7 18 kg = 1440 mg/day \u00f7 2 doses = 720 mg/dose; rounded to 810 mg/dose to match 400 mg/5 mL suspension",
  "citations": [
    "lexi-amoxicillin-2018#dosage",
    "lexi-amoxicillin-2018#peds"
  ],
  "notes": "High-dose amoxicillin 80\u201390 mg/kg/day for pediatric AOM"
}
```

### ✅ Expected Answer
```
Amoxicillin 90 mg/kg/day divided q12h → 810 mg per dose (≈10.1 mL of 400 mg/5 mL), round to 10 mL per dose; max 4 g/day.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The answer provides a dose of 810 mg and a volume of 10.1 mL, which are acceptable within the specified ranges. However, the calculation for the daily dose is incorrect (1440 mg/day instead of 1620 mg/day), and the rounding note is not aligned with the expected rounding to 10 mL. Citations are correctly included, but the calculation error affects the overall correctness.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-amoxicillin-2018#dosage, lexi-amoxicillin-2018#peds


---

## Question Q002 - Score: 70/100

### ❓ Question
> 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 180,
  "volume_per_dose_mL": 5.6,
  "frequency": "q6h",
  "max_daily_mg": 900,
  "rounding": "volume rounded to one decimal",
  "calc_rule": "15 mg/kg/dose \u00d7 12 kg = 180 mg/dose; 180 mg \u00f7 32 mg/mL = 5.625 mL (rounded to 5.6 mL); 180 mg/dose \u00d7 5 doses = 900 mg/day (\u226475 mg/kg/day \u00d7 12 kg)",
  "citations": [
    "lexi-acetaminophen-2018#peds-dosing",
    "lexi-acetaminophen-2018#max"
  ],
  "notes": "Use 15 mg/kg/dose q6h for fever in a 12 kg child."
}
```

### ✅ Expected Answer
```
Acetaminophen 15 mg/kg/dose q6h (range 10–15 mg/kg q4–6h) → 180 mg per dose ≈5.6 mL of 160 mg/5 mL; max 75 mg/kg/day (900 mg), and ≤4 g absolute.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly calculated the per-dose mg and volume, but the max daily mg exceeds the gold max of 900 mg/day, which is a critical error.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-acetaminophen-2018#max, lexi-acetaminophen-2018#peds-dosing


---

## Question Q003 - Score: 85/100

### ❓ Question
> 12 kg child with fever: ibuprofen dose and volume using 100 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 120,
  "volume_per_dose_mL": 6,
  "frequency": "q6h-q8h",
  "max_daily_mg": 480,
  "rounding": "round to nearest 5 mL",
  "calc_rule": "10 mg/kg/dose, q6-8h, max 40 mg/kg/day",
  "citations": [
    "lexi-ibuprofen-2018#peds-dosing",
    "lexi-ibuprofen-2018#max"
  ],
  "notes": "10 mg/kg per dose, q6-8h; max 40 mg/kg/day; avoid in dehydration or significant renal risk; not applicable to adults"
}
```

### ✅ Expected Answer
```
Ibuprofen 10 mg/kg/dose q6–8h → 120 mg per dose (6 mL of 100 mg/5 mL); max 40 mg/kg/day (480 mg). Avoid if <6 months or dehydrated/renal issues.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the ibuprofen dose, volume, frequency, and maximum daily dose, and cites allowed chunks. However, it incorrectly states 'round to nearest 5 mL' instead of '6 mL at 20 mg/mL'.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-ibuprofen-2018#max, lexi-ibuprofen-2018#peds-dosing


---

## Question Q004 - Score: 85/100

### ❓ Question
> Elderly patient with nonvalvular AF: 82 years, 58 kg, SCr 1.6 mg/dL. What apixaban dose?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 2.5,
  "volume_per_dose_mL": null,
  "frequency": "BID",
  "max_daily_mg": null,
  "rounding": "Rounded to nearest 0.5 mg",
  "calc_rule": "Reduce NVAF dose to 2.5 mg BID if patient has ≥2 of: age ≥80 years, weight ≤60 kg, or serum creatinine ≥1.5 mg/dL",
  "citations": ["lexi-apixaban-2018#dosing-nvaf"],
  "notes": "Avoid strong CYP3A4/P-gp inhibitors/inducers and assess hepatic impairment."
```

### ✅ Expected Answer
```
Apixaban 2.5 mg BID (meets 2 of 3: age ≥80, weight ≤60 kg, SCr ≥1.5). Avoid with strong CYP3A4/P-gp inhibitors.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the apixaban dose of 2.5 mg BID based on the criteria met, cites an allowed chunk ID, and includes relevant notes. It also adheres to the maximum daily dose and does not exceed contraindications.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-apixaban-2018#ddi, lexi-apixaban-2018#dosing-nvaf


---

## Question Q005 - Score: 85/100

### ❓ Question
> 85 kg adult with acute DVT: What is the enoxaparin treatment dose (normal renal function)?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 85,
  "volume_per_dose_mL": null,
  "frequency": "q12h",
  "max_daily_mg": 170,
  "rounding": "none",
  "calc_rule": "1 mg/kg SC q12h (preferred)",
  "citations": [
    "lexi-enoxaparin-2018#dosing"
  ],
  "notes": "Enoxaparin is dosed by weight for acute DVT treatment."
}
```

### ✅ Expected Answer
```
Enoxaparin 1 mg/kg SC q12h → 85 mg per dose (or 1.5 mg/kg once daily as alternative). If CrCl <30 mL/min: 1 mg/kg once daily.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the enoxaparin treatment dose of 1 mg/kg SC q12h, calculates the per-dose amount as 85 mg, and cites an allowed chunk ID. It also mentions the alternative dosing and renal adjustment, fulfilling the requirements.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-enoxaparin-2018#dosing, lexi-enoxaparin-2018#renal


---

## Question Q006 - Score: 80/100

### ❓ Question
> Type 2 diabetes patient with eGFR 28 mL/min/1.73 m². Can we use metformin?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": "Not applicable",
  "calc_rule": "Contraindicated if eGFR <30, avoid initiation if 30-45, continue with dose reduction and monitoring",
  "citations": ["lexi-metformin-2018#renal"],
  "notes": "Metformin contraindicated in eGFR <30 mL/min/1.73 m², avoid initiation in 30-45, and monitor renal function if continuing."
```

### ✅ Expected Answer
```
No—contraindicated at eGFR <30. If 30–45, generally avoid initiation; if continuing, reduce dose and monitor. Avoid in hepatic impairment.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly states that metformin is contraindicated for eGFR <30, but it does not mention the recommendation for patients with eGFR 30–45 or the need to avoid in hepatic impairment, which are key points from the gold reference.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-metformin-2018#contra, lexi-metformin-2018#renal


---

## Question Q007 - Score: 85/100

### ❓ Question
> Uncomplicated cystitis: Should we use nitrofurantoin if eGFR is 25?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": "Not applicable",
  "calc_rule": "Avoid use if eGFR/CrCl <30 due to inadequate urinary concentrations and increased toxicity risk.",
  "citations": ["lexi-nitrofurantoin-2018#renal"],
  "notes": "Avoid nitrofurantoin in eGFR/CrCl <30 due to efficacy and toxicity concerns. Common contraindications include significant renal dysfunction."
```

### ✅ Expected Answer
```
Avoid nitrofurantoin when eGFR/CrCl <30 mL/min due to low urinary levels and toxicity risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly advises against using nitrofurantoin with eGFR <30, cites relevant evidence, and mentions common contraindications.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-nitrofurantoin-2018#contra, lexi-nitrofurantoin-2018#renal


---

## Question Q008 - Score: 80/100

### ❓ Question
> Patient on warfarin needs TMP-SMX for UTI. Any interaction and what should we do?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": null,
  "calc_rule": null,
  "citations": [
    "lexi-warfarin-2018#ddi-tmp-smx"
  ],
  "notes": "Avoid TMP-SMX with warfarin; if used, reduce warfarin dose and monitor INR closely."
}
```

### ✅ Expected Answer
```
Major interaction (CYP2C9 inhibition + gut flora). Avoid if possible; otherwise increase INR monitoring and reduce warfarin dose.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and suggests monitoring and dose reduction, but does not explicitly mention the mechanism of CYP2C9 inhibition + gut flora change, which is a key point from the gold reference.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-warfarin-2018#ddi-tmp-smx


---

## Question Q009 - Score: 0/100

### ❓ Question
> Patient on atorvastatin needs clarithromycin. What’s the risk and action?

### 🤖 LLM Generated Answer
```json
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "",
  "citations": ["lexi-atorvastatin-2018#ddi-macrolides"],
  "notes": "Clarithromycin increases atorvastatin levels, raising myopathy/rhabdomyolysis risk; avoid or use lowest dose with monitoring."
}
```
```

### ✅ Expected Answer
```
Major interaction (strong CYP3A4 inhibitor). Avoid coadministration; if unavoidable, hold atorvastatin or limit to the lowest dose and monitor for myopathy.
```

### ⚖️ Judge's Assessment
- **Score:** 0/100
- **Verdict:** `manual_review`
- **Rationale:** Judge response not JSON: ```json
{
  "score": 60,
  "verdict": "fail",
  "rationale": "The answer does not correctly identify the major interacti...

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-atorvastatin-2018#ddi-macrolides


---

## Question Q010 - Score: 70/100

### ❓ Question
> Starting linezolid in a patient on sertraline. Is this safe?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": null,
  "calc_rule": null,
  "citations": [
    "lexi-linezolid-2018#ddi-ssri"
  ],
  "notes": "Avoid combination due to risk of serotonin syndrome; if unavoidable, withhold SSRI and monitor closely."
}
```

### ✅ Expected Answer
```
Major interaction: risk of serotonin syndrome (linezolid has MAOI activity). Avoid; if benefits outweigh risks, withhold SSRI or monitor very closely.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the major interaction and cites the appropriate chunk ID, but it lacks specific details on the mechanism and action as outlined in the gold reference.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-linezolid-2018#ddi-ssri


---

## Question Q011 - Score: 75/100

### ❓ Question
> Patient on digoxin will start amiodarone. What’s the interaction management?

### 🤖 LLM Generated Answer
```json

```

### ✅ Expected Answer
```
Major interaction (P-gp inhibition). Reduce digoxin dose by ~30–50% and monitor levels/ECG.
```

### ⚖️ Judge's Assessment
- **Score:** 75/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the major interaction and the need to reduce the digoxin dose by 30-50%, but it lacks specific numeric values for dose and frequency, which are important for correctness. It cites the allowed chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-digoxin-2018#ddi-amio


---

## Question Q012 - Score: 85/100

### ❓ Question
> COVID-19 patient asks for Paxlovid dosing and interaction checks.

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "q12h",
  "max_daily_mg": null,
  "rounding": "Rounded to whole tablet",
  "calc_rule": "Recency risk; refer to current guidance/pharmacist",
  "citations": ["kb-scope-2018#covid-absent"],
  "notes": "This context predates COVID-19 therapies; consult current guidance/pharmacist."
```

### ✅ Expected Answer
```
ABSTAIN. Not in ≤2018 corpus; recency-sensitive therapy. Advise consulting current guidance/pharmacist.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly abstains and advises consulting current guidance, aligning with the expected answer.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** 


---

## Question Q013 - Score: 90/100

### ❓ Question
> What is the current first-line pharmacologic therapy per latest hypertension guidelines?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "qd",
  "max_daily_mg": null,
  "rounding": "Rounded to whole mg",
  "calc_rule": "Recency risk: JNC8 (2014) is outdated for current guideline",
  "citations": [
    "lexi-hypertension-2014#jnc8"
  ],
  "notes": "Current guidelines should be consulted for first-line pharmacologic therapy for hypertension."
}
```

### ✅ Expected Answer
```
ABSTAIN if asked for ‘current’ guidance (2018 corpus includes JNC8/2014). Flag recency risk.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The model correctly abstains from providing current guidelines, recognizes JNC8 as outdated, and cites the appropriate chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-hypertension-2014#jnc8


---

