# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** 2025-10-02 04:23:00

## 📊 Executive Summary

- **Total Questions:** 13
- **Mean Score:** 64.2/100
- **Median Score:** 85.0/100
- **Recall@5:** 92.3%

---

## Question Q001 - Score: 90/100

### ❓ Question
> 5-year-old (18 kg) with acute otitis media. What is the amoxicillin dose and volume using 400 mg/5 mL suspension?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 810,
  "volume_per_dose_mL": 10.1,
  "frequency": "q12h",
  "max_daily_mg": 4000,
  "rounding": "round mL to one decimal",
  "calc_rule": "90 mg/kg/day divided q12h",
  "citations": [
    "lexi-amoxicillin-2018#dosage",
    "lexi-amoxicillin-2018#peds"
  ],
  "notes": "High-dose amoxicillin for pediatric AOM."
}
```

### ✅ Expected Answer
```
Amoxicillin 90 mg/kg/day divided q12h → 810 mg per dose (≈10.1 mL of 400 mg/5 mL), round to 10 mL per dose; max 4 g/day.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the amoxicillin dose, volume, frequency, and acknowledges the maximum daily dose. It cites allowed chunk IDs and does not exceed any contraindications or maximums.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-amoxicillin-2018#dosage, lexi-amoxicillin-2018#peds


---

## Question Q002 - Score: 90/100

### ❓ Question
> 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 180,
  "volume_per_dose_mL": 5.6,
  "frequency": "q6h",
  "max_daily_mg": 900,
  "rounding": "mL rounded to one decimal",
  "calc_rule": "15 mg/kg/dose, max 75 mg/kg/day",
  "citations": [
    "lexi-acetaminophen-2018#peds-dosing",
    "lexi-acetaminophen-2018#max"
  ],
  "notes": "Use 15 mg/kg/dose for fever in children."
}
```

### ✅ Expected Answer
```
Acetaminophen 15 mg/kg/dose q6h (range 10–15 mg/kg q4–6h) → 180 mg per dose ≈5.6 mL of 160 mg/5 mL; max 75 mg/kg/day (900 mg), and ≤4 g absolute.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly calculates the dose and volume, cites appropriate chunks, and adheres to safety guidelines.

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
  "frequency": "q6h",
  "max_daily_mg": 480,
  "rounding": "mL rounded to whole mL",
  "calc_rule": "10 mg/kg/dose x 12 kg",
  "citations": [
    "lexi-ibuprofen-2018#peds-dosing",
    "lexi-ibuprofen-2018#max"
  ],
  "notes": "Ibuprofen 10 mg/kg/dose q6h for fever in a 12 kg child."
}
```

### ✅ Expected Answer
```
Ibuprofen 10 mg/kg/dose q6–8h → 120 mg per dose (6 mL of 100 mg/5 mL); max 40 mg/kg/day (480 mg). Avoid if <6 months or dehydrated/renal issues.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the ibuprofen dose, volume, frequency, and maximum daily dose, and cites allowed chunks. However, it uses 'q6h' instead of 'q6–8h', which is a minor deviation.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-ibuprofen-2018#max, lexi-ibuprofen-2018#peds-dosing


---

## Question Q004 - Score: 0/100

### ❓ Question
> Elderly patient with nonvalvular AF: 82 years, 58 kg, SCr 1.6 mg/dL. What apixaban dose?

### 🤖 LLM Generated Answer
```json
```json
{"dose_per_dose_mg":2.5,"volume_per_dose_mL":null,"frequency":"q12h","max_daily_mg":5,"rounding":"whole mg","calc_rule":"2 of 3 criteria met (age ≥80, weight ≤60 kg, SCr ≥1.5 mg/dL)","citations":["lexi-apixaban-2018#dosing-nvaf"],"notes":"Reduced dose due to age and renal impairment."}
```
```

### ✅ Expected Answer
```
Apixaban 2.5 mg BID (meets 2 of 3: age ≥80, weight ≤60 kg, SCr ≥1.5). Avoid with strong CYP3A4/P-gp inhibitors.
```

### ⚖️ Judge's Assessment
- **Score:** 0/100
- **Verdict:** `manual_review`
- **Rationale:** Judge response not JSON: ```json
{
  "score": 75,
  "verdict": "manual_review",
  "rationale": "The answer correctly identifies the dose of 2.5 m...

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
  "max_daily_mg": null,
  "rounding": "whole mg",
  "calc_rule": "1 mg/kg q12h",
  "citations": [
    "lexi-enoxaparin-2018#dosing"
  ],
  "notes": "Enoxaparin 1 mg/kg SC q12h is preferred for acute DVT treatment."
}
```

### ✅ Expected Answer
```
Enoxaparin 1 mg/kg SC q12h → 85 mg per dose (or 1.5 mg/kg once daily as alternative). If CrCl <30 mL/min: 1 mg/kg once daily.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the enoxaparin treatment dose of 1 mg/kg q12h, resulting in 85 mg per dose, and cites an allowed chunk ID. It also mentions the alternative dosing and renal adjustment, fulfilling the key points.

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
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "Contraindicated if eGFR <30 mL/min/1.73 m\u00b2",
  "citations": [
    "lexi-metformin-2018#renal"
  ],
  "notes": "Metformin is contraindicated in this patient due to eGFR <30 mL/min/1.73 m\u00b2."
}
```

### ✅ Expected Answer
```
No—contraindicated at eGFR <30. If 30–45, generally avoid initiation; if continuing, reduce dose and monitor. Avoid in hepatic impairment.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies that metformin is contraindicated at eGFR <30, but it does not mention the recommendation for patients with eGFR 30–45 or the need to avoid in hepatic impairment, which are key points from the gold reference.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-metformin-2018#contra, lexi-metformin-2018#renal


---

## Question Q007 - Score: 90/100

### ❓ Question
> Uncomplicated cystitis: Should we use nitrofurantoin if eGFR is 25?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "Avoid if eGFR <30 mL/min",
  "citations": [
    "lexi-nitrofurantoin-2018#renal",
    "lexi-nitrofurantoin-2018#contra"
  ],
  "notes": "Nitrofurantoin is contraindicated in eGFR <30 mL/min."
}
```

### ✅ Expected Answer
```
Avoid nitrofurantoin when eGFR/CrCl <30 mL/min due to low urinary levels and toxicity risk.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly advises against using nitrofurantoin with eGFR <30 mL/min, cites appropriate evidence chunks, and addresses safety concerns.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-nitrofurantoin-2018#contra, lexi-nitrofurantoin-2018#renal


---

## Question Q008 - Score: 75/100

### ❓ Question
> Patient on warfarin needs TMP-SMX for UTI. Any interaction and what should we do?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "",
  "citations": [
    "lexi-warfarin-2018#ddi-tmp-smx"
  ],
  "notes": "TMP-SMX increases warfarin effect; prefer alternatives or monitor INR closely."
}
```

### ✅ Expected Answer
```
Major interaction (CYP2C9 inhibition + gut flora). Avoid if possible; otherwise increase INR monitoring and reduce warfarin dose.
```

### ⚖️ Judge's Assessment
- **Score:** 75/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and cites the appropriate evidence chunk, but it lacks a detailed explanation of the mechanism and action, which affects the completeness of the response.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-warfarin-2018#ddi-tmp-smx


---

## Question Q009 - Score: 0/100

### ❓ Question
> Patient on atorvastatin needs clarithromycin. What’s the risk and action?

### 🤖 LLM Generated Answer
```json
```json
{"dose_per_dose_mg":null,"volume_per_dose_mL":null,"frequency":"","max_daily_mg":null,"rounding":"","calc_rule":"","citations":["lexi-atorvastatin-2018#ddi-macrolides"],"notes":"Avoid clarithromycin due to strong CYP3A4 inhibition and increased myopathy risk; if unavoidable, use lowest atorvastatin dose with monitoring."}
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
  "score": 80,
  "verdict": "manual_review",
  "rationale": "The answer correctly identifies the major interac...

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-atorvastatin-2018#ddi-macrolides


---

## Question Q010 - Score: 0/100

### ❓ Question
> Starting linezolid in a patient on sertraline. Is this safe?

### 🤖 LLM Generated Answer
```json
```json
{"dose_per_dose_mg":null,"volume_per_dose_mL":null,"frequency":"","max_daily_mg":null,"rounding":"","calc_rule":"","citations":["lexi-linezolid-2018#ddi-ssri"],"notes":"Avoid linezolid with sertraline due to risk of serotonin syndrome."}
```
```

### ✅ Expected Answer
```
Major interaction: risk of serotonin syndrome (linezolid has MAOI activity). Avoid; if benefits outweigh risks, withhold SSRI or monitor very closely.
```

### ⚖️ Judge's Assessment
- **Score:** 0/100
- **Verdict:** `manual_review`
- **Rationale:** Judge response not JSON: ```json
{
  "score": 80,
  "verdict": "manual_review",
  "rationale": "The answer correctly identifies the major interac...

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-linezolid-2018#ddi-ssri


---

## Question Q011 - Score: 85/100

### ❓ Question
> Patient on digoxin will start amiodarone. What’s the interaction management?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "",
  "citations": [
    "lexi-digoxin-2018#ddi-amio"
  ],
  "notes": "Reduce digoxin dose by 30\u201350% and monitor serum levels, renal function, and ECG."
}
```

### ✅ Expected Answer
```
Major interaction (P-gp inhibition). Reduce digoxin dose by ~30–50% and monitor levels/ECG.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the major interaction and appropriate management steps, including dose reduction and monitoring, while citing an allowed chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-digoxin-2018#ddi-amio


---

## Question Q012 - Score: 70/100

### ❓ Question
> COVID-19 patient asks for Paxlovid dosing and interaction checks.

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "",
  "citations": [
    "kb-scope-2018#covid-absent"
  ],
  "notes": "Context predates COVID-19 therapies; refer to current guidance."
}
```

### ✅ Expected Answer
```
ABSTAIN. Not in ≤2018 corpus; recency-sensitive therapy. Advise consulting current guidance/pharmacist.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly abstains and cites a relevant chunk ID, but does not explicitly mention advising a pharmacist or current guidance.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** 


---

## Question Q013 - Score: 85/100

### ❓ Question
> What is the current first-line pharmacologic therapy per latest hypertension guidelines?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "",
  "calc_rule": "",
  "citations": [
    "lexi-hypertension-2014#jnc8"
  ],
  "notes": "JNC8 (2014) is outdated for current hypertension guidelines."
}
```

### ✅ Expected Answer
```
ABSTAIN if asked for ‘current’ guidance (2018 corpus includes JNC8/2014). Flag recency risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly abstains from providing current guidelines, recognizes JNC8 as outdated, and cites an allowed chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-hypertension-2014#jnc8


---

