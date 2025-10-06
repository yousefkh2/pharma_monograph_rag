# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** 2025-10-02 06:33:55

## 📊 Executive Summary

- **Total Questions:** 13
- **Mean Score:** 71.9/100
- **Median Score:** 80.0/100
- **Recall@5:** 92.3%

---

## Question Q001 - Score: 80/100

### ❓ Question
> 5-year-old (18 kg) with acute otitis media. What is the amoxicillin dose and volume using 400 mg/5 mL suspension?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 810,
  "volume_per_dose_mL": 10.1,
  "frequency": "q12h",
  "max_daily_mg": 4000,
  "rounding": "rounding to practical mL is acceptable",
  "calc_rule": "High-dose 80-90 mg/kg/day regimen; BID (q12h) frequency; Per-dose dose calculated as (18 kg * 80 mg/kg) / 5 mL (80 mg/mL); rounding to 10 mL OK; Max daily dose acknowledged",
  "citations": ["lexi-amoxicillin-2018#dosage"],
  "notes": "Uses high-dose 80-90 mg/kg/day regimen for pediatric AOM; BID (q12h) frequency; Per-dose ≈810 mg for 18 kg; Volume ≈10.1 mL
```

### ✅ Expected Answer
```
Amoxicillin 90 mg/kg/day divided q12h → 810 mg per dose (≈10.1 mL of 400 mg/5 mL), round to 10 mL per dose; max 4 g/day.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the dose, volume, frequency, and max daily mg, but the rounding rule is not fully aligned with the gold standard. It cites one allowed chunk but misses the second citation for pediatric dosage, which affects evidence coverage.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-amoxicillin-2018#dosage, lexi-amoxicillin-2018#peds


---

## Question Q002 - Score: 60/100

### ❓ Question
> 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 180,
  "volume_per_dose_mL": 5.625,
  "frequency": "q6h",
  "max_daily_mg": 1440,
  "max_daily_mL": null,
  "rounding": "rounded to one decimal",
  "calc_rule": "15 mg/kg per dose, q4-6h, target q6h, 32 mg/mL",
  "citations": [
    "lexi-acetaminophen-2018#peds-dosing",
    "lexi-acetaminophen-2018#max"
  ],
  "notes": "Pediatric dosing; liquid form"
}
```

### ✅ Expected Answer
```
Acetaminophen 15 mg/kg/dose q6h (range 10–15 mg/kg q4–6h) → 180 mg per dose ≈5.6 mL of 160 mg/5 mL; max 75 mg/kg/day (900 mg), and ≤4 g absolute.
```

### ⚖️ Judge's Assessment
- **Score:** 60/100
- **Verdict:** `fail`
- **Rationale:** The model's max_daily_mg of 1440 mg exceeds the gold maximum of 900 mg, resulting in an auto-fail.

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
  "volume_per_dose_mL": 6.0,
  "frequency": "q6-8h",
  "max_daily_mg": 480,
  "rounding": "round to nearest 0.1 mL",
  "calc_rule": "10 mg/kg per dose, maximum 40 mg/kg/day",
  "citations": [
    "lexi-ibuprofen-2018#peds-dosing",
    "lexi-ibuprofen-2018#max"
  ],
  "notes": "12 kg child with fever: 120 mg ibuprofen per dose (6 mL of 100 mg/5 mL suspension), q6-8h; maximum daily dose 480 mg; contraindicated in children under 6 months and use with caution in renal/dehydration risk"
}
```

### ✅ Expected Answer
```
Ibuprofen 10 mg/kg/dose q6–8h → 120 mg per dose (6 mL of 100 mg/5 mL); max 40 mg/kg/day (480 mg). Avoid if <6 months or dehydrated/renal issues.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the ibuprofen dose, volume, frequency, and maximum daily dose, all supported by allowed citations. The rounding rule is slightly misrepresented but does not affect the overall correctness.

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
  "rounding": "Not applicable (tablets)",
  "calc_rule": "Dose reduction criteria: 2 of 3: age ≥80 years, weight ≤60 kg, or serum creatinine ≥1.5 mg/dL",
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
- **Rationale:** The answer correctly identifies the dose of apixaban as 2.5 mg BID based on the criteria provided. It cites the appropriate chunk for dosing and includes a relevant interaction caveat. All clinically substantive fields are supported by the evidence.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-apixaban-2018#ddi, lexi-apixaban-2018#dosing-nvaf


---

## Question Q005 - Score: 80/100

### ❓ Question
> 85 kg adult with acute DVT: What is the enoxaparin treatment dose (normal renal function)?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 85,
  "volume_per_dose_mL": null,
  "frequency": "q12h",
  "max_daily_mg": null,
  "rounding": "Rounded to nearest whole number",
  "calc_rule": "1 mg/kg q12h (primary regimen) or 1.5 mg/kg q24h (alternative), adjusted for renal function (CrCl <30: 1 mg/kg q24h)",
  "citations": [
    "lexi-enoxaparin-2018#dosing"
  ],
  "notes": "Renal adjustment for CrCl <30 may be necessary"
}
```

### ✅ Expected Answer
```
Enoxaparin 1 mg/kg SC q12h → 85 mg per dose (or 1.5 mg/kg once daily as alternative). If CrCl <30 mL/min: 1 mg/kg once daily.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the dose per dose (85 mg) and frequency (q12h), but the max_daily_mg is not provided, which is critical. The renal adjustment is mentioned but not fully aligned with the gold reference. Citations are appropriate, but the rounding rule is not supported by evidence.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-enoxaparin-2018#dosing, lexi-enoxaparin-2018#renal


---

## Question Q006 - Score: 85/100

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
  "notes": "Metformin is contraindicated in eGFR <30 mL/min/1.73 m², and should be avoided in initiation for eGFR 30-45. If continued, dose reduction and monitoring of renal function is necessary."
```

### ✅ Expected Answer
```
No—contraindicated at eGFR <30. If 30–45, generally avoid initiation; if continuing, reduce dose and monitor. Avoid in hepatic impairment.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the contraindication of metformin at eGFR <30 and provides appropriate guidance for eGFR 30-45. It cites an allowed chunk and covers all key points.

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
  "calc_rule": "Avoid use if eGFR/CrCl <30 mL/min due to inadequate urinary concentrations and increased toxicity risk (lexi-nitrofurantoin-2018#renal)",
  "citations": [
    "lexi-nitrofurantoin-2018#renal"
  ],
  "notes": "Avoid nitrofurantoin in eGFR/CrCl <30 mL/min due to efficacy and toxicity concerns. Common contraindications include significant renal dysfunction (lexi-nitrofurantoin-2018#contra)."
}
```

### ✅ Expected Answer
```
Avoid nitrofurantoin when eGFR/CrCl <30 mL/min due to low urinary levels and toxicity risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly advises against nitrofurantoin use with eGFR <30 mL/min, cites relevant chunks, and covers contraindications.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-nitrofurantoin-2018#contra, lexi-nitrofurantoin-2018#renal


---

## Question Q008 - Score: 70/100

### ❓ Question
> Patient on warfarin needs TMP-SMX for UTI. Any interaction and what should we do?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "q12h",
  "max_daily_mg": null,
  "rounding": "round to nearest 5mg",
  "calc_rule": "Avoid TMP-SMX due to major interaction (CYP2C9 inhibition and gut flora changes), leading to increased INR. If used, reduce warfarin dose and increase INR monitoring.",
  "citations": [
    "lexi-warfarin-2018#ddi-tmp-smx"
  ],
  "notes": "Avoid TMP-SMX with warfarin; close INR monitoring with dose reduction if used."
}
```

### ✅ Expected Answer
```
Major interaction (CYP2C9 inhibition + gut flora). Avoid if possible; otherwise increase INR monitoring and reduce warfarin dose.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and appropriate actions, but it does not provide a specific dose or frequency for warfarin, which is critical for completeness. The frequency mentioned (q12h) is irrelevant to the interaction context. The citation supports the interaction but does not cover the missing dose information.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-warfarin-2018#ddi-tmp-smx


---

## Question Q009 - Score: 85/100

### ❓ Question
> Patient on atorvastatin needs clarithromycin. What’s the risk and action?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": "Not applicable",
  "calc_rule": "Strong CYP3A4 inhibition increases atorvastatin levels and myopathy/rhabdomyolysis risk; avoid; if unavoidable, hold or use the lowest dose with close monitoring",
  "citations": [
    "lexi-atorvastatin-2018#ddi-macrolides"
  ],
  "notes": "Avoid coadministration of clarithromycin with atorvastatin due to increased risk of myopathy/rhabdomyolysis; if unavoidable, hold or use the lowest dose with close monitoring"
}
```

### ✅ Expected Answer
```
Major interaction (strong CYP3A4 inhibitor). Avoid coadministration; if unavoidable, hold atorvastatin or limit to the lowest dose and monitor for myopathy.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly identifies the major interaction and appropriate action, supported by the allowed citation. All key points are covered.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-atorvastatin-2018#ddi-macrolides


---

## Question Q010 - Score: 80/100

### ❓ Question
> Starting linezolid in a patient on sertraline. Is this safe?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": null,
  "max_daily_mg": null,
  "rounding": "Not applicable",
  "calc_rule": "Avoid combination due to Linezolid's reversible MAOI activity precipitating serotonin syndrome with SSRIs",
  "citations": [
    "lexi-linezolid-2018#ddi-ssri"
  ],
  "notes": "Avoid Linezolid in patients on Sertraline due to risk of serotonin syndrome"
}
```

### ✅ Expected Answer
```
Major interaction: risk of serotonin syndrome (linezolid has MAOI activity). Avoid; if benefits outweigh risks, withhold SSRI or monitor very closely.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and cites the appropriate chunk, but it lacks explicit mention of the action to avoid the combination or the alternative monitoring strategy in the notes, which is critical for safety.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-linezolid-2018#ddi-ssri


---

## Question Q011 - Score: 70/100

### ❓ Question
> Patient on digoxin will start amiodarone. What’s the interaction management?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "qd",
  "max_daily_mg": null,
  "rounding": "Not applicable",
  "calc_rule": "P-gp inhibition ↑ digoxin dose by ~30-50%, monitor serum levels, renal function, and ECG (3 criteria met)",
  "citations": ["lexi-digoxin-2018#ddi-amio"],
  "notes": "Reduce digoxin dose by 30-50% when starting amiodarone, monitor digoxin levels, renal function, and ECG"
```

### ✅ Expected Answer
```
Major interaction (P-gp inhibition). Reduce digoxin dose by ~30–50% and monitor levels/ECG.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and the need to reduce the digoxin dose by 30-50%, supported by the allowed citation. However, it includes monitoring renal function, which is not specified in the gold reference, leading to a deduction for evidence coverage. The frequency and other fields are not applicable in this context.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-digoxin-2018#ddi-amio


---

## Question Q012 - Score: 0/100

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
  "calc_rule": "Recency-sensitive; abstain and refer to current guidance/pharmacist",
  "citations": ["kb-scope-2018#covid-absent"],
  "notes": "COVID-19 therapies (e.g., nirmatrelvir/ritonavir) are not covered in this context; refer to current guidance."
```

### ✅ Expected Answer
```
ABSTAIN. Not in ≤2018 corpus; recency-sensitive therapy. Advise consulting current guidance/pharmacist.
```

### ⚖️ Judge's Assessment
- **Score:** 0/100
- **Verdict:** `fail`
- **Rationale:** Model did not abstain as required; provided dosing information and frequency despite the need for abstention.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** 


---

## Question Q013 - Score: 70/100

### ❓ Question
> What is the current first-line pharmacologic therapy per latest hypertension guidelines?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "q12h",
  "max_daily_mg": null,
  "rounding": "Rounded to nearest whole number",
  "calc_rule": "First-line pharmacologic therapy per JNC8 (2014) guidelines not recommended as current; recency_risk = true",
  "citations": [
    "lexi-hypertension-2014#jnc8"
  ],
  "notes": "First-line pharmacologic therapy per JNC8 (2014) is not the current recommendation due to recency risk."
}
```

### ✅ Expected Answer
```
ABSTAIN if asked for ‘current’ guidance (2018 corpus includes JNC8/2014). Flag recency risk.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly abstains from providing current guidelines and acknowledges the recency risk, but it does not explicitly state 'ABSTAIN' as required.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-hypertension-2014#jnc8


---

