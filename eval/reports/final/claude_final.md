# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** 2025-10-02 06:33:55

## 📊 Executive Summary

- **Total Questions:** 13
- **Mean Score:** 80.8/100
- **Median Score:** 85.0/100
- **Recall@5:** 92.3%

---

## Question Q001 - Score: 85/100

### ❓ Question
> 5-year-old (18 kg) with acute otitis media. What is the amoxicillin dose and volume using 400 mg/5 mL suspension?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 810,
  "volume_per_dose_mL": 10,
  "frequency": "q12h",
  "max_daily_mg": 4000,
  "rounding": "Round to nearest whole mL for volumes \u22655 mL",
  "calc_rule": "90 mg/kg/day \u00f7 2 doses = 45 mg/kg/dose \u00d7 18 kg = 810 mg/dose; volume = 810 mg \u00f7 (80 mg/mL)",
  "citations": [
    "lexi-amoxicillin-2018#dosage",
    "lexi-amoxicillin-2018#peds"
  ],
  "notes": "High-dose amoxicillin regimen for AOM using 400 mg/5 mL suspension with practical mL rounding."
}
```

### ✅ Expected Answer
```
Amoxicillin 90 mg/kg/day divided q12h → 810 mg per dose (≈10.1 mL of 400 mg/5 mL), round to 10 mL per dose; max 4 g/day.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly calculates the dose, volume, frequency, and maximum daily dose, and cites appropriate chunks. However, the rounding rule is slightly misphrased but still acceptable.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-amoxicillin-2018#dosage, lexi-amoxicillin-2018#peds


---

## Question Q002 - Score: 85/100

### ❓ Question
> 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 180,
  "volume_per_dose_mL": 5.6,
  "frequency": "q6h",
  "max_daily_mg": 900,
  "rounding": "Round volume to nearest 0.1 mL",
  "calc_rule": "15 mg/kg/dose \u00d7 12 kg = 180 mg; Max daily = 75 mg/kg/day \u00d7 12 kg = 900 mg",
  "citations": [
    "lexi-acetaminophen-2018#peds-dosing",
    "lexi-acetaminophen-2018#max"
  ],
  "notes": "Oral liquid acetaminophen dosed at 15 mg/kg/dose using 160 mg/5 mL concentration for fever management."
}
```

### ✅ Expected Answer
```
Acetaminophen 15 mg/kg/dose q6h (range 10–15 mg/kg q4–6h) → 180 mg per dose ≈5.6 mL of 160 mg/5 mL; max 75 mg/kg/day (900 mg), and ≤4 g absolute.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly calculates the dose and volume, cites appropriate chunks, and adheres to the maximum daily dose. All clinically substantive fields are supported.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-acetaminophen-2018#max, lexi-acetaminophen-2018#peds-dosing


---

## Question Q003 - Score: 80/100

### ❓ Question
> 12 kg child with fever: ibuprofen dose and volume using 100 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 120,
  "volume_per_dose_mL": 6.0,
  "frequency": "q6-8h",
  "max_daily_mg": 480,
  "rounding": "Round to nearest 0.5 mL",
  "calc_rule": "10 mg/kg/dose \u00d7 12 kg = 120 mg; max 40 mg/kg/day \u00d7 12 kg = 480 mg; volume = 120 mg \u00f7 (100 mg/5 mL)",
  "citations": [
    "lexi-ibuprofen-2018#peds-dosing",
    "lexi-ibuprofen-2018#max"
  ],
  "notes": "Weight-based ibuprofen suspension dosing with caution in dehydration or renal risk."
}
```

### ✅ Expected Answer
```
Ibuprofen 10 mg/kg/dose q6–8h → 120 mg per dose (6 mL of 100 mg/5 mL); max 40 mg/kg/day (480 mg). Avoid if <6 months or dehydrated/renal issues.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly provides the dose, volume, frequency, and maximum daily dose, but the rounding rule is not supported by the evidence. The notes mention caution in dehydration or renal risk, which is acceptable, but the rounding rule is not aligned with the gold reference.

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
  "frequency": "q12h",
  "max_daily_mg": 5,
  "rounding": "No rounding needed for fixed-dose tablet",
  "calc_rule": "Meets 2 of 3 criteria for dose reduction: weight \u226460kg (58kg) and SCr \u22651.5mg/dL (1.6mg/dL)",
  "citations": [
    "lexi-apixaban-2018#dosing-nvaf",
    "lexi-apixaban-2018#ddi"
  ],
  "notes": "Reduced dose of 2.5mg BID indicated due to meeting 2 of 3 risk criteria, with tablet formulation not requiring volume measurement."
}
```

### ✅ Expected Answer
```
Apixaban 2.5 mg BID (meets 2 of 3: age ≥80, weight ≤60 kg, SCr ≥1.5). Avoid with strong CYP3A4/P-gp inhibitors.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the dose of apixaban as 2.5 mg BID based on the criteria met, cites appropriate chunks for support, and includes necessary interaction warnings. All clinically substantive fields are supported.

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
  "volume_per_dose_mL": 0.85,
  "frequency": "q12h",
  "max_daily_mg": 170,
  "rounding": "Round to nearest 0.1 mL",
  "calc_rule": "1 mg/kg \u00d7 85 kg = 85 mg per dose",
  "citations": [
    "lexi-enoxaparin-2018#dosing"
  ],
  "notes": "Preferred regimen is 1 mg/kg q12h for acute DVT treatment with normal renal function, with each 100 mg/mL dose given subcutaneously."
}
```

### ✅ Expected Answer
```
Enoxaparin 1 mg/kg SC q12h → 85 mg per dose (or 1.5 mg/kg once daily as alternative). If CrCl <30 mL/min: 1 mg/kg once daily.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the enoxaparin dose of 85 mg per dose, frequency of q12h, and cites an allowed chunk. It also includes the renal adjustment information, fulfilling the requirements.

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
  "frequency": "N/A",
  "max_daily_mg": null,
  "rounding": "N/A",
  "calc_rule": "Contraindicated if eGFR <30 mL/min/1.73m\u00b2; patient eGFR 28 is below threshold",
  "citations": [
    "lexi-metformin-2018#renal"
  ],
  "notes": "Metformin is contraindicated at this eGFR of 28 mL/min/1.73m\u00b2 due to safety concerns"
}
```

### ✅ Expected Answer
```
No—contraindicated at eGFR <30. If 30–45, generally avoid initiation; if continuing, reduce dose and monitor. Avoid in hepatic impairment.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies metformin as contraindicated at eGFR <30 and cites one allowed chunk. However, it lacks coverage for the recommendation to avoid initiation at eGFR 30–45 and does not mention the need to reduce dose and monitor if continuing, which is critical information.

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
  "frequency": "n/a",
  "max_daily_mg": null,
  "rounding": "n/a",
  "calc_rule": "Contraindicated if eGFR <30 mL/min",
  "citations": [
    "lexi-nitrofurantoin-2018#renal",
    "lexi-nitrofurantoin-2018#contra"
  ],
  "notes": "Nitrofurantoin is contraindicated at eGFR 25 mL/min due to inadequate urinary concentrations and increased toxicity risk."
}
```

### ✅ Expected Answer
```
Avoid nitrofurantoin when eGFR/CrCl <30 mL/min due to low urinary levels and toxicity risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly advises against nitrofurantoin use at eGFR 25 mL/min, citing appropriate evidence and rationale.

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
  "frequency": "n/a",
  "max_daily_mg": null,
  "rounding": "n/a",
  "calc_rule": "DDI management rule: TMP-SMX increases warfarin effect via dual mechanisms",
  "citations": [
    "lexi-warfarin-2018#ddi-tmp-smx"
  ],
  "notes": "TMP-SMX markedly increases warfarin effect via CYP2C9 inhibition and gut flora changes - avoid if possible or reduce warfarin dose with increased INR monitoring."
}
```

### ✅ Expected Answer
```
Major interaction (CYP2C9 inhibition + gut flora). Avoid if possible; otherwise increase INR monitoring and reduce warfarin dose.
```

### ⚖️ Judge's Assessment
- **Score:** 75/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and mechanism but does not explicitly state the severity or action in the expected format. It cites the correct chunk but lacks clarity in the action recommendation.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-warfarin-2018#ddi-tmp-smx


---

## Question Q009 - Score: 70/100

### ❓ Question
> Patient on atorvastatin needs clarithromycin. What’s the risk and action?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "hold",
  "max_daily_mg": null,
  "rounding": "not applicable",
  "calc_rule": "hold atorvastatin during clarithromycin due to strong CYP3A4 inhibition",
  "citations": [
    "lexi-atorvastatin-2018#ddi-macrolides"
  ],
  "notes": "Major interaction requiring atorvastatin hold or lowest dose with monitoring due to increased myopathy risk from CYP3A4 inhibition."
}
```

### ✅ Expected Answer
```
Major interaction (strong CYP3A4 inhibitor). Avoid coadministration; if unavoidable, hold atorvastatin or limit to the lowest dose and monitor for myopathy.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and cites the appropriate chunk, but the action is not clearly stated as 'avoid coadministration' and the frequency is inaccurately noted as 'hold' instead of specifying avoidance.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-atorvastatin-2018#ddi-macrolides


---

## Question Q010 - Score: 85/100

### ❓ Question
> Starting linezolid in a patient on sertraline. Is this safe?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "N/A",
  "max_daily_mg": null,
  "rounding": "N/A",
  "calc_rule": "DDI assessment: linezolid MAOI activity + SSRI = contraindicated combination",
  "citations": [
    "lexi-linezolid-2018#ddi-ssri"
  ],
  "notes": "Avoid concurrent use due to serotonin syndrome risk; if benefits outweigh risks, discontinue sertraline before starting linezolid and monitor closely."
}
```

### ✅ Expected Answer
```
Major interaction: risk of serotonin syndrome (linezolid has MAOI activity). Avoid; if benefits outweigh risks, withhold SSRI or monitor very closely.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly identifies the major interaction and provides appropriate actions. It cites an allowed chunk and covers the key points, although it lacks specific numeric values for dose and frequency.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-linezolid-2018#ddi-ssri


---

## Question Q011 - Score: 80/100

### ❓ Question
> Patient on digoxin will start amiodarone. What’s the interaction management?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "unchanged",
  "max_daily_mg": null,
  "rounding": "Reduce existing digoxin dose by 30-50%",
  "calc_rule": "Reduce existing digoxin dose by 30-50% due to P-glycoprotein inhibition by amiodarone",
  "citations": [
    "lexi-digoxin-2018#ddi-amio"
  ],
  "notes": "Due to P-glycoprotein inhibition increasing digoxin exposure, reduce digoxin dose by 30-50% and monitor serum levels, renal function, and ECG when adding amiodarone."
}
```

### ✅ Expected Answer
```
Major interaction (P-gp inhibition). Reduce digoxin dose by ~30–50% and monitor levels/ECG.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly identifies the major interaction and the need to reduce the digoxin dose by 30-50%, citing the appropriate chunk. However, it does not explicitly mention the mechanism of action (P-gp inhibition) in the main response, which is a critical aspect of the interaction management.

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
  "rounding": "not applicable",
  "calc_rule": "insufficient context - predates COVID therapies",
  "citations": [
    "kb-scope-2018#covid-absent"
  ],
  "notes": "Knowledge base predates COVID-19 therapies including Paxlovid (nirmatrelvir/ritonavir); refer to current clinical guidance and pharmacist."
}
```

### ✅ Expected Answer
```
ABSTAIN. Not in ≤2018 corpus; recency-sensitive therapy. Advise consulting current guidance/pharmacist.
```

### ⚖️ Judge's Assessment
- **Score:** 70/100
- **Verdict:** `manual_review`
- **Rationale:** The model correctly abstains and advises consulting current guidance, but the evidence coverage for the explicit abstention is not fully supported by the allowed chunk.

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
  "rounding": "not applicable",
  "calc_rule": "cannot determine from outdated 2014 guidelines",
  "citations": [
    "lexi-hypertension-2014#jnc8"
  ],
  "notes": "Current first-line therapy cannot be determined as available context is from 2014 JNC8 guidelines which are outdated for current recommendations"
}
```

### ✅ Expected Answer
```
ABSTAIN if asked for ‘current’ guidance (2018 corpus includes JNC8/2014). Flag recency risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly abstains from providing current guidelines, cites the outdated JNC8 (2014) reference, and sets recency risk. All key points are covered.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-hypertension-2014#jnc8


---

