# 🏥 Pharmacy Copilot Evaluation Report

**Generated:** 2025-10-02 04:22:54

## 📊 Executive Summary

- **Total Questions:** 13
- **Mean Score:** 85.4/100
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
  "volume_per_dose_mL": 10,
  "frequency": "q12h",
  "max_daily_mg": 4000,
  "rounding": "Rounded from 10.1 mL to 10 mL for practical measurement",
  "calc_rule": "90 mg/kg/day \u00f7 2 doses = 45 mg/kg/dose; 18 kg \u00d7 45 mg/kg = 810 mg/dose; volume = 810 mg \u00f7 (80 mg/mL)",
  "citations": [
    "lexi-amoxicillin-2018#dosage",
    "lexi-amoxicillin-2018#peds"
  ],
  "notes": "High-dose amoxicillin regimen for AOM using 400 mg/5 mL suspension with practical volume rounding per guidance."
}
```

### ✅ Expected Answer
```
Amoxicillin 90 mg/kg/day divided q12h → 810 mg per dose (≈10.1 mL of 400 mg/5 mL), round to 10 mL per dose; max 4 g/day.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the amoxicillin dose, volume, frequency, and acknowledges the maximum daily dose. It cites allowed chunk IDs and adheres to safety guidelines.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-amoxicillin-2018#dosage, lexi-amoxicillin-2018#peds


---

## Question Q002 - Score: 95/100

### ❓ Question
> 12 kg child with fever: acetaminophen dose and volume using 160 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 180,
  "volume_per_dose_mL": 5.6,
  "frequency": "q6h",
  "max_daily_mg": 900,
  "rounding": "Rounded volume from 5.625 mL to 5.6 mL",
  "calc_rule": "15 mg/kg/dose \u00d7 12 kg = 180 mg; max 75 mg/kg/day = 900 mg",
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
- **Score:** 95/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly calculates the acetaminophen dose and volume, cites appropriate evidence chunks, and adheres to safety guidelines.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-acetaminophen-2018#max, lexi-acetaminophen-2018#peds-dosing


---

## Question Q003 - Score: 90/100

### ❓ Question
> 12 kg child with fever: ibuprofen dose and volume using 100 mg/5 mL?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 120,
  "volume_per_dose_mL": 6.0,
  "frequency": "q6-8h",
  "max_daily_mg": 480,
  "rounding": "Rounded to nearest 0.5 mL",
  "calc_rule": "10 mg/kg/dose \u00d7 12 kg = 120 mg; volume = 120 mg \u00f7 (100 mg/5 mL) = 6 mL; max 40 mg/kg/day = 480 mg",
  "citations": [
    "lexi-ibuprofen-2018#peds-dosing",
    "lexi-ibuprofen-2018#max"
  ],
  "notes": "Weight-based dosing using 100 mg/5 mL suspension with caution in dehydration or renal risk."
}
```

### ✅ Expected Answer
```
Ibuprofen 10 mg/kg/dose q6–8h → 120 mg per dose (6 mL of 100 mg/5 mL); max 40 mg/kg/day (480 mg). Avoid if <6 months or dehydrated/renal issues.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the ibuprofen dose, volume, frequency, and maximum daily dose, and cites allowed chunk IDs without any disallowed ones.

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
  "rounding": "No rounding needed for tablet dose",
  "calc_rule": "Meets 2 of 3 criteria for dose reduction: age \u226580y (82y), weight \u226460kg (58kg), SCr \u22651.5mg/dL (1.6mg/dL)",
  "citations": [
    "lexi-apixaban-2018#dosing-nvaf"
  ],
  "notes": "Reduced dose of 2.5mg tablets twice daily indicated due to meeting multiple criteria for dose reduction in NVAF."
}
```

### ✅ Expected Answer
```
Apixaban 2.5 mg BID (meets 2 of 3: age ≥80, weight ≤60 kg, SCr ≥1.5). Avoid with strong CYP3A4/P-gp inhibitors.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the apixaban dose as 2.5 mg BID based on the criteria met, cites an allowed chunk ID, and does not exceed maximum dosing or violate contraindications.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-apixaban-2018#ddi, lexi-apixaban-2018#dosing-nvaf


---

## Question Q005 - Score: 90/100

### ❓ Question
> 85 kg adult with acute DVT: What is the enoxaparin treatment dose (normal renal function)?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": 85,
  "volume_per_dose_mL": 0.9,
  "frequency": "q12h",
  "max_daily_mg": 170,
  "rounding": "Round to nearest 0.1 mL based on 100 mg/mL concentration",
  "calc_rule": "1 mg/kg q12h (preferred regimen) x 85 kg = 85 mg per dose",
  "citations": [
    "lexi-enoxaparin-2018#dosing"
  ],
  "notes": "Preferred regimen is 1 mg/kg q12h for acute DVT with normal renal function, with 1.5 mg/kg daily as an alternative."
}
```

### ✅ Expected Answer
```
Enoxaparin 1 mg/kg SC q12h → 85 mg per dose (or 1.5 mg/kg once daily as alternative). If CrCl <30 mL/min: 1 mg/kg once daily.
```

### ⚖️ Judge's Assessment
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly provides the enoxaparin treatment dose of 85 mg per dose, cites the appropriate chunk ID, and includes the alternative dosing regimen. It adheres to safety guidelines without exceeding maximum doses.

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
  "frequency": "n/a",
  "max_daily_mg": null,
  "rounding": "n/a",
  "calc_rule": "eGFR <30 mL/min/1.73m\u00b2 = contraindicated",
  "citations": [
    "lexi-metformin-2018#renal"
  ],
  "notes": "Metformin is contraindicated in this patient with eGFR 28 mL/min/1.73m\u00b2 due to risk of lactic acidosis."
}
```

### ✅ Expected Answer
```
No—contraindicated at eGFR <30. If 30–45, generally avoid initiation; if continuing, reduce dose and monitor. Avoid in hepatic impairment.
```

### ⚖️ Judge's Assessment
- **Score:** 80/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies that metformin is contraindicated at eGFR <30, but it does not mention the recommendation for patients with eGFR 30–45. It cites one allowed chunk ID but misses another relevant one, which affects the completeness of the evidence support.

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
  "rounding": "Not applicable - contraindicated",
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
- **Score:** 90/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly states to avoid nitrofurantoin if eGFR <30 mL/min, cites relevant evidence chunks, and provides a rationale for the contraindication.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-nitrofurantoin-2018#contra, lexi-nitrofurantoin-2018#renal


---

## Question Q008 - Score: 85/100

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
  "calc_rule": "Drug interaction via CYP2C9 inhibition and gut flora changes",
  "citations": [
    "lexi-warfarin-2018#ddi-tmp-smx"
  ],
  "notes": "TMP-SMX markedly increases warfarin effect requiring warfarin dose reduction and increased INR monitoring, with preference for alternative antibiotics."
}
```

### ✅ Expected Answer
```
Major interaction (CYP2C9 inhibition + gut flora). Avoid if possible; otherwise increase INR monitoring and reduce warfarin dose.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The answer correctly identifies the major interaction mechanism and appropriate actions, and it cites an allowed chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-warfarin-2018#ddi-tmp-smx


---

## Question Q009 - Score: 75/100

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
  "calc_rule": "hold atorvastatin during clarithromycin due to CYP3A4 inhibition; if unavoidable use lowest dose with monitoring",
  "citations": [
    "lexi-atorvastatin-2018#ddi-macrolides"
  ],
  "notes": "Major interaction increases myopathy/rhabdomyolysis risk through strong CYP3A4 inhibition."
}
```

### ✅ Expected Answer
```
Major interaction (strong CYP3A4 inhibitor). Avoid coadministration; if unavoidable, hold atorvastatin or limit to the lowest dose and monitor for myopathy.
```

### ⚖️ Judge's Assessment
- **Score:** 75/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the major interaction and mechanism but does not explicitly state 'avoid coadministration' as required. It also provides appropriate action but lacks clarity in the phrasing of the action. The citation is valid.

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
  "frequency": "n/a",
  "max_daily_mg": null,
  "rounding": "n/a",
  "calc_rule": "Contraindicated due to MAOI activity unless benefits outweigh risks",
  "citations": [
    "lexi-linezolid-2018#ddi-ssri"
  ],
  "notes": "Avoid combination due to serotonin syndrome risk; if necessary, withhold SSRI and monitor closely."
}
```

### ✅ Expected Answer
```
Major interaction: risk of serotonin syndrome (linezolid has MAOI activity). Avoid; if benefits outweigh risks, withhold SSRI or monitor very closely.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly identifies the major interaction and provides appropriate actions, citing the allowed chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-linezolid-2018#ddi-ssri


---

## Question Q011 - Score: 75/100

### ❓ Question
> Patient on digoxin will start amiodarone. What’s the interaction management?

### 🤖 LLM Generated Answer
```json
{
  "dose_per_dose_mg": null,
  "volume_per_dose_mL": null,
  "frequency": "maintain current",
  "max_daily_mg": null,
  "rounding": "reduce current dose by 30-50%",
  "calc_rule": "reduce existing digoxin dose by 30-50% due to P-glycoprotein inhibition",
  "citations": [
    "lexi-digoxin-2018#ddi-amio"
  ],
  "notes": "Reduce digoxin dose by 30-50% when starting amiodarone and monitor digoxin levels, renal function, and ECG."
}
```

### ✅ Expected Answer
```
Major interaction (P-gp inhibition). Reduce digoxin dose by ~30–50% and monitor levels/ECG.
```

### ⚖️ Judge's Assessment
- **Score:** 75/100
- **Verdict:** `manual_review`
- **Rationale:** The answer correctly identifies the interaction and recommends a dose reduction of 30-50%, but it lacks clarity in the mechanism and severity. It cites the correct chunk ID, but the phrasing could be more aligned with the expected answer.

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
  "frequency": "",
  "max_daily_mg": null,
  "rounding": "Not applicable - insufficient data",
  "calc_rule": "Cannot calculate - predates COVID-19 therapies",
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
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** Correctly abstains and advises consulting current guidance; cites allowed chunk ID.

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
  "rounding": "Not applicable",
  "calc_rule": "No current dosing available",
  "citations": [
    "lexi-hypertension-2014#jnc8"
  ],
  "notes": "Cannot provide current first-line therapy as available guideline data from JNC8 (2014) is outdated."
}
```

### ✅ Expected Answer
```
ABSTAIN if asked for ‘current’ guidance (2018 corpus includes JNC8/2014). Flag recency risk.
```

### ⚖️ Judge's Assessment
- **Score:** 85/100
- **Verdict:** `pass`
- **Rationale:** The model correctly abstains from providing current guidelines, recognizes the outdated nature of JNC8 (2014), and cites the appropriate chunk ID.

### 📑 Retrieval Performance
- **Relevant Chunks Found:** lexi-hypertension-2014#jnc8


---

