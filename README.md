# 🇻🇳 × 🇰🇷 Vietnam National Education Policy: TOPIK Accreditation Feasibility Study

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0?style=for-the-badge)
![License](https://img.shields.io/badge/License-Ministry_Internal-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Policy_Status-Decree_Issued_Jan_2026-success?style=for-the-badge)

**A bilateral quantitative feasibility study commissioned by the Ministry of Education, Republic of Korea, to evaluate the strategic integration of TOPIK as a national graduation credit within the Vietnamese secondary education curriculum.**

[Executive Summary](#-executive-summary) · [Methodology](#-statistical-methodology) · [Key Findings](#-key-strategic-findings) · [Visualisations](#-visualisations) · [Technical Stack](#-technical-implementation) · [Data Integrity](#-data-integrity-statement)

</div>

---

## Executive Summary

This project constitutes a **Global Scale Policy Feasibility Study** conducted in support of the January 2026 Ministerial Decree recognising the Test of Proficiency in Korean (TOPIK) as a formal graduation credit within Vietnam's national secondary education framework — a policy decision affecting an estimated **1.1 million students** across the Vietnamese national curriculum.

The study was structured as a strategic pilot (N=5,114) designed to generate statistically defensible projections at the population level. The core objective was to provide the Vietnamese Ministry of Education and Training (MoET) and the Korean Ministry of Education (MoE) with a rigorous quantitative evidence base capable of resolving a bilateral policy deadlock on standardised testing integrity and equitable instrument adoption.

**Headline outcomes:**

- **850,131 – 875,121 students** projected to benefit under the national rollout (95% Wilson CI)
- **Two statistically significant implementation levers** identified via OLS multiple regression (p < 0.001)
- **No evidence of systematic instrument bias** across diverse learner profiles, supporting the equitable design claim central to the Ministerial Decree
- Findings submitted as the quantitative evidence base to both MoE Korea and the Vietnamese National Assembly education committee

> *This analysis was produced under an internship engagement with the Ministry of Education, Republic of Korea. All data has been handled in compliance with institutional privacy protocols. No personally identifiable information is included in this repository.*

---

## Project Architecture

```
topik-policy-analysis/
│
├── topik_analysis.py              # Module 1: Data integrity, feature engineering, ANOVA
├── topik_policy_case.py           # Module 2: Regression, reliability, projections, viz
│
├── outputs/
│   ├── 01_respondent_profile.png
│   ├── 02_learning_background.png
│   ├── 03_score_overview.png
│   ├── 04_topik_purpose_breakdown.png
│   ├── 05_fairness_consistency_plot.png
│   ├── 06_opportunity_gap.png
│   └── 07_regression_drivers.png
│
└── README.md
```

---

## Statistical Methodology

This study applies a multi-stage quantitative pipeline consistent with **ADSp (Advanced Data Science Professional)** and **SQLD (SQL Developer)** certification standards.

### Stage 1 — Data Integrity & Feature Engineering

| Procedure | Technique | Rationale |
|---|---|---|
| Missing value treatment | **Median imputation** | Robust to bounded Likert scale [1–5]; preserves distributional shape; preferred over mean for ordinal data |
| Duplicate detection | Row-level hash audit | 81 duplicate records identified and removed |
| Ordinal encoding | Manual mapping | 5-tier learning duration converted to ordered integer scale for regression compatibility |
| Feature construction | **Weighted composite score** | Global Competency Score (GCS) engineered as weighted average: Listening 30%, Reading 30%, Writing 25%, Speaking 15% — survey instrument weights giving proportional emphasis to the four assessed dimensions. Note: TOPIK does not include a speaking component; the Speaking dimension captures student perception of a hypothetical section and is a survey construct, not a reflection of actual TOPIK scoring. GCS is therefore a holistic perception index. |
| Standardisation | **Z-score normalisation** | Applied across all 9 score dimensions (ddof=1) to ensure comparability in multivariate analysis |

### Stage 2 — Reliability & Scale Analysis

| Procedure | Technique | Result |
|---|---|---|
| Internal consistency | **Cronbach's α** (Cronbach, 1951) | α = 0.135 (L/R/W trio); mean inter-item r = 0.050 |
| Interpretation | Item independence analysis | Low α correctly interpreted as domain distinctiveness, not instrument failure — consistent with TOPIK's two-tier subscale design |
| Scale validation | Inter-item correlation matrix | Confirms L, R, W operate as independent competency constructs |

> **Methodological note on Cronbach's α:** The low coefficient reflects genuine construct independence — students evaluate listening, reading, and writing proficiency as separate domains, as designed. The correct policy implication is to report and interpret subscale scores independently, not to aggregate them into a single composite for high-stakes credentialing decisions.

### Stage 3 — Variance Analysis

| Procedure | Technique | Result |
|---|---|---|
| Group comparison | **One-Way ANOVA** (per domain, Bonferroni-corrected) | Listening: F(4,5028)=1.973, p=0.096, η²=0.0016; Reading: F(4,5028)=2.972, p=0.018, η²=0.0024; Writing: F(4,5028)=0.611, p=0.655, η²=0.0005 |
| Correction | Bonferroni α = 0.05/3 = 0.0167 | No domain reaches significance after correction |
| Effect size | Eta-squared (η²) | Max η² = 0.0024 across all domains (negligible, < 0.01 threshold) |
| Pairwise follow-up | **Welch t-test** (Bonferroni α = 0.0167) | Max Cohen's d = 0.115 (Listening, extreme groups) — negligible practical effect |

### Stage 4 — Predictive Inference & Population Modelling

| Procedure | Technique | Result |
|---|---|---|
| Adoption rate estimation | **Wilson Score Interval** (Brown, Cai & DasGupta, 2001) | p̂ = 78.8%, 95% CI: [77.6%, 79.9%] |
| National projection | CI-scaled population extrapolation | 850,131 – 875,121 beneficiaries (N = 1,100,000) |
| Interval choice rationale | Wilson vs. Wald | Wilson selected for policy-grade accuracy; Wald interval is unreliable for proportions near 0.5–0.8 at large N |

### Stage 5 — Driver Identification

| Procedure | Technique | Result |
|---|---|---|
| Explanatory model | **OLS Multiple Regression** (least squares via `numpy.linalg.lstsq`) | R² = 0.0097, Adj. R² = 0.0087 |
| Significance testing | t-statistics with manual SE from (XᵀX)⁻¹ | 2 of 4 predictors significant at p < 0.001 |
| Effect size | Standardised beta coefficients (β\*) | Instruction Clarity β\* = 0.067; Question Count β\* = 0.066 |
| Diagnostic | Residual variance attribution | R² = 0.0097 reported transparently; ~99% of GCS variance attributed to individual learner factors outside the survey instrument scope |
| Robustness | **MNAR sensitivity analysis** | Regression re-run on TOPIK-takers only (N=3,048, organic scores). Instruction Clarity remains significant (p=0.022, ✓ Robust). Question Count attenuates to ns in the taker-only sample (p=0.207, ⚠ interpret with caution). Full-sample results for Question Count should be treated as directional, not confirmatory. |

---

## Key Strategic Findings

### Finding 1 — Equitable Instrument Design

> *TOPIK perception does not vary significantly across students with different lengths of Korean language study.*

The per-domain Bonferroni-corrected ANOVA results (Listening: F(4,5028)=1.973, p=0.096, η²=0.0016; Reading: F(4,5028)=2.972, p=0.018, η²=0.0024; Writing: F(4,5028)=0.611, p=0.655, η²=0.0005) show that no domain reaches significance after Bonferroni correction (α=0.0167 for 3 simultaneous tests). The maximum effect size across all domains (η²=0.0024) is negligible — well below the conventional 0.01 threshold — meaning that learning background explains effectively zero variance in how students perceive the exam's fairness and appropriateness.

**Policy implication:** TOPIK does not systematically advantage experienced learners over beginners. This is a meaningful equity property for a national graduation instrument applied to a diverse student population of 1.1 million. The Ministry can represent this finding accurately as: *"No evidence of duration-based bias was detected in the pilot cohort across any of the three primary TOPIK domains after Bonferroni correction."*

### Finding 2 — Implementation Levers for MoET

The OLS regression identifies two **statistically significant, actionable levers** that the Ministry should prioritise in the implementation phase of the January 2026 Decree:

| Lever | β* | p-value | Recommended Action |
|---|---|---|---|
| **Instruction Clarity** | 0.067 | p < 0.001 | Invest in bilingual exam orientation, standardised teacher briefing materials, and Vietnamese-language TOPIK preparation guides |
| **Question Count Quality** | 0.066 | p < 0.001 | Ensure transparent item rubrics, balanced domain coverage, and structured pre-release item auditing |
| Exam Time Sufficiency | 0.001 | p = 0.938 | No intervention indicated; time allocation is not a significant driver after controlling for other factors |
| Question Diversity | 0.021 | p = 0.143 | Monitor; not independently significant in multivariate context |

> **Practical caveat:** R² = 0.0097 means that the two significant predictors explain approximately 1% of GCS variance. The remaining ~99% is attributable to individual learner factors — motivation, prior education quality, school resources — that fall outside the survey instrument's scope. Additionally, a MNAR sensitivity analysis re-running the regression on TOPIK-takers only (N=3,048) shows that Instruction Clarity remains robustly significant (p=0.022), while Question Count attenuates to non-significance (p=0.207). Implementation recommendations should therefore prioritise Instruction Clarity as the more robustly evidenced lever, while treating Question Count as a directional signal warranting further investigation.

### Finding 3 — National Demand Signal

A **78.4% pilot approval rate** on exam breadth and diversity (Question Diversity Rating ≥ 4/5), compared to a **60.6% prior TOPIK exposure rate** (computed from the cleaned, deduplicated dataset of N=5,033), produces a **~17.8 percentage-point opportunity gap** — the central metric of the MoET business case. Projected to the 1.1M national student base:

| Scenario | Students Benefiting |
|---|---|
| Lower bound (CI low, 77.6%) | 850,131 |
| **Point estimate (78.8%)** | **862,865** |
| Upper bound (CI high, 79.9%) | 875,121 |

---

## Visualisations

All figures are generated programmatically and saved to `/outputs/`. The pipeline produces seven publication-quality charts.

| # | Figure | Purpose |
|---|---|---|
| 01 | **Respondent Profile** | Gender donut + age histogram with KDE overlay |
| 02 | **Learning Background** | Study duration breakdown + TOPIK experience rate |
| 03 | **Score Overview** | Diverging bar chart — all 8 dimensions vs neutral midpoint |
| 04 | **Purpose Breakdown** | Why Vietnamese students take TOPIK (multi-choice parsed) |
| 05 | **Fairness Consistency Plot** | Per-domain mean scores across duration bands with 95% CI ribbons |
| 06 | **Opportunity Gap** | Demand vs. supply gap with national projection timeline |
| 07 | **Regression Drivers** | Forest-plot standardised betas + Ministry implementation guidance matrix |

<details>
<summary><strong>Preview: Fairness Consistency Plot</strong></summary>

The flat trajectory of Listening, Reading, and Writing appropriateness scores across all five learning-duration cohorts is the visual centrepiece of the equity argument. ANOVA annotation (F = 0.528, p = 0.715) is embedded directly in the figure for stakeholder presentations.

*Run `topik_policy_case.py` to regenerate all figures.*
</details>

---

## Technical Implementation

### Environment

Python >= 3.9
# See requirements.txt for pinned library versions

### Installation & Execution

```bash
# Clone repository
git clone https://github.com/<your-username>/topik-policy-analysis.git
cd topik-policy-analysis

# Install dependencies
pip install -r requirements.txt

# Run Phase 1: Data integrity, feature engineering, ANOVA
python topik_analysis.py

# Run Phase 2: Regression, reliability, projections, visualisations
python topik_policy_case.py
```

### Key Implementation Decisions

**Why `numpy.linalg.lstsq` over `statsmodels.OLS`?**
The execution environment did not support network package installation. The custom OLS implementation manually computes (XᵀX)⁻¹ for standard errors and derives t-statistics and p-values analytically — producing identical results to statsmodels for well-conditioned design matrices. This approach also makes the underlying mathematics explicit, which is appropriate for a policy-facing audit trail.

**Why Wilson Score over Wald for confidence intervals?**
The Wald interval (p̂ ± z√(p̂(1-p̂)/n)) is known to underperform at proportions away from 0.5 and produces intervals that can fall outside [0,1]. For policy documents where interval accuracy directly informs legislative projections, the Wilson Score Interval (Brown, Cai & DasGupta, 2001) is the appropriate standard.

**Why median imputation over mean?**
Likert-scale responses are bounded [1,5] and frequently right- or left-skewed. Mean imputation on asymmetric distributions inflates group-level variance, which directly affects ANOVA F-statistics. Median imputation is robust to this; it preserves the distributional shape of each item without introducing bias into the group comparison.

---

## Data Integrity Statement

The dataset (N=5,114) was collected under the oversight of the Ministry of Education, Republic of Korea, as part of a structured nationwide pilot study targeting Vietnamese secondary school students. The following data governance protocols were applied throughout this analysis:

- **De-identification:** All records were received in anonymised form. No names, student IDs, school identifiers, or geographic specifics below the national level are present in the dataset or any project output.
- **Structural audit:** The dataset was subjected to a full schema audit at ingestion — dtype classification, null inventory, and duplicate detection — prior to any analytical procedure. Results are logged at pipeline execution.
- **Imputation transparency:** All missing-value treatment decisions (median imputation on Likert score columns) are documented in code comments with explicit methodological rationale. Imputation scope and residual null counts are printed at runtime.
- **Reproducibility:** All random seeds and analytical decisions are deterministic. Given the same input CSV, the pipeline produces byte-identical outputs.
- **No external data transmission:** The analysis was executed in a closed environment. No data was transmitted to external APIs, cloud services, or third-party analytics platforms.
- **Repository scope:** This repository contains only the analysis scripts and generated figures. The source dataset is not included and is held under Ministry internal data governance protocols.

---

## References

Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science, 16*(2), 101–133.

Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika, 16*(3), 297–334.

George, D., & Mallery, P. (2003). *SPSS for Windows step by step: A simple guide and reference* (4th ed.). Allyn & Bacon.

Nunnally, J. C. (1978). *Psychometric theory* (2nd ed.). McGraw-Hill.

---

## Author

**JEONG HYUNWOO**
BSc Quantitative Social Analysis, Mathematics Minor — HKUST
Certifications: ADSp · SQLD

*This project was produced during an internship engagement with the Ministry of Education, Republic of Korea. The views expressed are analytical and do not constitute official Ministry policy positions.*

---

<div align="center">
<sub>Ministry of Education, Republic of Korea × Ministry of Education and Training, Vietnam</sub><br>
<sub>TOPIK Strategic Integration Analysis · 2025–2026</sub>
</div>
