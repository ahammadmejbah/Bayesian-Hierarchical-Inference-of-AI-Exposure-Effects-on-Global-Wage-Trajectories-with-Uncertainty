# Bayesian Hierarchical Inference of AI Exposure Effects on Global Wage Trajectories

An uncertainty-aware research workspace for examining salary patterns across occupations, countries, experience levels, and years. The repository includes a Streamlit observatory, a classical machine-learning benchmark, and an IEEE-inspired HTML research paper.

> **Evidence status:** This release is exploratory. The repository contains salary observations but does not yet contain the external occupational AI-exposure index required for causal or Bayesian exposure-effect estimates. The dashboard therefore reports descriptive relationships and clearly labels its title-based proxy as non-causal.

## Contents

- [Research objective](#research-objective)
- [What is implemented](#what-is-implemented)
- [Dataset](#dataset)
- [Machine-learning benchmark](#machine-learning-benchmark)
- [Run locally](#run-locally)
- [Project structure](#project-structure)
- [Methodology roadmap](#methodology-roadmap)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Reproducibility](#reproducibility)
- [Data availability](#data-availability)
- [Citation](#citation)

## Research objective

The planned study asks whether occupational exposure to generative AI is associated with heterogeneous wage trajectories after accounting for:

1. Salary reporting and measurement uncertainty.
2. Country, sector, role, experience, and year structure.
3. Differences between augmentation-oriented and substitution-oriented occupations.
4. Predictive uncertainty across competing model specifications.

The intended final model is a Bayesian hierarchical wage model with partial pooling, role-specific observation scales, country-level exposure slopes, temporal effects, posterior predictive checks, and model averaging. The current application is the data-audit and predictive-benchmark stage of that workflow.

## What is implemented

### Streamlit observatory

Run the app and use the workspace selector to access:

- **Home and documentation:** project purpose, field guide, proxy definition, uncertainty notes, and workflow instructions.
- **Research paper:** an IEEE-inspired double-column research reading view with the full paper outline, figures, tables, methodology, limitations, and citations.
- **Observatory:** interactive salary analysis with filters for year, experience level, and work mode.
- **Hierarchical summaries:** group-level log-salary summaries with normal-approximation intervals and observation thresholds.
- **Compression screen:** descriptive salary trajectories by title-based screening band and country-level uncertainty width.
- **ML benchmark:** ten classical regression models evaluated on a forward 2025 holdout.
- **Data quality:** missingness diagnostics and filtered-observation CSV export.

### Standalone paper

[`paper.html`](paper.html) is a browser-readable paper artifact styled after an IEEE conference manuscript:

- US Letter page geometry.
- Times New Roman body typography.
- CSS two-column flow.
- IEEE-style headings and numbered references.
- Equations, figures, data tables, captions, and data-availability notes.
- Kaggle dataset and GitHub repository citations.

For formal submission, convert the manuscript into the target conference's official IEEE LaTeX or Word template and validate the resulting PDF with the conference's required process.

## Dataset

The application reads [`Jobs.csv`](Jobs.csv), a salary snapshot file with the following verified coverage as of September 9, 2026:

| Statistic | Value |
| --- | ---: |
| Salary records | 71,913 |
| Years | 2020–2025 |
| Employee-residence groups | 104 |
| Role-family values | 11 |
| Overall median salary | $138,750 |

Important fields include:

| Field | Description |
| --- | --- |
| `work_year` | Snapshot year. |
| `experience_level` / `experience_level_label` | Coded and readable seniority level. |
| `employment_type` / `employment_type_label` | Coded and readable employment type. |
| `job_title` | Reported job title. |
| `salary`, `salary_currency` | Original salary and currency metadata. |
| `salary_in_usd` | USD-converted salary used by the application. |
| `employee_residence` | Employee residence country code. |
| `remote_ratio` / `work_mode` | Remote-work percentage and readable work-mode category. |
| `company_location` | Company location country code. |
| `company_size` | Company-size category. |
| `salary_outlier_flag` | Source-level outlier indicator. |
| `role_family` | Normalized role grouping. |
| `isco_group_hint` | Available occupation classification hint. |

The current file has 40,540 missing `isco_group_hint` values. This is a major integration constraint for the planned exposure-index join; missing mappings must be documented and handled before formal inference.

## Machine-learning benchmark

The ML benchmark predicts reported salary, not the causal effect of AI exposure. It uses these features:

- Experience level.
- Employment type.
- Role family.
- Employee residence.
- Company location.
- Company size.
- Work mode.
- Work year.
- Remote ratio.

Models train on 2020–2024 and are evaluated on the held-out 2025 records. The target is modeled on the log scale and predictions are returned to USD for reporting. Lower MAE, RMSE, and MAPE are better; higher R² is better.

| Model | MAE (USD) | RMSE (USD) | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Ridge Regression | $49,524 | $69,772 | 0.210 | 38.22% |
| Linear Regression | $49,515 | $69,811 | 0.209 | 38.13% |
| Elastic Net | $49,761 | $69,813 | 0.209 | 38.92% |
| Lasso Regression | $49,903 | $69,902 | 0.207 | 39.26% |
| Random Forest | $49,288 | $70,258 | 0.199 | 36.91% |
| Hist. Gradient Boosting | $49,214 | $70,289 | 0.198 | 36.70% |
| Extra Trees | $49,403 | $70,354 | 0.197 | 37.14% |
| Gradient Boosting | $49,411 | $70,495 | 0.194 | 37.12% |
| K-Nearest Neighbors | $51,528 | $73,813 | 0.116 | 37.26% |
| AdaBoost | $59,857 | $84,312 | -0.154 | 43.38% |

These results are a single temporal benchmark, not a complete model-selection study. They should be rerun whenever the input data, feature mapping, or split protocol changes.

## Run locally

### Requirements

- Python 3.10 or newer.
- `pip` and a virtual environment are recommended.

### Installation

```bash
git clone https://github.com/ahammadmejbah/Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty.git
cd Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start the dashboard

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit. The ML benchmark trains on first access and is cached for the current data and code state.

### Open the paper

The paper can be opened directly in a browser. For a local HTTP preview:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/paper.html`. Use the browser print dialog to create a PDF preview.

## Project structure

```text
.
├── app.py           # Streamlit application and benchmark implementation
├── Jobs.csv         # Salary snapshot input data
├── paper.html       # Standalone IEEE-inspired paper artifact
├── requirements.txt # Runtime dependencies
└── README.md        # Project documentation
```

## Methodology roadmap

The next research release should:

1. Add a validated occupational AI-exposure index and a versioned role/ISCO mapping table.
2. Define inclusion, exclusion, duplicate, and missingness rules before fitting models.
3. Add sector and experience harmonization where the source supports it.
4. Fit pooled, fixed-effects, and Bayesian hierarchical baselines.
5. Use prior predictive checks, posterior predictive checks, R-hat, effective sample size, divergence diagnostics, and sensitivity analysis.
6. Evaluate forward predictions with time-aware splits and calibrated uncertainty intervals.
7. Publish posterior samples, model configuration, environment metadata, and reproducible figures.

## Limitations

The title-based exposure proxy is generated dynamically by `app.py`; it is not a field in `Jobs.csv`, is not validated against an occupational exposure classification, and must not be interpreted as a causal treatment variable.

Salary differences can reflect role mix, seniority, country composition, company selection, self-reporting, missingness, currency conversion, and changing sample coverage. The dataset does not provide repeated-person identifiers, so observed year-to-year patterns are not automatically individual wage trajectories. The current application reports descriptive evidence and predictive benchmarks only.

The outputs should support aggregate research and workforce planning, not automated employment decisions or labels about individual workers.

## Reproducibility

- The benchmark uses fixed `random_state=42` values where supported.
- The temporal holdout is explicit: all records before the latest observed year are training data and the latest year is test data.
- The application records the five benchmark metrics in a stable table.
- The Data quality view exports the filtered rows used for exploratory analysis.
- The standalone paper records the verified data snapshot date and current evidence status.

## Data availability

The source dataset is available on Kaggle:

<https://www.kaggle.com/datasets/debayank2024/ai-impact-on-jobs-and-salaries-2020-2026>

The implementation is available on GitHub:

<https://github.com/ahammadmejbah/Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty>

Please consult the upstream Kaggle page for provenance, licensing, and revisions. No additional license is declared in this repository; add the appropriate license file before redistributing the software or data.

## Citation

```bibtex
@misc{ahammad2026aiwages,
  author       = {Ahammad, Mejbah},
  title        = {Bayesian Hierarchical Inference of AI Exposure Effects on Global Wage Trajectories with Uncertainty},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/ahammadmejbah/Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty}
}
```

## Author

**Mejbah Ahammad**<br>
Lead AI Instructor & Research Scientist

© 2026 Mejbah Ahammad | Lead AI Instructor & Research Scientist Portfolio
