# Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty

## Streamlit dashboard

The repository includes an exploratory dashboard for `Jobs.csv`. The current repository does not contain the external occupational exposure index described in the research plan, so the dashboard uses a transparent, title-based exposure proxy and labels it as descriptive rather than causal.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app includes salary distributions, observed year-level trajectories, group-level uncertainty summaries, a wage-compression screening view, missingness diagnostics, and filtered-data export. Replace the proxy with a validated exposure index before making causal claims.

## Standalone IEEE-style paper

The repository also includes [paper.html](paper.html), a print-ready HTML reading artifact styled after an IEEE conference paper: US Letter geometry, Times New Roman typography, justified text, two-column flow, IEEE headings, equations, figures, tables, numbered references, Kaggle data availability, and the GitHub project citation. Open it directly in a browser or print it to PDF. For formal submission, convert the manuscript to the target conference's official IEEE LaTeX or Word template and run PDF eXpress validation.