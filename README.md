# Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty

## Streamlit dashboard

The repository includes an exploratory dashboard for `Jobs.csv`. The current repository does not contain the external occupational exposure index described in the research plan, so the dashboard uses a transparent, title-based exposure proxy and labels it as descriptive rather than causal.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app includes salary distributions, observed year-level trajectories, group-level uncertainty summaries, a wage-compression screening view, missingness diagnostics, and filtered-data export. Replace the proxy with a validated exposure index before making causal claims.