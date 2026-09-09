from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


st.set_page_config(
    page_title="AI Exposure and Wage Observatory",
    page_icon="📊",
    layout="wide",
)


DATA_PATH = Path(__file__).parent / "Jobs.csv"
ML_CATEGORICAL = [
    "experience_level",
    "employment_type",
    "role_family",
    "employee_residence",
    "company_location",
    "company_size",
    "work_mode",
]
ML_NUMERIC = ["work_year", "remote_ratio"]


@st.cache_data(show_spinner=False)
def train_classical_models(data: pd.DataFrame) -> pd.DataFrame:
    features = data[ML_CATEGORICAL + ML_NUMERIC].copy()
    target = np.log(data["salary_in_usd"].clip(lower=1).astype(float))
    train_mask = data["work_year"] < data["work_year"].max()

    x_train, x_test = features[train_mask], features[~train_mask]
    y_train, y_test = target[train_mask], target[~train_mask]
    actual_salary = data.loc[~train_mask, "salary_in_usd"].to_numpy()

    linear_preprocessor = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
            ]), ML_CATEGORICAL),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), ML_NUMERIC),
        ]
    )
    ordinal_preprocessor = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), ML_CATEGORICAL),
            ("numeric", SimpleImputer(strategy="median"), ML_NUMERIC),
        ]
    )

    models = [
        ("Linear Regression", linear_preprocessor, LinearRegression()),
        ("Ridge Regression", linear_preprocessor, Ridge(alpha=10.0)),
        ("Lasso Regression", linear_preprocessor, Lasso(alpha=0.001, max_iter=3000)),
        ("Elastic Net", linear_preprocessor, ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=3000)),
        ("Random Forest", ordinal_preprocessor, RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1, min_samples_leaf=3)),
        ("Extra Trees", ordinal_preprocessor, ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=-1, min_samples_leaf=2)),
        ("Gradient Boosting", ordinal_preprocessor, GradientBoostingRegressor(n_estimators=120, random_state=42, loss="huber")),
        ("Hist. Gradient Boosting", ordinal_preprocessor, HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42)),
        ("AdaBoost", ordinal_preprocessor, AdaBoostRegressor(n_estimators=100, random_state=42, loss="square")),
        ("K-Nearest Neighbors", ordinal_preprocessor, KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)),
    ]

    results = []
    for name, preprocessor, estimator in models:
        model = Pipeline([( "preprocessor", preprocessor), ("model", estimator)])
        model.fit(x_train, y_train)
        predictions = np.exp(model.predict(x_test))
        predictions = np.maximum(predictions, 0)
        results.append({
            "Model": name,
            "MAE (USD)": mean_absolute_error(actual_salary, predictions),
            "RMSE (USD)": mean_squared_error(actual_salary, predictions) ** 0.5,
            "R²": r2_score(actual_salary, predictions),
            "MAPE (%)": np.mean(np.abs((actual_salary - predictions) / actual_salary)) * 100,
        })
    return pd.DataFrame(results).sort_values("RMSE (USD)").reset_index(drop=True)


def render_footer() -> None:
    st.markdown(
        """
        <style>
        .site-footer {
            border-top: 1px solid #c8d8d1;
            color: #687873;
            font-size: 0.82rem;
            margin-top: 2.5rem;
            padding: 1rem 0 0.5rem;
            text-align: center;
        }
        </style>
        <footer class="site-footer">© 2026 Mejbah Ahammad | Lead AI Instructor &amp; Research Scientist Portfolio</footer>
        """,
        unsafe_allow_html=True,
    )


def render_home(data: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        .top-nav-label { color: #176b5b; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em; text-align: center; text-transform: uppercase; }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] { border-color: #b9cec6; color: #176b5b; }
        .hero {
            padding: 2.5rem 2.8rem;
            border: 1px solid #c8d8d1;
            border-radius: 14px;
            background: linear-gradient(120deg, #e8f3ef 0%, #f7f3e8 100%);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .eyebrow { color: #176b5b; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
        .hero h1 { color: #153b36; font-size: 3rem; line-height: 1.05; margin: 0.45rem auto 0.8rem; max-width: 900px; }
        .hero p { color: #405550; font-size: 1.08rem; margin: 0 auto; max-width: 780px; }
        .doc-card { border-top: 3px solid #d58a4c; padding: 1rem 0.2rem 0.4rem; }
        .doc-card h3 { color: #153b36; margin-bottom: 0.35rem; }
        </style>
        <section class="hero">
            <div class="eyebrow">Research workspace · 2020–2025 salary snapshots</div>
            <h1>AI Exposure &amp; Wage Observatory</h1>
            <p>Explore how reported salaries vary across roles, countries, experience levels, and work modes while keeping uncertainty and data limits visible.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="top-nav-label">Navigate the research workspace</div>', unsafe_allow_html=True)
    nav_spacer_left, nav_home, nav_paper, nav_observatory, nav_spacer_right = st.columns([1, 1.2, 1.5, 1.2, 1])
    navigation = [(nav_home, "Home", "Home & documentation"), (nav_paper, "Research paper", "Research paper"), (nav_observatory, "Observatory", "Observatory")]
    for column, label, destination in navigation:
        with column:
            if st.button(label, key=f"home_nav_{destination}", use_container_width=True):
                st.session_state["page"] = destination
                st.rerun()

    overview = st.columns(4)
    overview[0].metric("Salary records", f"{len(data):,}")
    overview[1].metric("Countries", f"{data['employee_residence'].nunique():,}")
    overview[2].metric("Role families", f"{data['role_family'].nunique():,}")
    overview[3].metric("Years covered", f"{data['work_year'].nunique():,}")

    st.markdown("## Start here")
    cards = st.columns(3)
    with cards[0]:
        st.markdown('<div class="doc-card"><h3>1 · Explore</h3><p>Open <b>Observatory</b> in the sidebar. Filter years, experience, and work mode, then compare observed salary distributions and trajectories.</p></div>', unsafe_allow_html=True)
    with cards[1]:
        st.markdown('<div class="doc-card"><h3>2 · Compare</h3><p>Use <b>Hierarchical summaries</b> to inspect groups with sample-size thresholds and log-salary uncertainty intervals.</p></div>', unsafe_allow_html=True)
    with cards[2]:
        st.markdown('<div class="doc-card"><h3>3 · Audit</h3><p>Use <b>Data quality</b> to inspect missingness and download the filtered observations for reproducible follow-up work.</p></div>', unsafe_allow_html=True)

    st.markdown("## Documentation")
    with st.expander("Research purpose", expanded=True):
        st.write("This workspace is designed to investigate whether occupational AI exposure is associated with wage trajectories across countries and role families. The intended full study combines salary observations with a validated occupational exposure index and a Bayesian hierarchical model.")
        st.info("Current status: the repository contains salary snapshots only. The dashboard therefore reports descriptive relationships and a title-based screening proxy; it does not estimate a causal AI wage effect.")

    with st.expander("Data and field guide"):
        st.markdown(
            """
            | Field | Meaning |
            | --- | --- |
            | `work_year` | Year associated with the salary snapshot |
            | `salary_in_usd` | Salary converted to USD, used for visualizations |
            | `job_title` | Reported position title |
            | `role_family` | Normalized role grouping |
            | `employee_residence` | Country code for the employee's residence |
            | `experience_level_label` | Human-readable seniority category |
            | `work_mode` | On-site, hybrid, or remote classification |
            | `salary_outlier_flag` | Source-level outlier indicator |
            """
        )

    with st.expander("Exposure proxy: definition and limits"):
        st.write("The current proxy assigns 0.80 to titles containing AI, machine learning, software, developer, analytics, research, or related terms; 0.50 to a broader set of technology and product terms; and 0.20 otherwise.")
        st.markdown("**Interpretation rules:**")
        st.markdown("- It is a transparent screening feature, not the Gmyrek et al. exposure index.\n- It should not be read as a treatment assignment or causal estimate.\n- Keyword coverage can miss exposure in ordinary-sounding titles and overstate exposure in some specialized titles.\n- Replace it with an externally validated occupation-level score before fitting the proposed Bayesian model.")

    with st.expander("Uncertainty and model roadmap"):
        st.write("The current group summaries use the mean and standard deviation of log salary to form normal-approximation intervals. These intervals describe sampling variability within the available records; they are not posterior credible intervals.")
        st.markdown("**Planned model extensions:**")
        st.markdown("- Join a validated occupational AI exposure index using `role_family` and occupation codes.\n- Add country, sector, role, experience, and year effects with partial pooling.\n- Model role-specific salary variance and temporal dependence in a PyMC or Stan implementation.\n- Compare specifications with LOO-CV and report posterior predictive checks.")

    with st.expander("Reproducible setup"):
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
        st.write("The app reads `Jobs.csv` from the repository root. Use the download control in Data quality to export exactly the filtered rows used by the visible summaries.")

    st.caption("Source note: salary observations are exploratory and may reflect self-reporting, selection effects, currency conversion, and uneven coverage across countries and roles.")
    render_footer()


def render_paper(data: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        .paper-header { text-align: center; border-bottom: 1px solid #8c9895; padding: 1rem 0 1.4rem; margin-bottom: 1rem; }
        .paper-header h1 { color: #1c2f2b; font-family: Georgia, serif; font-size: 2rem; line-height: 1.12; margin: 0 auto 0.65rem; max-width: 900px; }
        .paper-header p { color: #596864; margin: 0.2rem 0; }
        .paper-column { color: #273532; font-family: Georgia, serif; font-size: 0.92rem; line-height: 1.48; text-align: justify; }
        .paper-column h2 { color: #176b5b; font-family: Georgia, serif; font-size: 1.08rem; margin: 1.2rem 0 0.35rem; }
        .paper-column h3 { color: #405550; font-family: Georgia, serif; font-size: 0.96rem; margin: 0.9rem 0 0.25rem; }
        .paper-note { background: #f6f1e6; border-left: 3px solid #d58a4c; padding: 0.7rem 0.8rem; margin: 0.8rem 0; font-family: Georgia, serif; }
        .paper-footer { border-top: 1px solid #8c9895; color: #687873; font-size: 0.8rem; margin-top: 1.5rem; padding-top: 0.7rem; }
        </style>
        <div class="paper-header">
            <h1>Bayesian Hierarchical Inference of AI Exposure Effects on Global Wage Trajectories with Uncertainty</h1>
            <p><b>Mejbah Ahammad</b> · Independent Research Workspace</p>
            <p>Conference-style research paper · Exploratory data release</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Reading view · IEEE-inspired double-column layout. The empirical claims below are bounded by the data currently available in this repository.")
    st.markdown(
        """
        <div class="paper-note">
        <b>Table of Contents</b><br>
        1. Abstract · 2. Keywords · 3. Introduction · 4. Related Work · 5. Research Methodology ·<br>
        6. Dataset Description · 7. Data Preprocessing · 8. Exploratory Data Analysis · 9. Feature Engineering ·<br>
        10. Proposed Methodology · 11. Experimental Setup · 12. Results and Analysis · 13. Comparative Analysis ·<br>
        14. Discussion · 15. Implications and Applications · 16. Limitations · 17. Conclusion and Future Work ·<br>
        18. Data Availability · 19. Conflict of Interest · 20. References
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""
            <div class="paper-column">
            <h2>Abstract</h2>
            <p>Generative artificial intelligence has renewed questions about how technology changes the distribution of work and wages. This project develops an uncertainty-aware framework for studying associations between occupational AI exposure and reported wage trajectories. The intended design combines salary snapshots with a validated occupational exposure index and estimates heterogeneous effects with a Bayesian hierarchical model across countries, sectors, roles, experience levels, and years.</p>
            <p>The current release is an exploratory observatory built from the supplied salary file. It provides descriptive summaries, transparent title-based exposure screening, and group-level uncertainty intervals. Because the external exposure index and full longitudinal linkage are not yet present, this release does not claim a causal AI wage effect. It is a reproducible foundation for the planned model rather than a completed causal analysis.</p>
            <p><b>Index Terms—</b> generative AI, wages, hierarchical Bayesian modeling, measurement error, labor economics, uncertainty quantification.</p>

            <h2>I. Introduction</h2>
            <p>The rapid deployment of generative AI has produced competing expectations: augmentation may increase the productivity and compensation of complementary workers, while substitution may reduce demand for tasks that can be automated. Aggregate narratives obscure the fact that occupational exposure, institutional context, experience, and selection into reported salary datasets can differ substantially across places and roles.</p>
            <p>This study asks four questions. First, what is the posterior distribution of exposure effects after accounting for salary uncertainty? Second, how do geography, sector, and role composition moderate those effects? Third, does model averaging improve forecasts over fixed-effects baselines? Fourth, which groups have the greatest posterior probability of compression or augmentation?</p>

            <h2>II. Research Contributions</h2>
            <p>The planned contribution is a single framework that propagates measurement uncertainty through a multilevel wage model. Country, sector, role, experience, and year effects permit partial pooling: sparse groups borrow information while retaining context-specific deviations. Competing specifications can be combined with leave-one-out predictive weights rather than selecting one model as if it were known to be correct.</p>

            <h2>III. Data</h2>
            <p>The current repository contains <b>{len(data):,}</b> salary records spanning <b>{data['work_year'].min():.0f}–{data['work_year'].max():.0f}</b>. Available fields include salary in USD, job title, normalized role family, employee residence, experience level, work mode, company location, and a source-level outlier flag.</p>
            <p>Salary observations are self-reported or otherwise subject to selection and measurement error. The planned integration will join these records to an external occupation-level exposure score using role and ISCO mappings. That score is not included in the current repository, so the application uses a title-based proxy only for exploratory screening.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="paper-column">
            <h2>IV. Methodology</h2>
            <p>Let <i>y</i><sub>ijktℓ</sub> denote log salary for country <i>i</i>, sector <i>j</i>, role <i>k</i>, year <i>t</i>, and experience level ℓ. The proposed model uses:</p>
            <div class="paper-note"><i>y</i><sub>ijktℓ</sub> ~ Normal(μ<sub>ijktℓ</sub>, σ<sub>role,k</sub>)<br><br>μ = α + βx + α<sub>country,i</sub> + α<sub>sector,j</sub> + α<sub>role,k</sub> + β<sub>country,i</sub>x + γ<sub>year,t</sub> + δ<sub>experience,ℓ</sub></div>
            <p>Here, <i>x</i> is the validated AI exposure score. Group-level priors provide partial pooling, role-specific scales represent heterogeneous reporting noise, and temporal priors capture dependence across adjacent years. Horseshoe or other shrinkage priors can regularize weakly identified exposure coefficients. Hamiltonian Monte Carlo in PyMC or Stan is the intended inference engine.</p>

            <h2>V. Current Observatory</h2>
            <p>The application operationalizes the first stage of the research workflow. It computes log salaries, applies filters, displays median trajectories, compares exposure-proxy bands, and reports normal-approximation intervals for selected groups. These summaries are useful for data auditing and hypothesis formation, but they are not posterior estimates.</p>
            <p>The title proxy assigns higher scores to terms such as AI, machine learning, software, developer, analytics, and research; moderate scores to broader product, engineering, management, and technology terms; and a lower score otherwise. It is intentionally visible and easy to replace.</p>

            <h2>VI. Results Status</h2>
            <p>No causal result is reported in this release. The hypothesized premiums and compression patterns in the project brief remain preregistered research expectations, not findings supported by the present app. A defensible results section requires the exposure index, a defined panel construction rule, model diagnostics, posterior predictive checks, and out-of-sample validation.</p>

            <h2>VII. Limitations and Ethics</h2>
            <p>Salary coverage is uneven across countries, roles, years, and experience levels. Reported salaries may reflect currency conversion, nonresponse, selection into the source, and unobserved company characteristics. Exposure scores can encode occupational assumptions and should not be used to label individual workers as replaceable. Findings should support aggregate workforce planning, not automated decisions about people.</p>

            <h2>VIII. Reproducibility</h2>
            <p>Install the pinned-range dependencies from <i>requirements.txt</i> and run <i>streamlit run app.py</i>. The Data quality view exports the exact filtered rows used in a session. The next release should add the validated exposure file, model scripts, posterior samples, convergence diagnostics, and a versioned mapping table.</p>

            <h2>References</h2>
            <p>[1] International Labour Organization, “Generative AI and Jobs,” 2023.</p>
            <p>[2] M. Gmyrek et al., “Generative AI and Jobs: A Global Analysis of Potential Effects on Job Quantity and Quality,” ILO, 2025.</p>
            <p>[3] A. Gelman et al., <i>Bayesian Data Analysis</i>, 3rd ed. CRC Press, 2013.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Detailed analysis protocol")
    protocol_left, protocol_right = st.columns(2, gap="large")
    with protocol_left:
        st.markdown(
            """
            <div class="paper-column">
            <h2>IX. Estimand and Identification</h2>
            <p>The primary estimand is the change in expected log salary associated with a one-unit increase in occupational exposure, conditional on the observed country, sector, role, experience, and year structure. Country-specific slopes translate the global association into a distribution of context-dependent effects. These are associational estimands unless the exposure design and confounding assumptions are defended with additional evidence.</p>
            <p>The design does not treat a cross-sectional salary difference as an intervention effect. Identification requires that the exposure index be measured before or independently of the wage outcome, that important confounders be represented or bounded, and that changes in sample composition are not mistaken for wage dynamics. A sensitivity analysis should vary these assumptions explicitly.</p>

            <h2>X. Panel Construction</h2>
            <p>Each usable record is mapped to a role family, residence country, sector or ISCO group, experience level, and calendar year. Positive USD salaries are transformed as <i>log(wage)</i> to reduce right-skew and interpret coefficients approximately as proportional changes. Duplicate records, impossible values, and missing keys are flagged before analysis rather than silently discarded.</p>
            <p>For a longitudinal comparison, the pipeline should define whether a record is a repeated snapshot, an independent report, or an aggregate cell. It should also report coverage by country-year-role cell, retain a missingness audit, and prevent future observations from entering a training split used to forecast earlier periods.</p>

            <h2>XI. Measurement Error</h2>
            <p>Self-reported salaries are modeled as noisy measurements of latent compensation. A role-level scale, σ<sub>role,k</sub>, allows reporting variance to differ across occupations. Where repeated reports or source quality metadata are available, they can inform this scale; otherwise, weakly informative half-normal priors should prevent implausibly precise estimates without overwhelming the data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with protocol_right:
        st.markdown(
            """
            <div class="paper-column">
            <h2>XII. Prior Specification</h2>
            <p>The global intercept receives a broad normal prior on log salary. The global exposure coefficient receives a regularizing normal or horseshoe prior centered at zero. Country and role intercepts and slopes are drawn from shared distributions with half-normal scale priors. Year effects use an autoregressive structure, γ<sub>t</sub> ~ Normal(ρ γ<sub>t−1</sub>, σ<sub>year</sub>), with ρ constrained to a stationary interval.</p>
            <p>Prior predictive simulation is required before sampling the observed data. It should verify that plausible salary ranges, exposure effects, and between-group variation are represented while extreme values receive low probability. Sensitivity runs should compare regularized normal, Student-t, and horseshoe specifications.</p>

            <h2>XIII. Inference and Diagnostics</h2>
            <p>The planned sampler uses four chains with warmup and sufficient post-warmup draws for stable tail probabilities. The analysis reports R-hat, effective sample size, divergent transitions, maximum tree depth, energy diagnostics, and Monte Carlo standard error. A model is not treated as converged because a single summary statistic looks acceptable.</p>
            <p>Posterior predictive checks compare observed and replicated salary distributions, group medians, year trends, and extreme-value behavior. Calibration is assessed on held-out country-year or time-based splits, with interval coverage reported alongside point prediction error.</p>

            <h2>XIV. Model Averaging</h2>
            <p>Candidate specifications vary random slopes, temporal effects, likelihood tails, and shrinkage priors. Pareto-smoothed leave-one-out cross-validation provides predictive scores and stacking weights. Ensemble intervals must combine both within-model posterior uncertainty and between-model disagreement; weights are not evidence that one specification is true.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="paper-column">
        <h2>XV. Planned Reporting Standard</h2>
        <p>Every substantive claim will be accompanied by its estimand, unit of exposure, reference group, posterior mean or median, 95% credible interval, probability of a positive or negative effect, effective sample size, and relevant coverage count. Country and occupation rankings will include uncertainty rather than presenting a deterministic league table.</p>
        <p>The final report will distinguish three layers of evidence: descriptive patterns in the raw snapshots, model-based associations after adjustment and partial pooling, and predictive performance on held-out data. A result will not be called a “wage premium,” “compression effect,” or “causal impact” unless the corresponding design and assumptions justify that language.</p>

        <h2>XVI. Release Checklist</h2>
        <p>A complete research release requires: (1) a versioned exposure-index file and mapping table; (2) documented inclusion and exclusion rules; (3) baseline fixed-effects and non-Bayesian benchmarks; (4) prior predictive and posterior predictive plots; (5) convergence and sensitivity tables; (6) time-aware out-of-sample evaluation; (7) posterior samples with an archival checksum; and (8) a data dictionary describing licensing, provenance, and known gaps.</p>

        <h2>XVII. Conclusion</h2>
        <p>This project provides a practical bridge between exploratory labor-market visualization and a fully specified Bayesian research workflow. The current observatory makes the data structure and uncertainty limits visible. The next scientific step is not to strengthen the narrative around a hypothesized result, but to add the missing exposure measure, execute the preregistered protocol, and let diagnostics determine which conclusions the evidence can support.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Paper sections and empirical exhibits")
    section_left, section_right = st.columns(2, gap="large")
    with section_left:
        st.markdown(
            f"""
            <div class="paper-column">
            <h2>IV. Related Work</h2>
            <p>Prior work on technology and labor markets spans task-level automation indices, occupational exposure measures, firm adoption studies, and wage-distribution analyses. The literature consistently warns that exposure is not equivalent to displacement: the same technology can substitute for some tasks while complementing others. This project follows that distinction by estimating heterogeneous exposure slopes rather than a single universal premium.</p>
            <p>Two methodological gaps motivate the design. First, salary observations often contain reporting error that is ignored when point estimates are treated as exact. Second, occupational and geographic heterogeneity is frequently handled with fixed effects that do not share information efficiently across sparse groups. Hierarchical Bayesian inference provides a direct way to represent both uncertainties.</p>

            <h2>V. Research Methodology</h2>
            <p>The workflow has four stages: validate and document the source data; join each role to an external exposure score; estimate baseline and hierarchical models; and evaluate predictions on held-out time or country-year cells. The application currently implements stage one and a transparent screening version of stage two.</p>

            <h2>VI. Dataset Description</h2>
            <p>The local file contains <b>{len(data):,}</b> records, {data['employee_residence'].nunique():,} employee-residence countries, {data['role_family'].nunique():,} role families, and years {data['work_year'].min():.0f}–{data['work_year'].max():.0f}. It includes salary in USD, original salary metadata, job title, experience level, employment type, remote ratio, company location, company size, role family, and ISCO group hint.</p>

            <h2>VII. Data Preprocessing</h2>
            <p>Numeric year and USD salary fields are coerced to numeric types. Records without usable year or positive salary are excluded from the analysis frame. Salary is transformed to log scale for group-level uncertainty summaries. Missing role and country labels are retained under explicit “Other / Unclassified” or “Unknown” categories where appropriate.</p>

            <h2>VIII. Exploratory Data Analysis</h2>
            <p>The exhibits below show observed salary patterns, not estimates of an AI treatment effect. They are intended to expose coverage, skew, sparse groups, and candidate relationships that the full model must test.</p>

            <h2>IX. Feature Engineering</h2>
            <p>Derived fields include log salary, a title-based exposure screening score, exposure-proxy bands, categorical group identifiers, and group-level standard errors. The proxy is deliberately interpretable: it is useful for visual triage, but it must be replaced by a validated occupation-level score for formal inference.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with section_right:
        st.markdown(
            """
            <div class="paper-column">
            <h2>X. Proposed Methodology</h2>
            <p>The proposed model estimates global, country-specific, sector-specific, role-specific, experience-specific, and year-level components. It uses partial pooling, role-specific observation scales, autoregressive year effects, and regularization for exposure coefficients. Posterior quantities include global and subgroup exposure effects, probabilities of compression, and predictive distributions for future wage trajectories.</p>

            <h2>XI. Experimental Setup</h2>
            <p>Baseline comparisons should include pooled regression, country and role fixed effects, regularized linear models, and a tree-based predictive benchmark. The hierarchical models should be evaluated with four chains, prior predictive checks, posterior predictive checks, R-hat, effective sample size, divergence counts, and time-aware held-out validation. All preprocessing decisions must be fitted inside each training split.</p>

            <h2>XII. Results and Analysis</h2>
            <p>Current results are descriptive: the app reports salary medians, log-salary intervals, coverage counts, and trajectories by observed group. No posterior AI exposure coefficient is reported because the required external exposure index is not present in this repository.</p>

            <h2>XIII. Comparative Analysis</h2>
            <p>The intended comparison is between fixed-effects baselines, single hierarchical specifications, and a stacked ensemble. Performance should be reported with held-out log predictive density, RMSE or MAE on log salary, interval coverage, interval width, and calibration of compression probabilities.</p>

            <h2>XIV. Discussion</h2>
            <p>Observed differences across proxy bands can reflect occupation mix, country composition, seniority, company selection, or reporting behavior. They should therefore be treated as hypotheses for the full model, not evidence that AI exposure caused a wage change.</p>

            <h2>XV. Implications and Applications</h2>
            <p>A validated uncertainty-aware model could support workforce scenario planning, reskilling prioritization, compensation benchmarking, and public policy analysis. Outputs should be aggregated, probabilistic, and accompanied by coverage warnings; they should not be used to score individual workers or automate employment decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    yearly_exhibit = data.groupby("work_year", as_index=False).agg(
        median_salary=("salary_in_usd", "median"),
        mean_salary=("salary_in_usd", "mean"),
        observations=("salary_in_usd", "size"),
    )
    role_table = data.groupby("role_family", as_index=False).agg(
        observations=("salary_in_usd", "size"),
        median_salary=("salary_in_usd", "median"),
        mean_exposure_proxy=("exposure_proxy", "mean"),
    ).sort_values("observations", ascending=False).head(12)

    exhibit_top_left, exhibit_top_right = st.columns(2, gap="large")
    with exhibit_top_left:
        st.markdown("### Figure 1. Observed salary trajectory")
        fig = px.line(
            yearly_exhibit,
            x="work_year",
            y=["median_salary", "mean_salary"],
            markers=True,
            labels={"value": "Salary (USD)", "work_year": "Work year", "variable": "Statistic"},
            title="Observed salary trajectory by year",
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: repository-local Jobs.csv. Descriptive summaries; not causal estimates.")
    with exhibit_top_right:
        st.markdown("### Table 1. Highest-volume roles")
        st.dataframe(
            role_table.style.format({"median_salary": "${:,.0f}", "mean_exposure_proxy": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Source: repository-local Jobs.csv. Exposure values are title-based screening scores.")

    exhibit_left, exhibit_right = st.columns(2, gap="large")
    with exhibit_left:
        st.markdown("### Figure 2. Salary by exposure screening band")
        proxy_exhibit = data.groupby("exposure_band", observed=False, as_index=False).agg(
            median_salary=("salary_in_usd", "median"), observations=("salary_in_usd", "size")
        )
        fig = px.bar(
            proxy_exhibit,
            x="exposure_band",
            y="median_salary",
            color="exposure_band",
            text="observations",
            category_orders={"exposure_band": ["Lower proxy", "Moderate proxy", "Higher proxy"]},
            labels={"exposure_band": "Title-based proxy band", "median_salary": "Median salary (USD)"},
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: repository-local Jobs.csv. Proxy bands do not represent validated occupational exposure.")
    with exhibit_right:
        st.markdown("### Figure 3. Coverage by experience")
        experience_exhibit = data["experience_level_label"].value_counts().rename_axis("experience").reset_index(name="observations")
        fig = px.bar(
            experience_exhibit,
            x="experience",
            y="observations",
            color="experience",
            labels={"experience": "Experience level", "observations": "Salary records"},
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: repository-local Jobs.csv. Counts indicate coverage, not population prevalence.")

    final_left, final_right = st.columns(2, gap="large")
    with final_left:
        st.markdown(
            """
            <div class="paper-column">
            <h2>XVI. Limitations</h2>
            <p>The current release lacks the external exposure index, sector normalization, repeated-person identifiers, and a fully defined causal identification strategy. The source may overrepresent workers who report salaries online. Country and role cells are unbalanced, USD conversions may hide local purchasing-power differences, and the title proxy can misclassify exposure. These limitations affect both effect estimates and predictive calibration.</p>

            <h2>XVII. Conclusion and Future Work</h2>
            <p>This release establishes a reproducible, inspectable research interface. Future work will add the occupation-level exposure data, finalize the panel mapping, fit the hierarchical PyMC model, compare baselines, validate predictions on held-out years, and publish posterior samples with diagnostic artifacts. The project will report what the data support, including null or uncertain effects.</p>

            <h2>XVIII. Data Availability</h2>
            <p>The source dataset is available through Kaggle: <a href="https://www.kaggle.com/datasets/debayank2024/ai-impact-on-jobs-and-salaries-2020-2026" target="_blank">AI Impact on Jobs and Salaries (2020–2026)</a>. The current repository contains the working CSV used by this dashboard. Users should consult the Kaggle page for provenance, licensing, and any upstream revisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with final_right:
        st.markdown(
            """
            <div class="paper-column">
            <h2>XIX. Conflict of Interest</h2>
            <p>The author declares no financial conflict of interest related to the salary dataset, exposure-index methodology, or dashboard implementation. This statement should be updated if funding, consulting, employment, or data-provider relationships change.</p>

            <h2>XX. References</h2>
            <p>[1] International Labour Organization, “Generative AI and Jobs,” 2023.</p>
            <p>[2] M. Gmyrek et al., “Generative AI and Jobs: A Global Analysis of Potential Effects on Job Quantity and Quality,” ILO, 2025.</p>
            <p>[3] A. Gelman et al., <i>Bayesian Data Analysis</i>, 3rd ed. CRC Press, 2013.</p>
            <p>[4] R. Vehtari, A. Gelman, and J. Gabry, “Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC,” <i>Statistics and Computing</i>, 2017.</p>
            <p>[5] D. Ahammad, “Bayesian Hierarchical Inference of AI Exposure Effects on Global Wage Trajectories with Uncertainty,” GitHub repository, 2026. <a href="https://github.com/ahammadmejbah/Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty" target="_blank">github.com/ahammadmejbah/Bayesian-Hierarchical-Inference-of-AI-Exposure-Effects-on-Global-Wage-Trajectories-with-Uncertainty</a>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="paper-footer">AI Exposure and Wage Observatory · Reproducibility-first exploratory release · Last data check: repository-local Jobs.csv</div>', unsafe_allow_html=True)
    render_footer()


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["work_year"] = pd.to_numeric(data["work_year"], errors="coerce")
    data["salary_in_usd"] = pd.to_numeric(data["salary_in_usd"], errors="coerce")
    data["log_salary"] = np.log(data["salary_in_usd"].clip(lower=1))
    data["role_family"] = data["role_family"].fillna("Other / Unclassified")
    data["employee_residence"] = data["employee_residence"].fillna("Unknown")
    data["company_size"] = data["company_size"].fillna("Unknown")
    return data.dropna(subset=["work_year", "salary_in_usd"])


def add_exposure_proxy(data: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent title-based screening proxy, not a causal exposure score."""
    high = r"ai|machine learning|data scientist|data engineer|software|developer|analytics|research"
    medium = r"product|engineering|designer|marketing|consult|manager|architect|scientist"
    title = data["job_title"].fillna("").str.lower()
    data = data.copy()
    data["exposure_proxy"] = np.select(
        [title.str.contains(high, regex=True), title.str.contains(medium, regex=True)],
        [0.80, 0.50],
        default=0.20,
    )
    data["exposure_band"] = pd.cut(
        data["exposure_proxy"],
        bins=[-0.01, 0.25, 0.65, 1.0],
        labels=["Lower proxy", "Moderate proxy", "Higher proxy"],
    )
    return data


def group_summary(data: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = (
        data.groupby(group, observed=True)
        .agg(
            observations=("salary_in_usd", "size"),
            median_salary=("salary_in_usd", "median"),
            mean_log_salary=("log_salary", "mean"),
            sd_log_salary=("log_salary", "std"),
            mean_exposure=("exposure_proxy", "mean"),
        )
        .reset_index()
    )
    summary["sd_log_salary"] = summary["sd_log_salary"].fillna(0)
    summary["se_log_salary"] = summary["sd_log_salary"] / np.sqrt(summary["observations"].clip(lower=1))
    summary["lower_salary"] = np.exp(summary["mean_log_salary"] - 1.96 * summary["se_log_salary"])
    summary["upper_salary"] = np.exp(summary["mean_log_salary"] + 1.96 * summary["se_log_salary"])
    return summary.sort_values("observations", ascending=False)


try:
    df = add_exposure_proxy(load_data(DATA_PATH))
except (OSError, pd.errors.ParserError, KeyError) as error:
    st.error(f"Could not load Jobs.csv: {error}")
    st.stop()

workspace_views = ["Home & documentation", "Research paper", "Observatory"]
if "page" not in st.session_state:
    st.session_state["page"] = workspace_views[0]

with st.sidebar:
    st.markdown("## AI Wage Observatory")
    view = st.radio(
        "Workspace",
        workspace_views,
        index=workspace_views.index(st.session_state["page"]),
        label_visibility="collapsed",
    )
    st.divider()

st.session_state["page"] = view

if view == "Home & documentation":
    render_home(df)
    st.stop()

if view == "Research paper":
    render_paper(df)
    st.stop()

st.title("AI Exposure and Wage Observatory")
st.caption("Exploratory analysis of the supplied AI/ML salary snapshots. Estimates are descriptive until an external occupational exposure index is joined.")

with st.sidebar:
    st.header("Filters")
    years = sorted(df["work_year"].astype(int).unique())
    selected_years = st.slider("Work years", min(years), max(years), (min(years), max(years)))
    experience = st.multiselect(
        "Experience level",
        sorted(df["experience_level_label"].dropna().unique()),
        default=sorted(df["experience_level_label"].dropna().unique()),
    )
    modes = st.multiselect(
        "Work mode",
        sorted(df["work_mode"].dropna().unique()),
        default=sorted(df["work_mode"].dropna().unique()),
    )
    min_observations = st.number_input("Minimum observations per group", min_value=1, value=5, step=1)

filtered = df[
    df["work_year"].between(selected_years[0], selected_years[1])
    & df["experience_level_label"].isin(experience)
    & df["work_mode"].isin(modes)
].copy()

if filtered.empty:
    st.warning("No observations match these filters.")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Observations", f"{len(filtered):,}")
metric_cols[1].metric("Median salary", f"${filtered['salary_in_usd'].median():,.0f}")
metric_cols[2].metric("Countries", f"{filtered['employee_residence'].nunique():,}")
metric_cols[3].metric("Role families", f"{filtered['role_family'].nunique():,}")

tab_overview, tab_hierarchy, tab_risk, tab_ml, tab_data = st.tabs(
    ["Overview", "Hierarchical summaries", "Compression screen", "ML benchmark", "Data quality"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        role_summary = group_summary(filtered, "role_family").head(15)
        fig = px.bar(
            role_summary.sort_values("median_salary"),
            x="median_salary",
            y="role_family",
            orientation="h",
            color="mean_exposure",
            color_continuous_scale="Tealgrn",
            labels={"median_salary": "Median salary (USD)", "mean_exposure": "Exposure proxy"},
            title="Median salary by role family",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        trend = filtered.groupby("work_year", as_index=False).agg(
            median_salary=("salary_in_usd", "median"),
            observations=("salary_in_usd", "size"),
        )
        fig = px.line(
            trend,
            x="work_year",
            y="median_salary",
            markers=True,
            hover_data=["observations"],
            labels={"work_year": "Year", "median_salary": "Median salary (USD)"},
            title="Observed salary trajectory",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        filtered,
        x="exposure_band",
        y="salary_in_usd",
        color="exposure_band",
        category_orders={"exposure_band": ["Lower proxy", "Moderate proxy", "Higher proxy"]},
        points=False,
        log_y=True,
        title="Salary distribution by title-based exposure proxy",
        labels={"exposure_band": "Exposure proxy band", "salary_in_usd": "Salary (USD, log scale)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("The exposure proxy is a screening aid based on job-title keywords. It is not the Gmyrek et al. occupational exposure index and should not be interpreted as a causal treatment variable.")

with tab_hierarchy:
    st.subheader("Partial-pooling style summaries")
    st.write("Group summaries use log-salary means and 95% normal-approximation intervals. Groups with few observations are intentionally surfaced as uncertain.")
    hierarchy_level = st.selectbox("Summarize by", ["role_family", "employee_residence", "experience_level_label", "work_mode"])
    summary = group_summary(filtered, hierarchy_level)
    summary = summary[summary["observations"] >= min_observations]
    st.dataframe(
        summary[[hierarchy_level, "observations", "median_salary", "lower_salary", "upper_salary", "mean_exposure"]]
        .rename(columns={"lower_salary": "lower 95% salary", "upper_salary": "upper 95% salary", "mean_exposure": "mean exposure proxy"})
        .style.format({"median_salary": "${:,.0f}", "lower 95% salary": "${:,.0f}", "upper 95% salary": "${:,.0f}", "mean exposure proxy": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_risk:
    st.subheader("Wage compression screen")
    st.write("This screen compares observed salary trends across proxy bands. It does not identify causal AI effects or forecast future wages.")
    band_year = filtered.groupby(["work_year", "exposure_band"], observed=False).agg(
        median_salary=("salary_in_usd", "median"), observations=("salary_in_usd", "size")
    ).reset_index()
    fig = px.line(
        band_year,
        x="work_year",
        y="median_salary",
        color="exposure_band",
        markers=True,
        hover_data=["observations"],
        labels={"median_salary": "Median salary (USD)", "exposure_band": "Exposure proxy"},
        title="Median salary trajectories by exposure proxy",
    )
    st.plotly_chart(fig, use_container_width=True)
    country = group_summary(filtered, "employee_residence")
    country["uncertainty_width"] = country["upper_salary"] - country["lower_salary"]
    st.dataframe(
        country[["employee_residence", "observations", "median_salary", "uncertainty_width"]].head(25)
        .style.format({"median_salary": "${:,.0f}", "uncertainty_width": "${:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_ml:
    st.subheader("10 classical machine-learning models")
    st.write("Models predict log salary using role, geography, seniority, work mode, year, and remote ratio. Training uses 2020–2024 and the latest year is held out for testing, so the comparison is predictive rather than causal.")
    with st.spinner("Training ten models on the time-aware split..."):
        benchmark = train_classical_models(df)
    st.caption("Five-column performance table: lower MAE, RMSE, and MAPE are better; higher R² is better. Metrics are calculated on the held-out latest year in USD.")
    st.dataframe(
        benchmark.style.format({"MAE (USD)": "${:,.0f}", "RMSE (USD)": "${:,.0f}", "R²": "{:.3f}", "MAPE (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    chart = benchmark.sort_values("RMSE (USD)", ascending=True)
    fig = px.bar(
        chart,
        x="RMSE (USD)",
        y="Model",
        orientation="h",
        color="R²",
        color_continuous_scale="Tealgrn",
        title="Held-out RMSE by model",
        labels={"RMSE (USD)": "RMSE (USD)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("These are benchmark predictions of reported salaries. They do not estimate the causal effect of AI exposure, and the model features do not include the missing external exposure index.")

with tab_data:
    st.subheader("Coverage and missingness")
    quality = pd.DataFrame({"field": filtered.columns, "missing": filtered.isna().sum().values})
    quality["missing_pct"] = quality["missing"] / len(filtered)
    st.dataframe(quality.sort_values("missing", ascending=False).style.format({"missing_pct": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered observations",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_jobs.csv",
        mime="text/csv",
    )

render_footer()