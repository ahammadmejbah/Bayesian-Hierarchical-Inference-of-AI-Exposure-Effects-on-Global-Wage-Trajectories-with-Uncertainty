from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="AI Exposure and Wage Observatory",
    page_icon="📊",
    layout="wide",
)


DATA_PATH = Path(__file__).parent / "Jobs.csv"


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


st.title("AI Exposure and Wage Observatory")
st.caption("Exploratory analysis of the supplied AI/ML salary snapshots. Estimates are descriptive until an external occupational exposure index is joined.")

try:
    df = add_exposure_proxy(load_data(DATA_PATH))
except (OSError, pd.errors.ParserError, KeyError) as error:
    st.error(f"Could not load Jobs.csv: {error}")
    st.stop()

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

tab_overview, tab_hierarchy, tab_risk, tab_data = st.tabs(
    ["Overview", "Hierarchical summaries", "Compression screen", "Data quality"]
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