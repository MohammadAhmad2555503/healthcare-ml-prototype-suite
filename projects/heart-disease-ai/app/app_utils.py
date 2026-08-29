from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import feature_columns, load_dataset
from src.explainability import global_feature_importance
from src.predict import TabularClinicalPredictor
from src.utils import artifact_paths, load_config, load_json


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | {load_config()['project']['short_name']}", page_icon="AI", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f7fafc; }
        h1, h2, h3 { color: #182230; letter-spacing: 0; }
        .metric-card { background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 1rem; min-height: 104px; box-shadow: 0 8px 20px rgba(16,24,40,.04); }
        .metric-label { color: #667085; font-size: .85rem; }
        .metric-value { color: #182230; font-size: 1.45rem; font-weight: 700; line-height: 1.2; }
        .risk { display: inline-block; border-radius: 999px; padding: .35rem .75rem; font-weight: 700; }
        .low { background: #ecfdf3; color: #067647; border: 1px solid #abefc6; }
        .moderate { background: #fffaeb; color: #b54708; border: 1px solid #fedf89; }
        .high { background: #fef3f2; color: #b42318; border: 1px solid #fecdca; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_block(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def card(label: str, value: str) -> str:
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'


@st.cache_data(show_spinner=False)
def data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner=False)
def metrics() -> dict[str, object]:
    return load_json(artifact_paths(load_config())["metrics_path"], default={})


@st.cache_resource(show_spinner=False)
def predictor() -> TabularClinicalPredictor:
    return TabularClinicalPredictor()


def risk_badge(level: str) -> str:
    cls = "low"
    if level == "Moderate Risk":
        cls = "moderate"
    if level == "High Risk":
        cls = "high"
    return f'<span class="risk {cls}">{level}</span>'


def class_distribution_chart() -> go.Figure:
    df = data()
    config = load_config()
    target = config["data"]["target_column"]
    counts = df[target].value_counts().sort_index().reset_index()
    counts.columns = ["target", "count"]
    counts["label"] = counts["target"].astype(str).map(config["data"]["class_map"])
    fig = px.bar(counts, x="label", y="count", text="count", color="label")
    fig.update_layout(height=360, showlegend=False, xaxis_title="", yaxis_title="Patients")
    return fig


def probability_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Heart Disease Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1167b1"},
                "steps": [
                    {"range": [0, 40], "color": "#d1fadf"},
                    {"range": [40, 75], "color": "#fef0c7"},
                    {"range": [75, 100], "color": "#fee4e2"},
                ],
            },
        )
    )
    fig.update_layout(height=310)
    return fig


def feature_importance_chart() -> go.Figure | None:
    fi = global_feature_importance().head(20)
    if fi.empty:
        return None
    fig = px.bar(fi.sort_values("normalized_importance"), x="normalized_importance", y="feature", orientation="h")
    fig.update_layout(height=520)
    return fig

