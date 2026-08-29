from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import class_imbalance_report, clean_breast_cancer_dataframe, model_feature_importance
from src.predict import BreastCancerPredictor
from src.utils import DEFAULT_DATA_PATH, REPORTS_DIR, artifact_paths, load_config, load_dataset, load_json


DIAGNOSIS_LABELS = {"B": "Benign", "M": "Malignant"}
COLOR_MAP = {"Benign": "#0f766e", "Malignant": "#b42318"}


def configure_page(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Breast Cancer AI",
        page_icon="BC",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bcai-primary: #1167b1;
            --bcai-teal: #0f766e;
            --bcai-red: #b42318;
            --bcai-ink: #182230;
            --bcai-muted: #667085;
            --bcai-bg: #f6f8fb;
            --bcai-line: #d9e2ec;
        }
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
            color: var(--bcai-ink);
        }
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--bcai-line);
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--bcai-ink);
        }
        .bcai-page-title {
            padding: 0.35rem 0 1rem 0;
            border-bottom: 1px solid var(--bcai-line);
            margin-bottom: 1.15rem;
        }
        .bcai-page-title p {
            color: var(--bcai-muted);
            font-size: 1rem;
            margin: 0.25rem 0 0 0;
        }
        .bcai-card {
            background: #ffffff;
            border: 1px solid var(--bcai-line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.04);
            min-height: 108px;
        }
        .bcai-card-label {
            color: var(--bcai-muted);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .bcai-card-value {
            color: var(--bcai-ink);
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.15;
        }
        .bcai-card-help {
            color: var(--bcai-muted);
            font-size: 0.8rem;
            margin-top: 0.35rem;
        }
        .risk-low {
            color: #0f766e;
            background: #ecfdf3;
            border: 1px solid #abefc6;
        }
        .risk-moderate {
            color: #b54708;
            background: #fffaeb;
            border: 1px solid #fedf89;
        }
        .risk-high {
            color: #b42318;
            background: #fef3f2;
            border: 1px solid #fecdca;
        }
        .risk-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-weight: 700;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--bcai-line);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="bcai-page-title">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, help_text: str = "") -> str:
    return f"""
    <div class="bcai-card">
        <div class="bcai-card-label">{label}</div>
        <div class="bcai-card-value">{value}</div>
        <div class="bcai-card-help">{help_text}</div>
    </div>
    """


@st.cache_data(show_spinner=False)
def load_app_data() -> pd.DataFrame:
    data = clean_breast_cancer_dataframe(load_dataset(DEFAULT_DATA_PATH))
    data["diagnosis_label"] = data["diagnosis"].map(DIAGNOSIS_LABELS)
    return data


@st.cache_data(show_spinner=False)
def load_app_metrics() -> dict[str, Any]:
    config = load_config()
    paths = artifact_paths(config)
    return load_json(paths["metrics_path"], default={})


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    config = load_config()
    paths = artifact_paths(config)
    return load_json(paths["metadata_path"], default={})


@st.cache_data(show_spinner=False)
def load_feature_names() -> list[str]:
    config = load_config()
    paths = artifact_paths(config)
    payload = load_json(paths["feature_names_path"], default={"feature_names": []})
    return payload.get("feature_names", [])


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    path = REPORTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    metadata = load_model_metadata()
    feature_names = metadata.get("feature_names", [])
    if not feature_names:
        return pd.DataFrame(columns=["feature", "normalized_importance"])
    try:
        predictor = BreastCancerPredictor()
        return model_feature_importance(predictor.model, feature_names)
    except Exception:
        return pd.DataFrame({"feature": feature_names, "normalized_importance": [0.0] * len(feature_names)})


@st.cache_resource(show_spinner=False)
def load_predictor() -> BreastCancerPredictor:
    return BreastCancerPredictor()


def artifact_ready() -> bool:
    paths = artifact_paths(load_config())
    return all(paths[key].exists() for key in ("model_path", "scaler_path", "feature_names_path", "metadata_path"))


def feature_groups(feature_names: list[str]) -> dict[str, list[str]]:
    return {
        "Mean": [feature for feature in feature_names if feature.endswith("_mean")],
        "Standard Error": [feature for feature in feature_names if feature.endswith("_se")],
        "Worst": [feature for feature in feature_names if feature.endswith("_worst")],
        "Other": [
            feature
            for feature in feature_names
            if not (feature.endswith("_mean") or feature.endswith("_se") or feature.endswith("_worst"))
        ],
    }


def feature_defaults(data: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {}
    for feature in feature_names:
        values = pd.to_numeric(data[feature], errors="coerce").dropna()
        min_value = float(values.min())
        max_value = float(values.max())
        median_value = float(values.median())
        step = float(max((max_value - min_value) / 200, 0.0001))
        defaults[feature] = {
            "min": min_value,
            "max": max_value,
            "median": median_value,
            "step": step,
        }
    return defaults


def class_distribution_chart(data: pd.DataFrame) -> go.Figure:
    counts = data["diagnosis_label"].value_counts().reset_index()
    counts.columns = ["diagnosis", "count"]
    fig = px.bar(
        counts,
        x="diagnosis",
        y="count",
        color="diagnosis",
        color_discrete_map=COLOR_MAP,
        text="count",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Patients", height=360)
    return fig


def risk_badge(level: str) -> str:
    css_class = "risk-low"
    if level == "Moderate Risk":
        css_class = "risk-moderate"
    elif level == "High Risk":
        css_class = "risk-high"
    return f'<span class="risk-pill {css_class}">{level}</span>'


def probability_gauge(malignant_probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=malignant_probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1167b1"},
                "steps": [
                    {"range": [0, 35], "color": "#d1fadf"},
                    {"range": [35, 65], "color": "#fef0c7"},
                    {"range": [65, 100], "color": "#fee4e2"},
                ],
                "threshold": {
                    "line": {"color": "#182230", "width": 3},
                    "thickness": 0.8,
                    "value": malignant_probability * 100,
                },
            },
            title={"text": "Malignant Probability"},
        )
    )
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def image_if_exists(path: str | Path) -> Path | None:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    return None


def imbalance_summary(data: pd.DataFrame) -> dict[str, Any]:
    return class_imbalance_report(data)

