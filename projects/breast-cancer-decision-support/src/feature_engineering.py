from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.utils import ID_COLUMNS, TARGET_COLUMN, encode_diagnosis, get_feature_columns


@dataclass(frozen=True)
class OutlierSummary:
    feature: str
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float


def clean_breast_cancer_dataframe(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    removable_columns = [
        col
        for col in cleaned.columns
        if str(col).strip() == "" or str(col).startswith("Unnamed") or cleaned[col].isna().all()
    ]
    cleaned = cleaned.drop(columns=removable_columns, errors="ignore")

    if target_column in cleaned.columns:
        cleaned[target_column] = cleaned[target_column].astype(str).str.strip().str.upper()

    for col in get_feature_columns(cleaned, target_column):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def feature_dataframe(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    columns = get_feature_columns(df, target_column)
    return df[columns].apply(pd.to_numeric, errors="coerce")


def detect_outliers_iqr(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    features = feature_dataframe(df, target_column)
    summaries: list[OutlierSummary] = []
    for column in features.columns:
        values = features[column].dropna()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (features[column] < lower) | (features[column] > upper)
        count = int(mask.sum())
        percentage = float((count / len(features)) * 100) if len(features) else 0.0
        summaries.append(
            OutlierSummary(
                feature=column,
                lower_bound=float(lower),
                upper_bound=float(upper),
                count=count,
                percentage=percentage,
            )
        )
    return pd.DataFrame([summary.__dict__ for summary in summaries]).sort_values(
        ["count", "percentage"], ascending=False
    )


def correlation_analysis(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    high_corr_threshold: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_dataframe(df, target_column)
    corr = features.corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []

    for idx, feature_a in enumerate(corr.columns):
        for feature_b in corr.columns[idx + 1 :]:
            value = corr.loc[feature_a, feature_b]
            if pd.notna(value) and abs(value) >= high_corr_threshold:
                pairs.append(
                    {
                        "feature_1": feature_a,
                        "feature_2": feature_b,
                        "correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    }
                )

    high_corr = pd.DataFrame(pairs)
    if not high_corr.empty:
        high_corr = high_corr.sort_values("absolute_correlation", ascending=False)
    return corr, high_corr


def class_imbalance_report(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    encoded = encode_diagnosis(df[target_column])
    counts = encoded.value_counts().sort_index()
    benign = int(counts.get(0, 0))
    malignant = int(counts.get(1, 0))
    total = benign + malignant
    majority = max(benign, malignant)
    minority = min(benign, malignant)
    imbalance_ratio = float(majority / minority) if minority else float("inf")
    return {
        "total": total,
        "benign": benign,
        "malignant": malignant,
        "benign_percentage": float(benign / total * 100) if total else 0.0,
        "malignant_percentage": float(malignant / total * 100) if total else 0.0,
        "imbalance_ratio": imbalance_ratio,
        "assessment": "balanced" if imbalance_ratio <= 1.5 else "moderately imbalanced" if imbalance_ratio <= 3 else "highly imbalanced",
    }


def add_target_numeric(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    enriched = df.copy()
    enriched[f"{target_column}_numeric"] = encode_diagnosis(enriched[target_column])
    return enriched


def model_feature_importance(
    model: Any,
    feature_names: list[str],
    X: np.ndarray | pd.DataFrame | None = None,
    y: np.ndarray | pd.Series | None = None,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "model_feature_importances"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).reshape(-1)
        method = "absolute_model_coefficients"
    elif X is not None and y is not None:
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=20,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        values = np.asarray(result.importances_mean, dtype=float)
        method = "permutation_importance"
    else:
        values = np.zeros(len(feature_names), dtype=float)
        method = "unavailable"

    if values.size != len(feature_names):
        values = np.resize(values, len(feature_names))

    total = float(np.abs(values).sum())
    normalized = np.abs(values) / total if total else np.zeros_like(values)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "normalized_importance": normalized,
                "method": method,
            }
        )
        .sort_values("normalized_importance", ascending=False)
        .reset_index(drop=True)
    )


def dataset_insights(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    outliers = detect_outliers_iqr(df, target_column)
    _, high_corr = correlation_analysis(df, target_column)
    imbalance = class_imbalance_report(df, target_column)
    return {
        "outlier_features": outliers.head(10).to_dict(orient="records"),
        "high_correlation_pairs": high_corr.head(20).to_dict(orient="records") if not high_corr.empty else [],
        "class_imbalance": imbalance,
    }


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[col for col in df.columns if col in ID_COLUMNS], errors="ignore")

