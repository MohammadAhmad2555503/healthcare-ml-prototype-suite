from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import clean_breast_cancer_dataframe
from src.utils import TARGET_COLUMN, encode_diagnosis, get_feature_columns


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cleaned = clean_breast_cancer_dataframe(df, target_column)
    features = feature_columns or get_feature_columns(cleaned, target_column)
    X = cleaned[features].apply(pd.to_numeric, errors="coerce")
    y = encode_diagnosis(cleaned[target_column])
    return X, y, features


def build_scaler() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_transform_scaler(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Pipeline | None = None,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    active_scaler = scaler or build_scaler()
    X_train_scaled = active_scaler.fit_transform(X_train)
    X_test_scaled = active_scaler.transform(X_test)
    return active_scaler, X_train_scaled, X_test_scaled


def save_scaler(scaler: Pipeline, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)


def load_scaler(path: str | Path) -> Pipeline:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {input_path}")
    return joblib.load(input_path)


def validate_prediction_input(payload: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
    missing = [feature for feature in feature_names if feature not in payload]
    if missing:
        raise ValueError(f"Missing required feature values: {missing}")

    cleaned: dict[str, float] = {}
    invalid: list[str] = []
    for feature in feature_names:
        try:
            value = float(payload[feature])
        except (TypeError, ValueError):
            invalid.append(feature)
            continue
        if not np.isfinite(value):
            invalid.append(feature)
        cleaned[feature] = value

    if invalid:
        raise ValueError(f"Invalid numeric feature values: {invalid}")
    return cleaned


def build_input_frame(payload: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    cleaned = validate_prediction_input(payload, feature_names)
    return pd.DataFrame([[cleaned[feature] for feature in feature_names]], columns=feature_names)


def dataframe_from_records(records: list[dict[str, Any]], feature_names: list[str]) -> pd.DataFrame:
    rows = [validate_prediction_input(record, feature_names) for record in records]
    return pd.DataFrame(rows, columns=feature_names)

