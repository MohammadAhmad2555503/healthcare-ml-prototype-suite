from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_DATA_PATH = DATA_DIR / "breast_cancer.csv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

TARGET_COLUMN = "diagnosis"
ID_COLUMNS = {"id", "ID", "patient_id", "Patient_ID"}
TARGET_MAPPING = {"B": 0, "BENIGN": 0, "M": 1, "MALIGNANT": 1}
INVERSE_TARGET_MAPPING = {0: "B", 1: "M"}

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "Breast Cancer Prediction and Clinical Decision Support System",
        "version": "1.0.0",
        "random_state": 42,
    },
    "data": {
        "path": str(DEFAULT_DATA_PATH),
        "target_column": TARGET_COLUMN,
        "drop_columns": ["id"],
        "test_size": 0.2,
        "validation_size": 0.0,
    },
    "training": {
        "cv_folds": 5,
        "scoring": "roc_auc",
        "n_iter_random_search": 12,
        "refit_metric": "roc_auc",
        "risk_thresholds": {
            "low": 0.35,
            "moderate": 0.65,
        },
    },
    "artifacts": {
        "model_path": str(MODEL_DIR / "model.pkl"),
        "scaler_path": str(MODEL_DIR / "scaler.pkl"),
        "metadata_path": str(MODEL_DIR / "model_metadata.json"),
        "feature_names_path": str(MODEL_DIR / "feature_names.json"),
        "metrics_path": str(REPORTS_DIR / "metrics.json"),
    },
}


def project_path(*parts: str | Path) -> Path:
    """Return an absolute path inside the project root."""
    return PROJECT_ROOT.joinpath(*map(Path, parts)).resolve()


def ensure_directories() -> None:
    for path in (DATA_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "breast_cancer_ai", level: int = logging.INFO) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    log_file = LOGS_DIR / "application.log"
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file for handler in logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = deepcopy(DEFAULT_CONFIG)

    if not path.exists():
        return config

    try:
        import yaml
    except ImportError:
        return config

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _deep_merge(config, loaded)


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    drop_cols: list[str] = []
    for col in cleaned.columns:
        normalized = str(col).strip()
        if normalized == "" or normalized.startswith("Unnamed"):
            drop_cols.append(col)
        elif cleaned[col].isna().all():
            drop_cols.append(col)
    return cleaned.drop(columns=drop_cols, errors="ignore")


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = [str(col).strip() for col in df.columns]
    df = _drop_empty_columns(df)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    return df


def get_feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    excluded = {target_column, *ID_COLUMNS}
    numeric_columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def validate_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    feature_columns = get_feature_columns(df, target_column)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    target_values = df[target_column].astype(str).str.strip().str.upper()
    invalid_targets = sorted(set(target_values) - set(TARGET_MAPPING))
    if invalid_targets:
        raise ValueError(f"Invalid target labels found: {invalid_targets}. Expected M/B or Malignant/Benign.")

    missing_by_column = df.isna().sum().astype(int).to_dict()
    duplicate_count = int(df.duplicated().sum())
    numeric_summary = df[feature_columns].apply(pd.to_numeric, errors="coerce").describe().to_dict()
    target_counts = target_values.value_counts().to_dict()

    return {
        "generated_at": utc_now(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": missing_by_column,
        "duplicate_rows": duplicate_count,
        "target_distribution": {str(key): int(value) for key, value in target_counts.items()},
        "numeric_summary": numeric_summary,
    }


def encode_diagnosis(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    encoded = normalized.map(TARGET_MAPPING)
    if encoded.isna().any():
        bad_values = sorted(normalized[encoded.isna()].unique().tolist())
        raise ValueError(f"Unable to encode diagnosis labels: {bad_values}")
    return encoded.astype(int)


def decode_diagnosis(value: int | np.integer) -> str:
    return INVERSE_TARGET_MAPPING[int(value)]


def risk_level(malignant_probability: float, config: dict[str, Any] | None = None) -> str:
    cfg = config or load_config()
    thresholds = cfg["training"]["risk_thresholds"]
    if malignant_probability < float(thresholds["low"]):
        return "Low Risk"
    if malignant_probability < float(thresholds["moderate"]):
        return "Moderate Risk"
    return "High Risk"


def clinical_recommendation(level: str) -> str:
    recommendations = {
        "Low Risk": "Prediction is consistent with benign findings. Continue routine clinical review and confirm with physician judgment.",
        "Moderate Risk": "Prediction is uncertain or elevated. Recommend specialist review and correlation with imaging, pathology, and patient history.",
        "High Risk": "Prediction is consistent with malignant findings. Expedite clinical review, diagnostic confirmation, and care pathway escalation.",
    }
    return recommendations.get(level, "Clinical review required.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def artifact_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = config or load_config()
    paths: dict[str, Path] = {}
    for key, value in cfg["artifacts"].items():
        path = Path(value)
        paths[key] = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return paths


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

