from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_PATH = PROJECT_ROOT / "config.json"


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(LOGS_DIR / "application.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_zip_path(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    candidates = [
        os.getenv("HEALTH_DATASET_ZIP"),
        cfg["data"].get("zip_path"),
        cfg["data"].get("source_zip_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "Dataset zip not found. Copy datasets.zip to data/datasets.zip or set HEALTH_DATASET_ZIP."
    )


def artifact_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = config or load_config()
    return {key: resolve_path(value) for key, value in cfg["artifacts"].items()}


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(input_path)
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def risk_level(label: str, probability: float, config: dict[str, Any] | None = None) -> str:
    cfg = config or load_config()
    positive_labels = set(cfg["clinical"].get("positive_labels", []))
    if label not in positive_labels and probability >= 0.70:
        return "Low Risk"
    if label in positive_labels and probability >= 0.80:
        return "High Risk"
    if label in positive_labels and probability >= 0.50:
        return "Moderate Risk"
    return "Low Risk"


def clinical_recommendation(level: str) -> str:
    if level == "High Risk":
        return "Escalate for urgent specialist review and correlate with clinical history, imaging protocol, and confirmatory diagnostics."
    if level == "Moderate Risk":
        return "Recommend clinician review, repeat-quality assessment if needed, and correlation with patient context."
    return "Continue routine clinical workflow and confirm the result with qualified clinical judgment."

