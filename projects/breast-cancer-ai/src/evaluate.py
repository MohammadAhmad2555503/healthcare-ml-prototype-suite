from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import prepare_features_target, split_dataset
from src.utils import (
    DEFAULT_DATA_PATH,
    REPORTS_DIR,
    artifact_paths,
    load_config,
    load_dataset,
    load_json,
    save_json,
    setup_logging,
)


def probability_scores(model: Any, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    predictions = np.asarray(model.predict(X), dtype=float)
    return predictions


def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _save_confusion_matrix(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_roc_curve(y_true: np.ndarray | pd.Series, y_prob: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1167b1", linewidth=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return fpr, tpr, thresholds


def _save_precision_recall_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#0f766e", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return precision, recall, thresholds


def generate_evaluation_artifacts(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path = REPORTS_DIR,
    prefix: str = "",
) -> dict[str, Any]:
    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""

    confusion_path = reports_dir / f"{stem}confusion_matrix.png"
    roc_path = reports_dir / f"{stem}roc_curve.png"
    pr_path = reports_dir / f"{stem}precision_recall_curve.png"

    _save_confusion_matrix(y_true, y_pred, confusion_path)
    fpr, tpr, roc_thresholds = _save_roc_curve(y_true, y_prob, roc_path)
    precision, recall, pr_thresholds = _save_precision_recall_curve(y_true, y_prob, pr_path)

    matrix = confusion_matrix(y_true, y_pred)
    return {
        "metrics": calculate_metrics(y_true, y_pred, y_prob),
        "classification_report": classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True),
        "confusion_matrix": matrix.tolist(),
        "curves": {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            },
            "precision_recall": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
        },
        "plots": {
            "confusion_matrix": str(confusion_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
        },
    }


def evaluate_saved_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    logger = setup_logging("breast_cancer_ai.evaluate")
    config = load_config(config_path)
    paths = artifact_paths(config)

    model = joblib.load(paths["model_path"])
    scaler = joblib.load(paths["scaler_path"])
    feature_names = load_json(paths["feature_names_path"])["feature_names"]

    df = load_dataset(data_path)
    X, y, _ = prepare_features_target(df, feature_columns=feature_names)
    _, X_test, _, y_test = split_dataset(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["project"]["random_state"]),
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = probability_scores(model, X_test_scaled)
    report = generate_evaluation_artifacts(y_test, y_pred, y_prob, REPORTS_DIR)
    save_json(report, paths["metrics_path"])
    logger.info("Saved evaluation metrics to %s", paths["metrics_path"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained breast cancer model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to breast cancer CSV.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path.")
    args = parser.parse_args()
    evaluate_saved_model(args.data, args.config)


if __name__ == "__main__":
    main()

