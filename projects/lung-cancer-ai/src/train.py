from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.data import load_manifest, read_zip_image
from src.features import extract_image_features, feature_names
from src.utils import MODEL_DIR, REPORTS_DIR, artifact_paths, ensure_directories, load_config, save_json, setup_logging, utc_now


def _sample_manifest(manifest: pd.DataFrame, max_per_class: int | None, random_state: int) -> pd.DataFrame:
    if not max_per_class:
        return manifest.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    groups = []
    for _, group in manifest.groupby("label"):
        groups.append(group.sample(min(len(group), max_per_class), random_state=random_state))
    return pd.concat(groups, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_feature_matrix(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for record in manifest.to_dict(orient="records"):
        image_bytes = read_zip_image(record["zip_member"])
        rows.append(extract_image_features(image_bytes))
        labels.append(record["label"])
    return np.vstack(rows), np.asarray(labels)


def _candidate_models(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"model__C": [0.1, 1.0, 3.0]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [None, 8, 14],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "Support Vector Machine": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"model__C": [0.5, 1.0, 3.0], "model__kernel": ["rbf", "linear"]},
        ),
    }


def _probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.vstack([1.0 - p1, p1]).T
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if y_prob.shape[1] == 2:
            payload["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            payload["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        payload["roc_auc"] = 0.0
    return payload


def _save_plots(manifest: pd.DataFrame, comparison: pd.DataFrame, labels: list[str], matrix: list[list[int]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = manifest["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#1167b1")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    sorted_comparison = comparison.sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_comparison["model"], sorted_comparison["f1_macro"], color="#0f766e")
    ax.set_xlim(0, 1)
    ax.set_title("Model Comparison by F1 Macro")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def _feature_importance(model: Pipeline, names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
    else:
        values = np.zeros(len(names), dtype=float)
    total = float(np.abs(values).sum())
    normalized = np.abs(values) / total if total else np.zeros_like(values)
    return pd.DataFrame({"feature": names, "importance": values, "normalized_importance": normalized}).sort_values(
        "normalized_importance", ascending=False
    )


def train(max_per_class: int | None = None) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    ensure_directories()
    logger = setup_logging("prototype.train")
    config = load_config()
    paths = artifact_paths(config)
    random_state = int(config["training"]["random_state"])
    max_per_class = max_per_class if max_per_class is not None else config["training"].get("max_samples_per_class")

    manifest = _sample_manifest(load_manifest(), max_per_class, random_state)
    manifest.to_csv(REPORTS_DIR / "training_manifest.csv", index=False)
    X, labels = build_feature_matrix(manifest)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    class_names = encoder.classes_.tolist()
    test_size = float(config["training"]["test_size"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    cv_folds = min(int(config["training"]["cv_folds"]), int(np.bincount(y_train).min()))
    if cv_folds < 2:
        raise ValueError("Not enough samples per class for cross-validation.")

    records = []
    fitted: dict[str, Pipeline] = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for name, (estimator, params) in _candidate_models(random_state).items():
        logger.info("Training %s", name)
        search = GridSearchCV(estimator, params, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred = model.predict(X_test)
        prob = _probabilities(model, X_test)
        record = {"model": name, "best_cv_score": float(search.best_score_), "best_params": search.best_params_}
        record.update(_metrics(y_test, pred, prob))
        records.append(record)
        fitted[name] = model

    comparison = pd.DataFrame(records).sort_values(["f1_macro", "recall_macro", "accuracy"], ascending=False).reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])
    selected = fitted[selected_name]
    y_pred = selected.predict(X_test)
    y_prob = _probabilities(selected, X_test)
    metrics = _metrics(y_test, y_pred, y_prob)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    feature_importance = _feature_importance(selected, feature_names())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, paths["model_path"])
    joblib.dump(encoder, paths["label_encoder_path"])
    save_json({"feature_names": feature_names()}, paths["feature_names_path"])
    metadata = {
        "project": config["project"],
        "generated_at": utc_now(),
        "selected_model": selected_name,
        "classes": class_names,
        "feature_count": len(feature_names()),
        "training_rows": int(len(manifest)),
        "test_rows": int(len(y_test)),
        "test_metrics": metrics,
        "clinical": config["clinical"],
    }
    save_json(metadata, paths["metadata_path"])

    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    report = {
        "selected_model": selected_name,
        "generated_at": utc_now(),
        "test_metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": matrix,
        "classes": class_names,
        "comparison": comparison.to_dict(orient="records"),
        "feature_importance": feature_importance.head(40).to_dict(orient="records"),
    }
    save_json(report, paths["metrics_path"])
    _save_plots(manifest, comparison, class_names, matrix)
    logger.info("Selected %s with metrics %s", selected_name, metrics)
    return {"selected_model": selected_name, "metrics": metrics, "classes": class_names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()
    print(train(args.max_per_class))


if __name__ == "__main__":
    main()

