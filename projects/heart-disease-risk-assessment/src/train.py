from __future__ import annotations

import argparse
import warnings

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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import feature_columns, load_dataset, validation_report
from src.utils import MODEL_DIR, REPORTS_DIR, artifact_paths, ensure_directories, load_config, save_json, setup_logging, utc_now


def _models(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {"model__C": [0.05, 0.1, 1.0, 3.0]},
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _feature_importance(model: Pipeline, names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    else:
        values = np.zeros(len(names), dtype=float)
    total = float(np.abs(values).sum())
    normalized = np.abs(values) / total if total else np.zeros_like(values)
    return pd.DataFrame({"feature": names, "importance": values, "normalized_importance": normalized}).sort_values(
        "normalized_importance", ascending=False
    )


def _save_plots(df: pd.DataFrame, target: str, comparison: pd.DataFrame, matrix: list[list[int]], labels: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([labels[int(i)] for i in counts.index], counts.values, color=["#0f766e", "#b42318"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    sorted_comparison = comparison.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sorted_comparison["model"], sorted_comparison["roc_auc"], color="#1167b1")
    ax.set_xlim(0, 1)
    ax.set_title("Model Comparison by ROC-AUC")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
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


def train() -> dict[str, object]:
    warnings.filterwarnings("ignore")
    ensure_directories()
    logger = setup_logging("prototype.train")
    config = load_config()
    paths = artifact_paths(config)
    random_state = int(config["training"]["random_state"])
    df = load_dataset()
    target = config["data"]["target_column"]
    features = feature_columns()
    X = df[features]
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(config["training"]["test_size"]), random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=int(config["training"]["cv_folds"]), shuffle=True, random_state=random_state)
    records = []
    fitted = {}
    for name, (estimator, params) in _models(random_state).items():
        logger.info("Training %s", name)
        search = GridSearchCV(estimator, params, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        record = {"model": name, "best_cv_score": float(search.best_score_), "best_params": search.best_params_}
        record.update(_metrics(y_test, pred, prob))
        records.append(record)
        fitted[name] = model

    comparison = pd.DataFrame(records).sort_values(["roc_auc", "recall", "f1"], ascending=False).reset_index(drop=True)
    selected_name = str(comparison.loc[0, "model"])
    selected = fitted[selected_name]
    y_pred = selected.predict(X_test)
    y_prob = selected.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_prob)
    labels = [config["data"]["class_map"][str(value)] for value in sorted(y.unique())]
    matrix = confusion_matrix(y_test, y_pred).tolist()
    importance = _feature_importance(selected, features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, paths["model_path"])
    save_json({"feature_names": features}, paths["feature_names_path"])
    metadata = {
        "project": config["project"],
        "generated_at": utc_now(),
        "selected_model": selected_name,
        "target_column": target,
        "classes": labels,
        "feature_count": len(features),
        "test_metrics": metrics,
        "clinical": config["clinical"],
    }
    save_json(metadata, paths["metadata_path"])
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    save_json(validation_report(), REPORTS_DIR / "data_validation_report.json")
    report = {
        "selected_model": selected_name,
        "generated_at": utc_now(),
        "test_metrics": metrics,
        "classification_report": classification_report(y_test, y_pred, target_names=labels, output_dict=True),
        "confusion_matrix": matrix,
        "classes": labels,
        "comparison": comparison.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
    }
    save_json(report, paths["metrics_path"])
    _save_plots(df, target, comparison, matrix, labels)
    logger.info("Selected %s with metrics %s", selected_name, metrics)
    return {"selected_model": selected_name, "metrics": metrics, "classes": labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(train())


if __name__ == "__main__":
    main()

