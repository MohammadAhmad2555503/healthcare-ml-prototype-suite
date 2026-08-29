from __future__ import annotations

import argparse
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from src.evaluate import calculate_metrics, generate_evaluation_artifacts, probability_scores
from src.feature_engineering import (
    class_imbalance_report,
    correlation_analysis,
    dataset_insights,
    detect_outliers_iqr,
    model_feature_importance,
)
from src.preprocessing import fit_transform_scaler, prepare_features_target, save_scaler, split_dataset
from src.utils import (
    DEFAULT_DATA_PATH,
    MODEL_DIR,
    REPORTS_DIR,
    artifact_paths,
    ensure_directories,
    load_config,
    load_dataset,
    save_json,
    setup_logging,
    utc_now,
    validate_dataset,
)


def _optional_model_registry(random_state: int) -> tuple[OrderedDict[str, Any], dict[str, str]]:
    models: OrderedDict[str, Any] = OrderedDict()
    skipped: dict[str, str] = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    models["Support Vector Machine"] = SVC(
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception as exc:  # pragma: no cover - depends on optional package availability
        skipped["XGBoost"] = str(exc)

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:  # pragma: no cover - depends on optional package availability
        skipped["LightGBM"] = str(exc)

    try:
        from catboost import CatBoostClassifier

        models["CatBoost"] = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:  # pragma: no cover - depends on optional package availability
        skipped["CatBoost"] = str(exc)

    return models, skipped


def _search_space() -> dict[str, dict[str, Any]]:
    return {
        "Logistic Regression": {
            "method": "grid",
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "method": "random",
            "params": {
                "n_estimators": [150, 250, 400, 600],
                "max_depth": [None, 3, 5, 8, 12],
                "min_samples_split": [2, 4, 8, 12],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        },
        "Support Vector Machine": {
            "method": "grid",
            "params": {
                "C": [0.1, 1.0, 5.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
        "XGBoost": {
            "method": "random",
            "params": {
                "n_estimators": [100, 200, 350, 500],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.75, 0.9, 1.0],
                "colsample_bytree": [0.75, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
        },
        "LightGBM": {
            "method": "random",
            "params": {
                "n_estimators": [100, 200, 350, 500],
                "num_leaves": [7, 15, 31, 63],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "subsample": [0.75, 0.9, 1.0],
                "colsample_bytree": [0.75, 0.9, 1.0],
            },
        },
        "CatBoost": {
            "method": "random",
            "params": {
                "iterations": [150, 250, 400, 600],
                "depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
            },
        },
    }


def _build_search(
    model_name: str,
    estimator: Any,
    cv: StratifiedKFold,
    scoring: str,
    random_state: int,
    n_iter: int,
) -> GridSearchCV | RandomizedSearchCV:
    spec = _search_space()[model_name]
    if spec["method"] == "grid":
        return GridSearchCV(
            estimator=estimator,
            param_grid=spec["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=spec["params"],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )


def _save_class_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["diagnosis"].value_counts().reindex(["B", "M"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Benign", "Malignant"], counts.values, color=["#0f766e", "#b42318"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Patients")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 3, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_correlation_heatmap(correlation: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(correlation.columns, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_feature_importance_plot(importance: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top = importance.head(top_n).sort_values("normalized_importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["normalized_importance"], color="#1167b1")
    ax.set_xlabel("Normalized Importance")
    ax.set_title("Top Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_model_comparison_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    ordered = comparison.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["model"], ordered["roc_auc"], color="#155e75")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Model Comparison")
    for idx, value in enumerate(ordered["roc_auc"]):
        ax.text(min(value + 0.01, 0.98), idx, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    config_path: str | Path | None = None,
    output_dir: str | Path = MODEL_DIR,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_directories()
    logger = setup_logging("breast_cancer_ai.train")
    config = load_config(config_path)
    random_state = int(config["project"]["random_state"])
    test_size = float(config["data"]["test_size"])
    cv_folds = int(config["training"]["cv_folds"])
    scoring = str(config["training"]["scoring"])
    n_iter = int(config["training"]["n_iter_random_search"])

    model_dir = Path(output_dir)
    report_dir = Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", data_path)
    df = load_dataset(data_path)
    validation = validate_dataset(df)
    X, y, feature_names = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size, random_state=random_state)
    scaler, X_train_scaled, X_test_scaled = fit_transform_scaler(X_train, X_test)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models, skipped_models = _optional_model_registry(random_state)
    candidates: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    logger.info("Training %d available model families", len(models))
    for model_name, estimator in models.items():
        logger.info("Tuning %s", model_name)
        search = _build_search(model_name, estimator, cv, scoring, random_state, n_iter)
        search.fit(X_train_scaled, y_train)
        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test_scaled)
        y_prob = probability_scores(best_estimator, X_test_scaled)
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        cv_scores = cross_val_score(best_estimator, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        record = {
            "model": model_name,
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            **metrics,
        }
        candidates.append(record)
        fitted_models[model_name] = best_estimator
        logger.info("%s ROC-AUC %.4f | Recall %.4f | F1 %.4f", model_name, metrics["roc_auc"], metrics["recall"], metrics["f1"])

    comparison = pd.DataFrame(candidates)
    if comparison.empty:
        raise RuntimeError("No candidate models were available for training.")

    comparison = comparison.sort_values(["roc_auc", "recall", "f1"], ascending=False).reset_index(drop=True)
    selected_model_name = str(comparison.loc[0, "model"])
    selected_model = fitted_models[selected_model_name]
    logger.info("Selected best model: %s", selected_model_name)

    y_pred = selected_model.predict(X_test_scaled)
    y_prob = probability_scores(selected_model, X_test_scaled)
    evaluation_report = generate_evaluation_artifacts(y_test, y_pred, y_prob, report_dir)

    importance = model_feature_importance(
        selected_model,
        feature_names,
        X=X_test_scaled,
        y=y_test,
        scoring=scoring,
        random_state=random_state,
    )

    correlation_matrix, high_corr_pairs = correlation_analysis(df)
    outliers = detect_outliers_iqr(df)
    imbalance = class_imbalance_report(df)
    insights = dataset_insights(df)

    comparison_path = report_dir / "model_comparison.csv"
    importance_path = report_dir / "feature_importance.csv"
    validation_path = report_dir / "data_validation_report.json"
    outlier_path = report_dir / "outlier_report.csv"
    correlation_path = report_dir / "correlation_matrix.csv"
    high_corr_path = report_dir / "high_correlation_pairs.csv"

    comparison.to_csv(comparison_path, index=False)
    importance.to_csv(importance_path, index=False)
    outliers.to_csv(outlier_path, index=False)
    correlation_matrix.to_csv(correlation_path)
    high_corr_pairs.to_csv(high_corr_path, index=False)
    save_json(
        {
            "validation": validation,
            "class_imbalance": imbalance,
            "dataset_insights": insights,
            "skipped_models": skipped_models,
        },
        validation_path,
    )

    _save_class_distribution_plot(df, report_dir / "class_distribution.png")
    _save_correlation_heatmap(correlation_matrix, report_dir / "correlation_heatmap.png")
    _save_feature_importance_plot(importance, report_dir / "feature_importance.png")
    _save_model_comparison_plot(comparison, report_dir / "model_comparison.png")

    paths = artifact_paths(config)
    model_path = Path(paths["model_path"])
    scaler_path = Path(paths["scaler_path"])
    metadata_path = Path(paths["metadata_path"])
    feature_names_path = Path(paths["feature_names_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, model_path)
    save_scaler(scaler, scaler_path)
    save_json({"feature_names": feature_names}, feature_names_path)

    metadata = {
        "project": config["project"],
        "generated_at": utc_now(),
        "selected_model": selected_model_name,
        "target_column": "diagnosis",
        "target_mapping": {"B": 0, "M": 1},
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "scoring": scoring,
        "best_params": comparison.loc[0, "best_params"],
        "test_metrics": evaluation_report["metrics"],
        "skipped_models": skipped_models,
    }
    save_json(metadata, metadata_path)

    metrics_payload = {
        "selected_model": selected_model_name,
        "generated_at": utc_now(),
        "test_metrics": evaluation_report["metrics"],
        "classification_report": evaluation_report["classification_report"],
        "confusion_matrix": evaluation_report["confusion_matrix"],
        "curves": evaluation_report["curves"],
        "plots": evaluation_report["plots"],
        "comparison": comparison.to_dict(orient="records"),
        "feature_importance": importance.head(30).to_dict(orient="records"),
        "skipped_models": skipped_models,
    }
    save_json(metrics_payload, paths["metrics_path"])

    logger.info("Saved model to %s", model_path)
    logger.info("Saved scaler to %s", scaler_path)
    logger.info("Saved metrics to %s", paths["metrics_path"])
    return {
        "selected_model": selected_model_name,
        "metrics": evaluation_report["metrics"],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "comparison_path": str(comparison_path),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the best breast cancer prediction model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to breast cancer CSV.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path.")
    parser.add_argument("--output-dir", default=str(MODEL_DIR), help="Directory for model artifacts.")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR), help="Directory for reports and plots.")
    args = parser.parse_args()
    result = train_model(args.data, args.config, args.output_dir, args.reports_dir)
    print(result)


if __name__ == "__main__":
    main()

