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

from src.feature_engineering import model_feature_importance
from src.preprocessing import build_input_frame, prepare_features_target, split_dataset
from src.utils import DEFAULT_DATA_PATH, REPORTS_DIR, artifact_paths, load_config, load_dataset, load_json, save_json, setup_logging


def _require_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    return shap


def _extract_binary_shap_values(shap_values: Any, class_index: int = 1) -> Any:
    values = getattr(shap_values, "values", shap_values)
    if isinstance(values, list):
        return values[class_index]
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        copied = shap_values[..., class_index]
        return copied
    return shap_values


def build_explainer(model: Any, X_background_scaled: np.ndarray) -> Any:
    shap = _require_shap()
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)
    return shap.Explainer(model.predict_proba, X_background_scaled)


def generate_shap_artifacts(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = REPORTS_DIR,
    sample_index: int = 0,
    max_background_rows: int = 200,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    shap = _require_shap()
    logger = setup_logging("breast_cancer_ai.explainability")
    config = load_config(config_path)
    paths = artifact_paths(config)

    model = joblib.load(paths["model_path"])
    scaler = joblib.load(paths["scaler_path"])
    feature_names = load_json(paths["feature_names_path"])["feature_names"]

    df = load_dataset(data_path)
    X, y, _ = prepare_features_target(df, feature_columns=feature_names)
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["project"]["random_state"]),
    )
    X_background = X_train.sample(min(max_background_rows, len(X_train)), random_state=int(config["project"]["random_state"]))
    X_background_scaled = scaler.transform(X_background)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    explainer = build_explainer(model, X_background_scaled)
    shap_values = explainer(X_test_scaled[: min(100, len(X_test_scaled))])
    shap_values_binary = _extract_binary_shap_values(shap_values)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "shap_summary.png"
    importance_path = output_path / "shap_feature_importance.png"
    waterfall_path = output_path / "shap_waterfall.png"

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_binary,
        X_test_scaled_df.iloc[: min(100, len(X_test_scaled_df))],
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values_binary,
        X_test_scaled_df.iloc[: min(100, len(X_test_scaled_df))],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(importance_path, dpi=180, bbox_inches="tight")
    plt.close()

    safe_index = min(max(sample_index, 0), len(X_test_scaled) - 1)
    single_values = explainer(X_test_scaled[safe_index : safe_index + 1])
    single_values = _extract_binary_shap_values(single_values)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(single_values[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=180, bbox_inches="tight")
    plt.close()

    artifacts = {
        "summary_plot": str(summary_path),
        "waterfall_plot": str(waterfall_path),
        "feature_importance_plot": str(importance_path),
    }
    save_json({"shap_artifacts": artifacts}, output_path / "shap_artifacts.json")
    logger.info("Saved SHAP artifacts to %s", output_path)
    return artifacts


def explain_prediction(payload: dict[str, Any], config_path: str | Path | None = None) -> pd.DataFrame:
    shap = _require_shap()
    config = load_config(config_path)
    paths = artifact_paths(config)
    model = joblib.load(paths["model_path"])
    scaler = joblib.load(paths["scaler_path"])
    feature_names = load_json(paths["feature_names_path"])["feature_names"]

    input_frame = build_input_frame(payload, feature_names)
    input_scaled = scaler.transform(input_frame)
    explainer = build_explainer(model, input_scaled)
    shap_values = _extract_binary_shap_values(explainer(input_scaled))
    values = np.asarray(shap_values.values if hasattr(shap_values, "values") else shap_values).reshape(-1)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "input_value": input_frame.iloc[0].values,
                "shap_value": values,
                "absolute_impact": np.abs(values),
            }
        )
        .sort_values("absolute_impact", ascending=False)
        .reset_index(drop=True)
    )


def fallback_global_importance(config_path: str | Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    paths = artifact_paths(config)
    model = joblib.load(paths["model_path"])
    feature_names = load_json(paths["feature_names_path"])["feature_names"]
    return model_feature_importance(model, feature_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to breast cancer CSV.")
    parser.add_argument("--output-dir", default=str(REPORTS_DIR), help="Directory to store SHAP figures.")
    parser.add_argument("--sample-index", type=int, default=0, help="Held-out sample index for waterfall plot.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path.")
    args = parser.parse_args()
    generate_shap_artifacts(args.data, args.output_dir, args.sample_index, config_path=args.config)


if __name__ == "__main__":
    main()

