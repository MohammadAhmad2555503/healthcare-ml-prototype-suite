from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluate import probability_scores
from src.preprocessing import build_input_frame, dataframe_from_records
from src.utils import (
    artifact_paths,
    clinical_recommendation,
    decode_diagnosis,
    load_config,
    load_json,
    risk_level,
)


class BreastCancerPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        feature_names_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config = load_config(config_path)
        paths = artifact_paths(self.config)
        self.model_path = Path(model_path) if model_path else paths["model_path"]
        self.scaler_path = Path(scaler_path) if scaler_path else paths["scaler_path"]
        self.feature_names_path = Path(feature_names_path) if feature_names_path else paths["feature_names_path"]
        self.metadata_path = Path(metadata_path) if metadata_path else paths["metadata_path"]
        self.model = self._load_artifact(self.model_path, "model")
        self.scaler = self._load_artifact(self.scaler_path, "scaler")
        self.feature_names = load_json(self.feature_names_path)["feature_names"]
        self.metadata = load_json(self.metadata_path, default={})

    @staticmethod
    def _load_artifact(path: Path, label: str) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"{label.title()} artifact not found: {path}")
        return joblib.load(path)

    def predict_dataframe(self, records: pd.DataFrame) -> pd.DataFrame:
        X = records[self.feature_names].apply(pd.to_numeric, errors="raise")
        X_scaled = self.scaler.transform(X)
        predicted_classes = self.model.predict(X_scaled).astype(int)
        malignant_probabilities = probability_scores(self.model, X_scaled)
        benign_probabilities = 1.0 - malignant_probabilities

        results = []
        for label, malignant_probability, benign_probability in zip(
            predicted_classes,
            malignant_probabilities,
            benign_probabilities,
        ):
            level = risk_level(float(malignant_probability), self.config)
            results.append(
                {
                    "prediction_label": decode_diagnosis(int(label)),
                    "prediction": "Malignant" if int(label) == 1 else "Benign",
                    "benign_probability": float(benign_probability),
                    "malignant_probability": float(malignant_probability),
                    "confidence_score": float(max(benign_probability, malignant_probability)),
                    "risk_level": level,
                    "clinical_recommendation": clinical_recommendation(level),
                    "selected_model": self.metadata.get("selected_model", "unknown"),
                }
            )
        return pd.DataFrame(results)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = build_input_frame(payload, self.feature_names)
        result = self.predict_dataframe(frame).iloc[0].to_dict()
        result["input_features"] = {feature: float(frame.loc[0, feature]) for feature in self.feature_names}
        return result

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = dataframe_from_records(records, self.feature_names)
        return self.predict_dataframe(frame).to_dict(orient="records")


def load_predictor(config_path: str | Path | None = None) -> BreastCancerPredictor:
    return BreastCancerPredictor(config_path=config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict breast cancer diagnosis risk from JSON feature input.")
    parser.add_argument("--input-json", required=True, help="JSON object or path to a JSON file with model features.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path.")
    args = parser.parse_args()

    candidate = Path(args.input_json)
    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.input_json)

    predictor = BreastCancerPredictor(config_path=args.config)
    if isinstance(payload, list):
        output = predictor.predict_many(payload)
    else:
        output = predictor.predict(payload)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

