from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import pandas as pd

from src.utils import artifact_paths, clinical_recommendation, load_config, load_json, risk_level


class TabularClinicalPredictor:
    def __init__(self) -> None:
        self.config = load_config()
        self.paths = artifact_paths(self.config)
        self.model = joblib.load(self.paths["model_path"])
        self.feature_names = load_json(self.paths["feature_names_path"])["feature_names"]
        self.metadata = load_json(self.paths["metadata_path"], default={})

    def predict(self, payload: dict[str, Any]) -> dict[str, object]:
        missing = [feature for feature in self.feature_names if feature not in payload]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        row = {feature: float(payload[feature]) for feature in self.feature_names}
        frame = pd.DataFrame([row], columns=self.feature_names)
        probability = float(self.model.predict_proba(frame)[0, 1])
        prediction_value = int(probability >= 0.5)
        label = self.config["data"]["class_map"][str(prediction_value)]
        level = risk_level(probability)
        return {
            "prediction": label,
            "heart_disease_probability": probability,
            "confidence_score": max(probability, 1.0 - probability),
            "risk_level": level,
            "clinical_recommendation": clinical_recommendation(level),
            "selected_model": self.metadata.get("selected_model", "unknown"),
            "input_features": row,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    predictor = TabularClinicalPredictor()
    print(json.dumps(predictor.predict(json.loads(args.input_json)), indent=2))


if __name__ == "__main__":
    main()

