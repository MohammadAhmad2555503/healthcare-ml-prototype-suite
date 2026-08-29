from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import BinaryIO

import joblib

from src.features import extract_image_features
from src.utils import artifact_paths, clinical_recommendation, load_config, load_json, risk_level


class ImageClinicalPredictor:
    def __init__(self) -> None:
        self.config = load_config()
        self.paths = artifact_paths(self.config)
        self.model = joblib.load(self.paths["model_path"])
        self.encoder = joblib.load(self.paths["label_encoder_path"])
        self.metadata = load_json(self.paths["metadata_path"], default={})

    def predict_image(self, source: bytes | str | Path | BinaryIO) -> dict[str, object]:
        features = extract_image_features(source).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        index = int(probabilities.argmax())
        label = str(self.encoder.inverse_transform([index])[0])
        confidence = float(probabilities[index])
        level = risk_level(label, confidence, self.config)
        return {
            "prediction": label,
            "confidence_score": confidence,
            "risk_level": level,
            "clinical_recommendation": clinical_recommendation(level),
            "selected_model": self.metadata.get("selected_model", "unknown"),
            "probabilities": {
                str(label_name): float(prob)
                for label_name, prob in zip(self.encoder.classes_.tolist(), probabilities.tolist())
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    predictor = ImageClinicalPredictor()
    print(json.dumps(predictor.predict_image(args.image), indent=2))


if __name__ == "__main__":
    main()

