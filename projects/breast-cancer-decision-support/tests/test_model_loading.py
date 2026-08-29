from __future__ import annotations

import unittest

from src.predict import BreastCancerPredictor
from src.utils import artifact_paths, load_config, load_json


class ModelLoadingTests(unittest.TestCase):
    def test_model_artifacts_exist_and_load(self) -> None:
        paths = artifact_paths(load_config())
        for key in ("model_path", "scaler_path", "feature_names_path", "metadata_path"):
            self.assertTrue(paths[key].exists(), f"Missing artifact: {paths[key]}")

        predictor = BreastCancerPredictor()
        self.assertGreater(len(predictor.feature_names), 0)
        self.assertTrue(hasattr(predictor.model, "predict"))
        self.assertTrue(hasattr(predictor.scaler, "transform"))

    def test_metadata_contains_selected_model(self) -> None:
        paths = artifact_paths(load_config())
        metadata = load_json(paths["metadata_path"])
        self.assertIn("selected_model", metadata)
        self.assertIn("test_metrics", metadata)


if __name__ == "__main__":
    unittest.main()

