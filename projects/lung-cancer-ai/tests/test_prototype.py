from __future__ import annotations

import unittest

from src.data import load_manifest, read_zip_image
from src.predict import ImageClinicalPredictor
from src.utils import artifact_paths, load_config


class ImagePrototypeTests(unittest.TestCase):
    def test_manifest_has_labeled_images(self) -> None:
        manifest = load_manifest()
        self.assertFalse(manifest.empty)
        self.assertTrue({"zip_member", "label", "source_label"}.issubset(manifest.columns))
        self.assertGreaterEqual(manifest["label"].nunique(), 2)

    def test_model_artifacts_load(self) -> None:
        paths = artifact_paths(load_config())
        self.assertTrue(paths["model_path"].exists())
        self.assertTrue(paths["label_encoder_path"].exists())
        self.assertTrue(paths["metrics_path"].exists())

    def test_prediction_on_sample_image(self) -> None:
        manifest = load_manifest()
        image_bytes = read_zip_image(manifest.iloc[0]["zip_member"])
        result = ImageClinicalPredictor().predict_image(image_bytes)
        self.assertTrue(result["prediction"])
        self.assertGreaterEqual(result["confidence_score"], 0.0)
        self.assertLessEqual(result["confidence_score"], 1.0)
        self.assertIn(result["risk_level"], {"Low Risk", "Moderate Risk", "High Risk"})


if __name__ == "__main__":
    unittest.main()

