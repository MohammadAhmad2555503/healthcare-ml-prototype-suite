from __future__ import annotations

import unittest

from src.data import feature_columns, load_dataset, validation_report
from src.predict import TabularClinicalPredictor
from src.utils import artifact_paths, load_config


class TabularPrototypeTests(unittest.TestCase):
    def test_dataset_loads(self) -> None:
        df = load_dataset()
        report = validation_report()
        self.assertEqual(len(df), report["rows"])
        self.assertEqual(report["missing_values_total"], 0)
        self.assertGreater(len(feature_columns()), 0)

    def test_model_artifacts_load(self) -> None:
        paths = artifact_paths(load_config())
        self.assertTrue(paths["model_path"].exists())
        self.assertTrue(paths["metrics_path"].exists())

    def test_prediction_on_sample_patient(self) -> None:
        df = load_dataset()
        payload = df.iloc[0][feature_columns()].to_dict()
        result = TabularClinicalPredictor().predict(payload)
        self.assertTrue(result["prediction"])
        self.assertGreaterEqual(result["heart_disease_probability"], 0.0)
        self.assertLessEqual(result["heart_disease_probability"], 1.0)
        self.assertIn(result["risk_level"], {"Low Risk", "Moderate Risk", "High Risk"})


if __name__ == "__main__":
    unittest.main()

