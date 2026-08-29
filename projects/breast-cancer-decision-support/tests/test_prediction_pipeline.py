from __future__ import annotations

import unittest

from src.feature_engineering import clean_breast_cancer_dataframe
from src.predict import BreastCancerPredictor
from src.utils import DEFAULT_DATA_PATH, get_feature_columns, load_dataset


class PredictionPipelineTests(unittest.TestCase):
    def test_single_prediction_returns_clinical_fields(self) -> None:
        df = clean_breast_cancer_dataframe(load_dataset(DEFAULT_DATA_PATH))
        features = get_feature_columns(df)
        payload = df.iloc[0][features].to_dict()

        predictor = BreastCancerPredictor()
        result = predictor.predict(payload)

        self.assertIn(result["prediction"], {"Benign", "Malignant"})
        self.assertGreaterEqual(result["malignant_probability"], 0.0)
        self.assertLessEqual(result["malignant_probability"], 1.0)
        self.assertIn(result["risk_level"], {"Low Risk", "Moderate Risk", "High Risk"})
        self.assertIn("clinical_recommendation", result)

    def test_missing_feature_raises_error(self) -> None:
        predictor = BreastCancerPredictor()
        with self.assertRaises(ValueError):
            predictor.predict({"radius_mean": 1.0})


if __name__ == "__main__":
    unittest.main()

