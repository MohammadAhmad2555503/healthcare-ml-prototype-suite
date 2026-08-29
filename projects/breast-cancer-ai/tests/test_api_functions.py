from __future__ import annotations

import unittest

from src.preprocessing import build_input_frame
from src.utils import clinical_recommendation, risk_level


class ApiFunctionTests(unittest.TestCase):
    def test_risk_thresholds(self) -> None:
        self.assertEqual(risk_level(0.10), "Low Risk")
        self.assertEqual(risk_level(0.50), "Moderate Risk")
        self.assertEqual(risk_level(0.90), "High Risk")

    def test_clinical_recommendation_returns_text(self) -> None:
        text = clinical_recommendation("High Risk")
        self.assertIn("clinical", text.lower())

    def test_build_input_frame_validates_features(self) -> None:
        frame = build_input_frame({"a": "1.2", "b": 3}, ["a", "b"])
        self.assertEqual(frame.shape, (1, 2))
        self.assertAlmostEqual(frame.loc[0, "a"], 1.2)


if __name__ == "__main__":
    unittest.main()

