from __future__ import annotations

import unittest

from src.feature_engineering import class_imbalance_report, clean_breast_cancer_dataframe
from src.utils import DEFAULT_DATA_PATH, TARGET_COLUMN, get_feature_columns, load_dataset, validate_dataset


class DataLoadingTests(unittest.TestCase):
    def test_dataset_loads_and_validates(self) -> None:
        df = load_dataset(DEFAULT_DATA_PATH)
        report = validate_dataset(df)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertEqual(report["rows"], len(df))
        self.assertGreater(report["feature_count"], 0)
        self.assertEqual(report["missing_values_total"], 0)

    def test_cleaning_removes_empty_columns_and_preserves_features(self) -> None:
        df = clean_breast_cancer_dataframe(load_dataset(DEFAULT_DATA_PATH))
        feature_columns = get_feature_columns(df)
        self.assertNotIn("", df.columns)
        self.assertNotIn("id", feature_columns)
        self.assertGreaterEqual(len(feature_columns), 30)

    def test_class_imbalance_report(self) -> None:
        df = clean_breast_cancer_dataframe(load_dataset(DEFAULT_DATA_PATH))
        report = class_imbalance_report(df)
        self.assertEqual(report["total"], len(df))
        self.assertGreater(report["malignant"], 0)
        self.assertGreater(report["benign"], 0)


if __name__ == "__main__":
    unittest.main()

