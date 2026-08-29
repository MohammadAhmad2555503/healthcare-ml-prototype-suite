from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import extract_image_features, feature_names
from src.predict import ImageClinicalPredictor
from src.utils import REPORTS_DIR


def global_feature_importance() -> pd.DataFrame:
    path = REPORTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["feature", "importance", "normalized_importance"])


def explain_image(source: bytes | str | Path) -> pd.DataFrame:
    predictor = ImageClinicalPredictor()
    values = extract_image_features(source)
    importance = global_feature_importance().set_index("feature")
    rows = []
    for name, value in zip(feature_names(), values):
        weight = float(importance.loc[name, "normalized_importance"]) if name in importance.index else 0.0
        rows.append({"feature": name, "input_value": float(value), "global_importance": weight})
    return pd.DataFrame(rows).sort_values("global_importance", ascending=False).reset_index(drop=True)

