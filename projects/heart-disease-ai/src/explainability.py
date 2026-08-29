from __future__ import annotations

import pandas as pd

from src.utils import REPORTS_DIR


def global_feature_importance() -> pd.DataFrame:
    path = REPORTS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["feature", "importance", "normalized_importance"])

