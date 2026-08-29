from __future__ import annotations

import pandas as pd

from src.utils import DATA_DIR, load_config, resolve_path


def load_dataset() -> pd.DataFrame:
    config = load_config()
    path = resolve_path(config["data"]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def feature_columns() -> list[str]:
    config = load_config()
    df = load_dataset()
    target = config["data"]["target_column"]
    return [column for column in df.columns if column != target]


def validation_report() -> dict[str, object]:
    df = load_dataset()
    config = load_config()
    target = config["data"]["target_column"]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target,
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {str(key): int(value) for key, value in df[target].value_counts().items()},
        "feature_columns": feature_columns(),
    }

