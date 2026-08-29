from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR, load_config, resolve_zip_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def manifest_path() -> Path:
    return DATA_DIR / "manifest.csv"


def build_manifest() -> pd.DataFrame:
    config = load_config()
    zip_path = resolve_zip_path(config)
    prefix = config["data"]["zip_prefix"]
    class_map = config["data"]["class_map"]
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS or not info.filename.startswith(prefix):
                continue
            parts = info.filename.split("/")
            if len(parts) < 4:
                continue
            source_label = parts[2]
            label = class_map.get(source_label, source_label.replace("_", " ").title())
            rows.append(
                {
                    "zip_member": info.filename,
                    "label": label,
                    "source_label": source_label,
                    "filename": Path(info.filename).name,
                    "dataset": config["project"]["folder"],
                }
            )
    manifest = pd.DataFrame(rows).sort_values(["label", "filename"]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path(), index=False)
    return manifest


def load_manifest(rebuild: bool = False) -> pd.DataFrame:
    path = manifest_path()
    if rebuild or not path.exists():
        return build_manifest()
    return pd.read_csv(path)


def read_zip_image(member: str) -> bytes:
    zip_path = resolve_zip_path(load_config())
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member)


def class_distribution() -> pd.DataFrame:
    manifest = load_manifest()
    counts = manifest["label"].value_counts().rename_axis("label").reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return counts

