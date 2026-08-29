from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (96, 96)
THUMBNAIL_SIZE = (16, 16)


def _open_image(source: bytes | str | Path | BinaryIO) -> Image.Image:
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def feature_names() -> list[str]:
    names: list[str] = []
    for channel in ("red", "green", "blue"):
        for stat in ("mean", "std", "min", "max", "q25", "median", "q75"):
            names.append(f"{channel}_{stat}")
    for stat in ("gray_mean", "gray_std", "gray_min", "gray_max", "gray_entropy", "edge_mean", "edge_std", "edge_q90"):
        names.append(stat)
    for idx in range(16):
        names.append(f"gray_hist_{idx:02d}")
    for channel in ("red", "green", "blue"):
        for idx in range(8):
            names.append(f"{channel}_hist_{idx:02d}")
    for idx in range(THUMBNAIL_SIZE[0] * THUMBNAIL_SIZE[1]):
        names.append(f"thumb_pixel_{idx:03d}")
    return names


def extract_image_features(source: bytes | str | Path | BinaryIO) -> np.ndarray:
    image = ImageOps.exif_transpose(_open_image(source)).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    gray_img = resized.convert("L")
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0
    values: list[float] = []

    for channel_idx in range(3):
        channel = arr[:, :, channel_idx].reshape(-1)
        values.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(channel.min()),
                float(channel.max()),
                float(np.quantile(channel, 0.25)),
                float(np.quantile(channel, 0.50)),
                float(np.quantile(channel, 0.75)),
            ]
        )

    gray_flat = gray.reshape(-1)
    hist_values, _ = np.histogram(gray_flat, bins=32, range=(0.0, 1.0), density=False)
    probs = hist_values.astype(np.float64) / max(hist_values.sum(), 1)
    entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2).reshape(-1)
    values.extend(
        [
            float(gray_flat.mean()),
            float(gray_flat.std()),
            float(gray_flat.min()),
            float(gray_flat.max()),
            entropy,
            float(edge.mean()),
            float(edge.std()),
            float(np.quantile(edge, 0.90)),
        ]
    )

    gray_hist, _ = np.histogram(gray_flat, bins=16, range=(0.0, 1.0), density=True)
    values.extend([float(x) for x in gray_hist])
    for channel_idx in range(3):
        channel_hist, _ = np.histogram(arr[:, :, channel_idx].reshape(-1), bins=8, range=(0.0, 1.0), density=True)
        values.extend([float(x) for x in channel_hist])

    thumb = np.asarray(gray_img.resize(THUMBNAIL_SIZE), dtype=np.float32).reshape(-1) / 255.0
    values.extend([float(x) for x in thumb])
    return np.asarray(values, dtype=np.float32)

