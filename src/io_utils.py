from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import imageio.v2 as imageio

Array = np.ndarray
PathLike = Union[str, Path]


def read_image(path: PathLike) -> Array:
    return imageio.imread(path)


def to_gray(img: Array) -> Array:
    if img.ndim == 2:
        return img.astype(np.uint8)

    if img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return np.clip(gray, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def ensure_uint8(img: Array) -> Array:
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def save_image(path: PathLike, img: Array) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, ensure_uint8(img))