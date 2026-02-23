from __future__ import annotations

from typing import Optional
import numpy as np

Array = np.ndarray


def add_gaussian_noise(img: Array, var: float, seed: Optional[int] = 0) -> Array:
    if var < 0:
        raise ValueError("var must be >= 0")

    rng = np.random.default_rng(seed)

    if img.dtype == np.uint8:
        x = img.astype(np.float32) / 255.0
        n = rng.normal(0.0, np.sqrt(var), size=x.shape).astype(np.float32)
        y = np.clip(x + n, 0, 1)
        return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)

    x = img.astype(np.float32)
    n = rng.normal(0.0, np.sqrt(var), size=x.shape).astype(np.float32)
    return np.clip(x + n, 0, 1)