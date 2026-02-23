from __future__ import annotations

import math
from typing import Optional

import numpy as np

Array = np.ndarray


def mse(a: Array, b: Array) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(a: Array, b: Array, max_value: Optional[float] = None) -> float:
    m = mse(a, b)
    if m == 0:
        return float("inf")
    if max_value is None:
        max_value = 255.0 if (a.dtype == np.uint8 or b.dtype == np.uint8) else 1.0
    return float(10.0 * math.log10((max_value ** 2) / m))


def bit_accuracy(a_bits: Array, b_bits: Array) -> float:
    if a_bits.shape != b_bits.shape:
        raise ValueError(f"Shape mismatch: {a_bits.shape} vs {b_bits.shape}")
    a_bits = a_bits.astype(np.uint8).ravel()
    b_bits = b_bits.astype(np.uint8).ravel()
    return float(np.mean(a_bits == b_bits))