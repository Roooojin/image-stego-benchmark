from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

Array = np.ndarray
Method = Literal["nearest", "bilinear", "bicubic"]


def _clip_uint8(x: Array) -> Array:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _map_coords(out_len: int, scale: float) -> Array:
    # center-aligned mapping
    out = np.arange(out_len, dtype=np.float32)
    return (out + 0.5) / scale - 0.5


def resize(gray: Array, scale: float, method: Method = "bilinear") -> Array:
    """Resize grayscale (H,W) uint8. خروجی uint8."""
    if gray.ndim != 2:
        raise ValueError("resize() expects grayscale image (H,W)")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    h, w = gray.shape
    oh = max(1, int(round(h * scale)))
    ow = max(1, int(round(w * scale)))

    src = gray.astype(np.float32)
    sy, sx = scale, scale
    y = _map_coords(oh, sy)
    x = _map_coords(ow, sx)

    if method == "nearest":
        yi = np.clip(np.round(y).astype(int), 0, h - 1)
        xi = np.clip(np.round(x).astype(int), 0, w - 1)
        out = src[yi[:, None], xi[None, :]]
        return _clip_uint8(out)

    if method == "bilinear":
        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)
        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)

        wy = (y - y0).astype(np.float32)
        wx = (x - x0).astype(np.float32)

        Ia = src[y0c[:, None], x0c[None, :]]
        Ib = src[y0c[:, None], x1c[None, :]]
        Ic = src[y1c[:, None], x0c[None, :]]
        Id = src[y1c[:, None], x1c[None, :]]

        out = (
            Ia * (1 - wy)[:, None] * (1 - wx)[None, :]
            + Ib * (1 - wy)[:, None] * wx[None, :]
            + Ic * wy[:, None] * (1 - wx)[None, :]
            + Id * wy[:, None] * wx[None, :]
        )
        return _clip_uint8(out)

    if method == "bicubic":
        def kernel(t: Array, a: float = -0.5) -> Array:
            at = np.abs(t)
            at2 = at * at
            at3 = at2 * at
            k = np.zeros_like(at, dtype=np.float32)
            m1 = at <= 1
            m2 = (at > 1) & (at < 2)
            k[m1] = (a + 2) * at3[m1] - (a + 3) * at2[m1] + 1
            k[m2] = a * at3[m2] - 5 * a * at2[m2] + 8 * a * at[m2] - 4 * a
            return k

        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        out = np.zeros((oh, ow), dtype=np.float32)

        for m in range(-1, 3):
            ym = np.clip(y0 + m, 0, h - 1)
            wy = kernel(y - (y0 + m))
            for n in range(-1, 3):
                xn = np.clip(x0 + n, 0, w - 1)
                wx = kernel(x - (x0 + n))
                out += src[ym[:, None], xn[None, :]] * (wy[:, None] * wx[None, :])

        return _clip_uint8(out)

    raise ValueError(f"Unknown method: {method}")