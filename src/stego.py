from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

from .io_utils import to_gray
from .interpolation import resize as resize_gray

Array = np.ndarray
Method = Literal["lsb", "between"]

MAGIC = 0x53544547  # 'STEG'


def resize_logo(logo_img: Array, scale: float = 0.15) -> Array:
    """لوگو را کوچک می‌کند (با bilinear) و uint8 برمی‌گرداند."""
    g = to_gray(logo_img)
    return resize_gray(g, scale=scale, method="bilinear")


def binarize(img: Array, threshold: int = 128) -> Array:
    return (to_gray(img) > threshold).astype(np.uint8)


def _u32_to_bits(x: int) -> list[int]:
    return [(x >> (31 - i)) & 1 for i in range(32)]


def _u16_to_bits(x: int) -> list[int]:
    return [(x >> (15 - i)) & 1 for i in range(16)]


def _bits_to_u32(bits: list[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def _bits_to_u16(bits: list[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def make_payload(logo_bits: Array, threshold: int = 128) -> Array:
    """
    Payload:
    - 32 bits MAGIC
    - 16 bits H
    - 16 bits W
    - 8  bits threshold
    - logo bits (H*W)
    """
    h, w = logo_bits.shape
    header = (
        _u32_to_bits(MAGIC)
        + _u16_to_bits(h)
        + _u16_to_bits(w)
        + [(threshold >> (7 - i)) & 1 for i in range(8)]
    )
    payload = np.concatenate([np.array(header, dtype=np.uint8), logo_bits.astype(np.uint8).ravel()])
    return payload


def parse_header(bits72: Array) -> Dict[str, int]:
    bits = bits72.astype(np.uint8).tolist()
    magic = _bits_to_u32(bits[:32])
    if magic != MAGIC:
        raise ValueError("Invalid MAGIC. Not a STEG payload.")
    h = _bits_to_u16(bits[32:48])
    w = _bits_to_u16(bits[48:64])
    thr = 0
    for b in bits[64:72]:
        thr = (thr << 1) | b
    return {"height": h, "width": w, "threshold": thr, "header_bits": 72}


def capacity(host_gray: Array, method: Method) -> int:
    n = host_gray.size
    return n if method == "lsb" else max(0, n - 1)


def embed(host_img: Array, payload_bits: Array, method: Method) -> Array:
    host = to_gray(host_img)
    flat = host.flatten().astype(np.int16)
    bits = payload_bits.astype(np.uint8).ravel()

    if len(bits) > capacity(host, method):
        raise ValueError(f"Payload too large ({len(bits)} bits) > capacity ({capacity(host, method)} bits)")

    if method == "lsb":
        stego = flat.copy()
        stego[: len(bits)] = (stego[: len(bits)] & 0xFE) | bits
        return stego.reshape(host.shape).astype(np.uint8)

    # between-pixels parity method (مثل کد تو)
    stego = flat.copy()
    for i, bit in enumerate(bits):
        a = int(stego[i])
        b = int(stego[i + 1])
        parity = abs(a - b) & 1
        if parity != int(bit):
            stego[i] = a + 1 if a < 255 else a - 1
    return stego.reshape(host.shape).astype(np.uint8)


def extract_bits(stego_img: Array, method: Method, num_bits: int) -> Array:
    stego = to_gray(stego_img)
    flat = stego.flatten().astype(np.int16)

    if method == "lsb":
        return (flat[:num_bits] & 1).astype(np.uint8)

    return np.array([(abs(int(flat[i]) - int(flat[i + 1])) & 1) for i in range(num_bits)], dtype=np.uint8)


def extract_logo(stego_img: Array, method: Method, meta: Dict[str, int]) -> Array:
    h = int(meta["height"])
    w = int(meta["width"])
    header_bits = int(meta["header_bits"])
    total_bits = header_bits + h * w

    bits = extract_bits(stego_img, method=method, num_bits=total_bits)
    logo_bits = bits[header_bits:].reshape(h, w).astype(np.uint8)
    return (logo_bits * 255).astype(np.uint8)


def save_meta(path: str | Path, meta: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)