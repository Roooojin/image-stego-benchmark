from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import read_image, to_gray, save_image
from src.interpolation import resize


def make_grid(original: np.ndarray, variants: list[tuple[str, np.ndarray]], out_path: Path) -> None:
    cols = 1 + len(variants)
    plt.figure(figsize=(3.6 * cols, 3.6))
    plt.subplot(1, cols, 1)
    plt.title("original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    for i, (name, img) in enumerate(variants, start=2):
        plt.subplot(1, cols, i)
        plt.title(name)
        plt.imshow(img, cmap="gray")
        plt.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.3, 2.3])
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    img = to_gray(read_image(args.image))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    methods = ["nearest", "bilinear", "bicubic"]

    for s in args.scales:
        variants = []
        for m in methods:
            r = resize(img, scale=s, method=m)
            variants.append((f"{m} {s}x", r))
            save_image(outdir / f"resize_{m}_{s}x.png", r)

        make_grid(img, variants, outdir / f"grid_resize_{s}x.png")

    print(f"[OK] Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()