from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import read_image, to_gray, save_image
from src.metrics import mse, psnr, bit_accuracy
from src.noise import add_gaussian_noise
from src.stego import load_meta, extract_logo


def save_grid(items: list[tuple[str, np.ndarray]], out_path: Path, cols: int = 3) -> None:
    rows = int(np.ceil(len(items) / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (title, img) in enumerate(items, start=1):
        plt.subplot(rows, cols, i)
        plt.title(title)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stego", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--method", choices=["lsb", "between"], default="lsb")
    ap.add_argument("--variances", nargs="+", type=float, default=[0.02, 0.15, 0.5, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    tbldir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tbldir.mkdir(parents=True, exist_ok=True)

    stego = to_gray(read_image(args.stego))
    meta = load_meta(args.meta)

    gt_logo = extract_logo(stego, method=args.method, meta=meta)
    gt_bits = (gt_logo > 128).astype(np.uint8)

    rows = []
    previews = []

    for var in args.variances:
        noisy = add_gaussian_noise(stego, var=var, seed=args.seed)
        save_image(outdir / f"stego_{args.method}_noise_var{var}.png", noisy)

        rec = extract_logo(noisy, method=args.method, meta=meta)
        save_image(outdir / f"recovered_{args.method}_var{var}.png", rec)

        acc = bit_accuracy(gt_bits, (rec > 128).astype(np.uint8))
        m = mse(stego, noisy)
        p = psnr(stego, noisy, max_value=255.0)

        rows.append((var, m, p, acc))
        previews.append((f"var={var}", rec))

    csv_path = tbldir / f"noise_eval_{args.method}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variance,mse_stego,psnr_stego,logo_bit_accuracy\n")
        for var, m, p, acc in rows:
            f.write(f"{var},{m:.6f},{p:.4f},{acc:.6f}\n")

    save_grid(previews, figdir / f"grid_recovered_logos_{args.method}.png", cols=min(3, len(previews)))

    print(f"[OK] CSV:  {csv_path.resolve()}")
    print(f"[OK] Grid: {(figdir / f'grid_recovered_logos_{args.method}.png').resolve()}")


if __name__ == "__main__":
    main()