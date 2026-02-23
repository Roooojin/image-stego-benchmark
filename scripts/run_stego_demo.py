from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.io_utils import read_image, to_gray, save_image
from src.metrics import mse, psnr, bit_accuracy
from src.stego import resize_logo, binarize, make_payload, parse_header, embed, extract_logo, save_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True)
    ap.add_argument("--logo", required=True)
    ap.add_argument("--logo-scale", type=float, default=0.15)
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--method", choices=["lsb", "between", "both"], default="both")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cover = to_gray(read_image(args.cover))
    logo_raw = read_image(args.logo)

    logo_small = resize_logo(logo_raw, scale=args.logo_scale)
    logo_bits = (logo_small > args.threshold).astype(np.uint8)

    payload = make_payload(logo_bits, threshold=args.threshold)
    meta = parse_header(payload[:72])

    save_image(outdir / "cover.png", cover)
    save_image(outdir / "logo_resized.png", logo_small)

    def run_one(m: str):
        stego = embed(cover, payload, method=m)
        save_image(outdir / f"stego_{m}.png", stego)

        meta_path = outdir / f"stego_{m}.meta.json"
        save_meta(meta_path, meta)

        recovered = extract_logo(stego, method=m, meta=meta)
        save_image(outdir / f"recovered_{m}.png", recovered)

        host_mse = mse(cover, stego)
        host_psnr = psnr(cover, stego, max_value=255.0)
        acc = bit_accuracy(logo_bits, (recovered > 128).astype(np.uint8))

        print(f"\n=== {m.upper()} ===")
        print(f"MSE(host, stego):  {host_mse:.4f}")
        print(f"PSNR(host, stego): {host_psnr:.2f} dB")
        print(f"Logo bit-accuracy: {acc*100:.2f}%")
        print(f"Saved meta: {meta_path}")

    if args.method in ("lsb", "both"):
        run_one("lsb")
    if args.method in ("between", "both"):
        run_one("between")


if __name__ == "__main__":
    main()