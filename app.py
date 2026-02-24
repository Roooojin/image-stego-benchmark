from __future__ import annotations

import io
import json

import numpy as np
import streamlit as st
import imageio.v2 as imageio

from src.io_utils import to_gray
from src.interpolation import resize as resize_img
from src.metrics import mse, psnr, bit_accuracy
from src.noise import add_gaussian_noise
from src.stego import resize_logo, make_payload, parse_header, embed, extract_logo

st.set_page_config(page_title="Interpolation + Stego Benchmark", layout="wide")

st.title("🖼️ Image Interpolation + Steganography Benchmark")
st.caption("Nearest/Bilinear/Bicubic resizing + LSB/Between steganography + PSNR/MSE + noise robustness")


def load_upload(u) -> np.ndarray:
    data = u.read()
    return imageio.imread(io.BytesIO(data))


# ---------------------------
# Session-state initialization
# ---------------------------
# We cache last generated stego/meta per method, so Noise tab can work without manual JSON upload.
if "stego_cache" not in st.session_state:
    # structure: { "lsb": {"stego":..., "meta":..., "gt_bits":..., "cover":..., "logo_small":...}, "between": {...} }
    st.session_state["stego_cache"] = {}


tabs = st.tabs(["Resizing", "Steganography", "Noise Robustness"])


# ==========================
# Tab 1: Resizing
# ==========================
with tabs[0]:
    st.subheader("Resizing")

    left, right = st.columns([1, 1], gap="large")

    with left:
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="resize")
        method = st.selectbox("Method", ["nearest", "bilinear", "bicubic"], index=1, key="resize_method")
        scale = st.slider("Scale", 0.1, 3.0, 2.3, 0.1, key="resize_scale")

    if up is not None:
        img = to_gray(load_upload(up))
        out = resize_img(img, scale=scale, method=method)

        with right:
            c1, c2 = st.columns(2)
            c1.markdown("**Original**")
            c1.image(img, clamp=True)
            c2.markdown(f"**Resized ({method}, {scale}×)**")
            c2.image(out, clamp=True)


# ==========================
# Tab 2: Steganography
# ==========================
with tabs[1]:
    st.subheader("Steganography (LSB / Between-pixels)")

    a, b, c = st.columns([1, 1, 1], gap="large")

    with a:
        cover_up = st.file_uploader("Cover image", type=["png", "jpg", "jpeg"], key="cover")
        logo_up = st.file_uploader("Logo image", type=["png", "jpg", "jpeg"], key="logo")

        method = st.selectbox("Method", ["lsb", "between", "both"], index=2, key="stego_method")
        logo_scale = st.slider("Logo scale", 0.05, 0.5, 0.15, 0.01, key="logo_scale")
        threshold = st.slider("Threshold", 0, 255, 128, 1, key="logo_thr")

        st.caption("Tip: Choose **both** to generate & cache results for LSB and Between in one shot.")

    if cover_up is not None and logo_up is not None:
        cover = to_gray(load_upload(cover_up))
        logo_raw = load_upload(logo_up)

        logo_small = resize_logo(logo_raw, scale=logo_scale)
        logo_bits = (logo_small > threshold).astype(np.uint8)

        payload = make_payload(logo_bits, threshold=threshold)
        meta = parse_header(payload[:72])

        with b:
            st.markdown("**Cover**")
            st.image(cover, clamp=True)
            st.markdown("**Logo (resized)**")
            st.image(logo_small, clamp=True)

        def run_one(m: str):
            stego_img = embed(cover, payload, method=m)
            rec_logo = extract_logo(stego_img, method=m, meta=meta)

            host_mse = mse(cover, stego_img)
            host_psnr = psnr(cover, stego_img, max_value=255.0)
            acc = bit_accuracy(logo_bits, (rec_logo > 128).astype(np.uint8))

            # Cache everything for Noise tab
            st.session_state["stego_cache"][m] = {
                "stego": stego_img,
                "meta": meta,
                "gt_bits": logo_bits,         # IMPORTANT: true ground truth
                "cover": cover,
                "logo_small": logo_small,
            }

            return stego_img, rec_logo, host_mse, host_psnr, acc

        # Display area
        with c:
            if method in ("lsb", "between"):
                stego_img, rec_logo, host_mse, host_psnr, acc = run_one(method)

                st.markdown(f"**Stego ({method})**")
                st.image(stego_img, clamp=True)
                st.markdown("**Recovered logo**")
                st.image(rec_logo, clamp=True)

                st.success(
                    f"Host → MSE={host_mse:.4f}, PSNR={host_psnr:.2f} dB | "
                    f"Logo accuracy={acc*100:.2f}%"
                )

                st.download_button(
                    f"Download meta JSON ({method})",
                    data=json.dumps(meta, indent=2).encode("utf-8"),
                    file_name=f"stego_meta_{method}.json",
                    mime="application/json",
                )

            else:
                # BOTH: run and show side-by-side
                st.markdown("**Generated for both methods**")

                col1, col2 = st.columns(2)

                with col1:
                    stego_lsb, rec_lsb, mse_lsb, psnr_lsb, acc_lsb = run_one("lsb")
                    st.markdown("### LSB")
                    st.image(stego_lsb, clamp=True, caption="stego_lsb")
                    st.image(rec_lsb, clamp=True, caption="recovered_lsb")
                    st.info(f"MSE={mse_lsb:.4f} | PSNR={psnr_lsb:.2f} dB | Acc={acc_lsb*100:.2f}%")
                    st.download_button(
                        "Download meta JSON (lsb)",
                        data=json.dumps(meta, indent=2).encode("utf-8"),
                        file_name="stego_meta_lsb.json",
                        mime="application/json",
                        key="dl_meta_lsb",
                    )

                with col2:
                    stego_b, rec_b, mse_b, psnr_b, acc_b = run_one("between")
                    st.markdown("### Between-pixels")
                    st.image(stego_b, clamp=True, caption="stego_between")
                    st.image(rec_b, clamp=True, caption="recovered_between")
                    st.info(f"MSE={mse_b:.4f} | PSNR={psnr_b:.2f} dB | Acc={acc_b*100:.2f}%")
                    st.download_button(
                        "Download meta JSON (between)",
                        data=json.dumps(meta, indent=2).encode("utf-8"),
                        file_name="stego_meta_between.json",
                        mime="application/json",
                        key="dl_meta_between",
                    )

        st.caption("✅ Cached outputs are now available in the Noise tab without uploading JSON.")


# ==========================
# Tab 3: Noise Robustness
# ==========================
with tabs[2]:
    st.subheader("Noise Robustness (Gaussian)")
    st.caption("Adds Gaussian noise to the stego image and measures degradation + recovered-logo accuracy.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        method = st.selectbox("Method", ["lsb", "between"], index=0, key="noise_method")

        # If we have cached data for this method, allow using it.
        has_cache = method in st.session_state["stego_cache"]
        use_cache_default = True if has_cache else False
        use_cache = st.checkbox(
            "Use last generated stego from Steganography tab (no JSON upload)",
            value=use_cache_default,
            disabled=not has_cache,
            help="Generate stego in the Steganography tab first to enable this option."
        )

        variances = st.multiselect(
            "Variances (σ²)", [0.02, 0.15, 0.5, 2.0],
            default=[0.02, 0.15, 0.5, 2.0],
            key="noise_vars"
        )
        seed = st.number_input("Random seed", value=0, step=1, key="noise_seed")

        st.divider()

        # Fallback uploaders (only if not using cache)
        stego_up = None
        meta_up = None
        logo_up = None

        if not use_cache:
            stego_up = st.file_uploader("Stego image", type=["png", "jpg", "jpeg"], key="stego_noise")
            meta_up = st.file_uploader("Meta JSON", type=["json"], key="meta_noise")

            st.caption("Optional: upload the original logo to compute TRUE accuracy.")
            logo_up = st.file_uploader("Original logo (optional)", type=["png", "jpg", "jpeg"], key="logo_noise")

    def run_noise_eval(stego_img: np.ndarray, meta: dict, gt_bits: np.ndarray | None):
        # If gt_bits not provided, we fall back to extraction on clean stego as reference
        if gt_bits is None:
            ref = extract_logo(stego_img, method=method, meta=meta)
            gt_bits_local = (ref > 128).astype(np.uint8)
            gt_source_note = "Reference bits are taken from clean extraction (fallback)."
        else:
            gt_bits_local = gt_bits
            gt_source_note = "Reference bits are ground-truth logo bits from Steganography tab."

        table = []
        previews = []

        for v in variances:
            noisy = add_gaussian_noise(stego_img, var=float(v), seed=int(seed))
            rec = extract_logo(noisy, method=method, meta=meta)

            acc = bit_accuracy(gt_bits_local, (rec > 128).astype(np.uint8))
            m = mse(stego_img, noisy)
            p = psnr(stego_img, noisy, max_value=255.0)

            table.append({"variance": float(v), "mse(stego,noisy)": m, "psnr(dB)": p, "logo_acc": acc})
            previews.append((f"var={v}", rec))

        return table, previews, gt_source_note

    if len(variances) == 0:
        st.warning("Please select at least one variance.")
    else:
        # Decide data source: cache or uploads
        if "noise_method" in st.session_state:
            method = st.session_state["noise_method"]

        if method in st.session_state["stego_cache"] and use_cache:
            cached = st.session_state["stego_cache"][method]
            stego_img = cached["stego"]
            meta = cached["meta"]
            gt_bits = cached.get("gt_bits", None)

            table, previews, note = run_noise_eval(stego_img, meta, gt_bits)

            with right:
                st.info(f"Using cached stego/meta for method **{method}**. {note}")

                st.markdown("**Recovered logos**")
                cols = st.columns(min(3, len(previews)))
                for i, (title, img) in enumerate(previews):
                    cols[i % len(cols)].markdown(f"**{title}**")
                    cols[i % len(cols)].image(img, clamp=True)

                st.markdown("**Metrics**")
                st.dataframe(table, use_container_width=True)

        else:
            # Upload path
            if stego_up is not None and meta_up is not None:
                stego_img = to_gray(load_upload(stego_up))
                meta = json.loads(meta_up.read().decode("utf-8"))

                # If user provided original logo, compute true gt_bits
                gt_bits = None
                if logo_up is not None:
                    logo_raw = load_upload(logo_up)
                    logo_small = resize_logo(logo_raw, scale=float(st.session_state.get("logo_scale", 0.15)))
                    thr = int(meta.get("threshold", 128))
                    gt_bits = (to_gray(logo_small) > thr).astype(np.uint8)

                table, previews, note = run_noise_eval(stego_img, meta, gt_bits)

                with right:
                    st.info(f"Using uploaded stego/meta for method **{method}**. {note}")

                    st.markdown("**Recovered logos**")
                    cols = st.columns(min(3, len(previews)))
                    for i, (title, img) in enumerate(previews):
                        cols[i % len(cols)].markdown(f"**{title}**")
                        cols[i % len(cols)].image(img, clamp=True)

                    st.markdown("**Metrics**")
                    st.dataframe(table, use_container_width=True)
            else:
                with right:
                    if method not in st.session_state["stego_cache"]:
                        st.warning(
                            "No cached data for this method yet. "
                            "Go to the Steganography tab, generate stego for this method, "
                            "then come back here (or upload stego + meta JSON)."
                        )
                    else:
                        st.warning("Please upload both the stego image and meta JSON (or enable cache).")