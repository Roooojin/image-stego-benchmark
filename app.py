from __future__ import annotations

import io
import json
import csv
from typing import Any, Dict, List, Tuple

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
# Cache last generated stego/meta per method, so Noise tab can work without manual JSON upload.
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
                "gt_bits": logo_bits,         # TRUE ground truth bits
                "cover": cover,
                "logo_small": logo_small,
            }

            return stego_img, rec_logo, host_mse, host_psnr, acc

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
        view_mode = st.selectbox(
            "View mode",
            ["Single method", "Compare (LSB vs Between)"],
            index=1 if ("lsb" in st.session_state["stego_cache"] and "between" in st.session_state["stego_cache"]) else 0,
            key="noise_view_mode",
        )

        method = st.selectbox("Method (single mode)", ["lsb", "between"], index=0, key="noise_method")

        include_zero = st.checkbox("Include variance 0.0 (no noise)", value=True, key="include_var0")

        variances = st.multiselect(
            "Variances (σ²)",
            [0.02, 0.15, 0.5, 2.0],
            default=[0.02, 0.15, 0.5, 2.0],
            key="noise_vars",
        )

        seed = st.number_input("Random seed", value=0, step=1, key="noise_seed")

        st.divider()

        # Cache availability
        has_cache_lsb = "lsb" in st.session_state["stego_cache"]
        has_cache_between = "between" in st.session_state["stego_cache"]
        can_use_cache_single = True if (method in st.session_state["stego_cache"]) else False
        can_use_cache_compare = has_cache_lsb and has_cache_between

        use_cache = st.checkbox(
            "Use cached stego/meta from Steganography tab (recommended)",
            value=True if (can_use_cache_compare if view_mode.startswith("Compare") else can_use_cache_single) else False,
            disabled=not (can_use_cache_compare if view_mode.startswith("Compare") else can_use_cache_single),
            help="Generate stego in the Steganography tab first to enable this option."
        )

        # Upload fallbacks (only if not using cache)
        stego_up = None
        meta_up = None
        if not use_cache:
            st.caption("Upload stego + meta JSON (fallback mode)")
            stego_up = st.file_uploader("Stego image", type=["png", "jpg", "jpeg"], key="stego_noise")
            meta_up = st.file_uploader("Meta JSON", type=["json"], key="meta_noise")

    def _make_variance_list() -> List[float]:
        vs = [float(v) for v in variances]
        if include_zero and 0.0 not in vs:
            vs = [0.0] + vs
        return vs

    def _table_to_csv_bytes(rows: List[Dict[str, Any]], filename: str) -> Tuple[bytes, str]:
        # rows: list of dicts with same keys
        if not rows:
            return b"", filename
        keys = list(rows[0].keys())
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return out.getvalue().encode("utf-8"), filename

    def evaluate_noise_for_method(m: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, np.ndarray]], str]:
        """
        Returns:
          - table rows
          - previews (title, recovered_logo_img)
          - note string about GT source
        """
        # get stego/meta either from cache or upload
        if use_cache and m in st.session_state["stego_cache"]:
            cached = st.session_state["stego_cache"][m]
            stego_img = cached["stego"]
            meta = cached["meta"]
            gt_bits = cached.get("gt_bits", None)
            note = "GT bits: from Steganography tab (true ground-truth)."
        else:
            if stego_up is None or meta_up is None:
                raise RuntimeError("Missing uploads.")
            stego_img = to_gray(load_upload(stego_up))
            meta = json.loads(meta_up.read().decode("utf-8"))
            gt_bits = None
            note = "GT bits: fallback from clean extraction (uploads mode)."

        # prepare GT
        if gt_bits is None:
            ref = extract_logo(stego_img, method=m, meta=meta)
            gt_bits_local = (ref > 128).astype(np.uint8)
        else:
            gt_bits_local = gt_bits

        vs = _make_variance_list()
        table: List[Dict[str, Any]] = []
        previews: List[Tuple[str, np.ndarray]] = []

        for v in vs:
            if v == 0.0:
                noisy = stego_img
            else:
                noisy = add_gaussian_noise(stego_img, var=float(v), seed=int(seed))

            rec = extract_logo(noisy, method=m, meta=meta)

            acc = bit_accuracy(gt_bits_local, (rec > 128).astype(np.uint8))
            mval = mse(stego_img, noisy)
            pval = psnr(stego_img, noisy, max_value=255.0)

            table.append({
                "method": m,
                "variance": float(v),
                "mse(stego,noisy)": float(mval),
                "psnr(dB)": float(pval),
                "logo_acc": float(acc),
            })
            previews.append((f"var={v}", rec))

        return table, previews, note

    def render_method_block(col, title: str, table: List[Dict[str, Any]], previews: List[Tuple[str, np.ndarray]]):
        with col:
            st.markdown(f"### {title}")

            st.markdown("**Recovered logos**")
            cols = st.columns(min(3, len(previews)))
            for i, (t, img) in enumerate(previews):
                cols[i % len(cols)].markdown(f"**{t}**")
                cols[i % len(cols)].image(img, clamp=True)

            st.markdown("**Metrics**")
            st.dataframe(table, use_container_width=True)

            csv_bytes, fname = _table_to_csv_bytes(table, f"noise_eval_{title.lower().replace(' ', '_')}.csv")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
                key=f"dl_csv_{title}",
            )

    # ---------------------------
    # Render logic
    # ---------------------------
    if len(variances) == 0:
        with right:
            st.warning("Please select at least one variance.")
    else:
        if view_mode.startswith("Compare"):
            # Compare requires cache for both methods (recommended UX)
            if not (use_cache and ("lsb" in st.session_state["stego_cache"]) and ("between" in st.session_state["stego_cache"])):
                with right:
                    st.warning(
                        "Compare mode works best with cached results.\n\n"
                        "Go to **Steganography** tab → select **both** → generate once.\n"
                        "Then come back here and enable cache."
                    )
            else:
                table_lsb, previews_lsb, note_lsb = evaluate_noise_for_method("lsb")
                table_bet, previews_bet, note_bet = evaluate_noise_for_method("between")

                with right:
                    st.info(f"LSB: {note_lsb}  |  Between: {note_bet}")

                    colA, colB = st.columns(2, gap="large")
                    render_method_block(colA, "LSB", table_lsb, previews_lsb)
                    render_method_block(colB, "Between", table_bet, previews_bet)

                    # Combined CSV
                    combined = table_lsb + table_bet
                    csv_bytes, fname = _table_to_csv_bytes(combined, "noise_eval_compare_lsb_between.csv")
                    st.download_button(
                        "Download combined CSV (LSB + Between)",
                        data=csv_bytes,
                        file_name=fname,
                        mime="text/csv",
                        key="dl_csv_combined",
                    )

        else:
            # Single method view
            try:
                table, previews, note = evaluate_noise_for_method(method)
                with right:
                    st.info(f"Method **{method}**. {note}")

                    col = st.container()
                    render_method_block(col, method.upper(), table, previews)
            except RuntimeError:
                with right:
                    if use_cache and method not in st.session_state["stego_cache"]:
                        st.warning(
                            f"No cached data for **{method}** yet. "
                            "Go to Steganography tab and generate it first (or disable cache and upload stego + meta)."
                        )
                    else:
                        st.warning("Please upload both the stego image and meta JSON (or enable cache).")