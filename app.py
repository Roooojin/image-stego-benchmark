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


tabs = st.tabs(["Resizing", "Steganography", "Noise Robustness"])


with tabs[0]:
    st.subheader("Resizing")
    left, right = st.columns([1, 1], gap="large")

    with left:
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="resize")
        method = st.selectbox("Method", ["nearest", "bilinear", "bicubic"], index=1)
        scale = st.slider("Scale", 0.1, 3.0, 2.3, 0.1)

    if up is not None:
        img = to_gray(load_upload(up))
        out = resize_img(img, scale=scale, method=method)

        with right:
            c1, c2 = st.columns(2)
            c1.markdown("**Original**")
            c1.image(img, clamp=True)
            c2.markdown(f"**Resized ({method}, {scale}×)**")
            c2.image(out, clamp=True)


with tabs[1]:
    st.subheader("Steganography (LSB / Between-pixels)")
    a, b, c = st.columns([1, 1, 1], gap="large")

    with a:
        cover_up = st.file_uploader("Cover image", type=["png", "jpg", "jpeg"], key="cover")
        logo_up = st.file_uploader("Logo image", type=["png", "jpg", "jpeg"], key="logo")
        method = st.selectbox("Method", ["lsb", "between"], index=0, key="stego_method")
        logo_scale = st.slider("Logo scale", 0.05, 0.5, 0.15, 0.01)
        threshold = st.slider("Threshold", 0, 255, 128, 1)

    if cover_up is not None and logo_up is not None:
        cover = to_gray(load_upload(cover_up))
        logo_raw = load_upload(logo_up)

        logo_small = resize_logo(logo_raw, scale=logo_scale)
        logo_bits = (logo_small > threshold).astype(np.uint8)
        payload = make_payload(logo_bits, threshold=threshold)
        meta = parse_header(payload[:72])

        stego_img = embed(cover, payload, method=method)
        rec_logo = extract_logo(stego_img, method=method, meta=meta)

        with b:
            st.markdown("**Cover**")
            st.image(cover, clamp=True)
            st.markdown("**Logo (resized)**")
            st.image(logo_small, clamp=True)

        with c:
            st.markdown("**Stego**")
            st.image(stego_img, clamp=True)
            st.markdown("**Recovered logo**")
            st.image(rec_logo, clamp=True)

        host_mse = mse(cover, stego_img)
        host_psnr = psnr(cover, stego_img, max_value=255.0)
        acc = bit_accuracy(logo_bits, (rec_logo > 128).astype(np.uint8))

        st.success(f"Host → MSE={host_mse:.2f}, PSNR={host_psnr:.2f} dB | Logo accuracy={acc*100:.2f}%")
        st.download_button(
            "Download meta JSON",
            data=json.dumps(meta, indent=2).encode("utf-8"),
            file_name=f"stego_meta_{method}.json",
            mime="application/json",
        )


with tabs[2]:
    st.subheader("Noise Robustness (Gaussian)")
    left, right = st.columns([1, 2], gap="large")

    with left:
        stego_up = st.file_uploader("Stego image", type=["png", "jpg", "jpeg"], key="stego_noise")
        meta_up = st.file_uploader("Meta JSON", type=["json"], key="meta_noise")
        method = st.selectbox("Method", ["lsb", "between"], index=0, key="noise_method")
        variances = st.multiselect("Variances (σ²)", [0.02, 0.15, 0.5, 2.0], default=[0.02, 0.15, 0.5, 2.0])
        seed = st.number_input("Seed", value=0, step=1)

    if stego_up is not None and meta_up is not None and len(variances) > 0:
        stego_img = to_gray(load_upload(stego_up))
        meta = json.loads(meta_up.read().decode("utf-8"))

        gt = extract_logo(stego_img, method=method, meta=meta)
        gt_bits = (gt > 128).astype(np.uint8)

        table = []
        previews = []

        for v in variances:
            noisy = add_gaussian_noise(stego_img, var=float(v), seed=int(seed))
            rec = extract_logo(noisy, method=method, meta=meta)

            acc = bit_accuracy(gt_bits, (rec > 128).astype(np.uint8))
            m = mse(stego_img, noisy)
            p = psnr(stego_img, noisy, max_value=255.0)

            table.append({"variance": v, "mse": m, "psnr(dB)": p, "logo_acc": acc})
            previews.append((f"var={v}", rec))

        with right:
            st.markdown("**Recovered logos**")
            cols = st.columns(min(3, len(previews)))
            for i, (title, img) in enumerate(previews):
                cols[i % len(cols)].markdown(f"**{title}**")
                cols[i % len(cols)].image(img, clamp=True)

            st.markdown("**Metrics**")
            st.dataframe(table, use_container_width=True)