# Image Interpolation + Steganography Benchmark (Python)

Modular implementation of:
- Nearest / Bilinear / Bicubic resizing (NumPy)
- LSB + Between-pixels steganography
- MSE / PSNR + logo bit-accuracy
- Gaussian noise robustness evaluation
- Streamlit UI demo

## Install
```bash
pip install -r requirements.txt
```
1) Resizing benchmark
```bash
python -m scripts.run_resize_benchmark --image data/input/lena.png --scales 0.3 2.3
```
Outputs:
```md

results/figures/grid_resize_0.3x.png

results/figures/grid_resize_2.3x.png
```
2) Steganography demo
```bash
python -m scripts.run_stego_demo --cover data/input/cameraman.png --logo data/input/university_logo.png --logo-scale 0.15 --method both
```
Outputs:
```md

results/stego_lsb.png, results/recovered_lsb.png, results/stego_lsb.meta.json

results/stego_between.png, results/recovered_between.png, results/stego_between.meta.json

results/cover.png, results/logo_resized.png
```
3) Noise robustness test
```bash
python -m scripts.run_noise_robustness --stego results/stego_lsb.png --meta results/stego_lsb.meta.json --method lsb --variances 0.02 0.15 0.5 2
```
Outputs:
```md

results/tables/noise_eval_lsb.csv

results/figures/grid_recovered_logos_lsb.png
```
## UI (Streamlit)
```bash
streamlit run app.py
```




