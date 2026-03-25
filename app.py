# QCI AstroEntangle Refiner – v5 FULL MERGE (UI + Full Physics Pipeline)
# Combines your original v4 physics + upgraded real-time UI

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve2d

try:
    from PIL import Image as PILImage
    PIL_OK = True
except:
    PIL_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except:
    TORCH_OK = False

# ── CONFIG ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI Refiner v5", page_icon="🔭")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0b1a; }
[data-testid="stSidebar"] { background: #10102a; }
</style>
""", unsafe_allow_html=True)

# ── NEURAL SR (YOUR ORIGINAL) ──────────────────────────
if TORCH_OK:
    class EDSR_Small(nn.Module):
        def __init__(self, scale=2):
            super().__init__()
            self.scale = scale
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.res = nn.Sequential(*[self._rb() for _ in range(8)])
            self.conv_up = nn.Conv2d(32, 32 * scale**2, 3, padding=1)
            self.conv_out = nn.Conv2d(32, 1, 3, padding=1)
        def _rb(self):
            return nn.Sequential(
                nn.Conv2d(32,32,3,padding=1), nn.ReLU(True), nn.Conv2d(32,32,3,padding=1))
        def forward(self, x):
            x = F.relu(self.conv1(x)); r = x
            x = self.res(x) + r
            return self.conv_out(F.pixel_shuffle(self.conv_up(x), self.scale))

    @st.cache_resource
    def load_model():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = EDSR_Small(2).to(dev); m.eval()
        return m, dev

# ── CORE FUNCTIONS ─────────────────────────────────────

def normalize(arr):
    vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)


def psf_correct(data):
    kernel = Gaussian2DKernel(x_stddev=2)
    psf = kernel.array / kernel.array.sum()
    blurred = convolve2d(data, psf, mode="same")
    return np.clip(data + 0.5 * (data - blurred), 0, 1)


def neural_sr(data):
    if not TORCH_OK:
        return data
    model, dev = load_model()
    t = torch.tensor(data[None,None], dtype=torch.float32).to(dev)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    return np.clip(out,0,1)


def entangle(data, omega, fringe):
    return np.clip(np.sin(data * fringe * np.pi) * omega + data, 0, 1)

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v5")
    uploaded = st.file_uploader("Upload", type=["fits","png","jpg","jpeg"])
    st.caption("Full physics pipeline active")

# ── TOP CONTROLS ───────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    omega = st.slider("Ω Entanglement", 0.05, 0.5, 0.2)
with col2:
    fringe = st.slider("Fringe Scale", 10, 80, 40)
with col3:
    brightness = st.slider("Brightness", 0.5, 3.0, 1.2)

st.markdown("---")

# ── MAIN PIPELINE ──────────────────────────────────────
if uploaded:
    ext = uploaded.name.split(".")[-1]
    data = uploaded.read()

    if ext == "fits":
        with fits.open(io.BytesIO(data)) as h:
            raw = h[0].data.astype(np.float32)
    else:
        img = PILImage.open(io.BytesIO(data)).convert("L")
        raw = np.array(img, dtype=np.float32)

    norm = normalize(raw)
    psf = psf_correct(norm)
    sr = neural_sr(psf)
    sr = np.clip(sr * brightness, 0, 1)
    ent = entangle(sr, omega, fringe)

    c1, c2, c3, c4 = st.columns(4)

    def show(img, title):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(img, cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
        st.pyplot(fig)

    with c1: show(norm, "Input")
    with c2: show(psf, "PSF")
    with c3: show(sr, "Neural SR")
    with c4: show(ent, "Entangled")

    # DOWNLOADS
    st.markdown("---")
    st.subheader("Downloads")

    def save_png(arr):
        buf = io.BytesIO()
        plt.imsave(buf, arr, cmap="inferno")
        return buf.getvalue()

    st.download_button("Download Entangled PNG", save_png(ent), "entangled.png")

else:
    st.info("Upload a file — full pipeline runs instantly.")
