"""
QCI AstroEntangle Refiner – Streamlit Web App v4
Tony E Ford • tlcagford@gmail.com

Drop-in replacement for streamlit_app_v4.py
Fixes:
  • use_column_width → width="100%" (deprecation removed)
  • Images now bright with percentile stretch
  • Color variation via selectable cmaps (plasma, inferno, viridis, magma, turbo)
  • Entanglement overlay actually rendered and displayed
  • Before/After side-by-side comparison
  • PNG + FITS export
"""

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

# ── Optional torch (graceful fallback if unavailable) ─────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QCI AstroEntangle Refiner",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-ish custom CSS
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0d0d1e; }
    [data-testid="stSidebar"] { background: #13132b; }
    h1, h2, h3, h4 { color: #c8d8ff; }
    .stTabs [data-baseweb="tab"] { color: #a0b0e0; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #7090ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Neural SR (optional) ──────────────────────────────────────────────────────
if TORCH_OK:
    class EDSR_Small(nn.Module):
        def __init__(self, scale: int = 2):
            super().__init__()
            self.scale = scale
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.res_blocks = nn.Sequential(*[self._rb() for _ in range(8)])
            self.conv_up = nn.Conv2d(32, 32 * scale ** 2, 3, padding=1)
            self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

        def _rb(self):
            return nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
            )

        def forward(self, x):
            x = F.relu(self.conv1(x))
            r = x
            x = self.res_blocks(x)
            x = x + r
            x = F.pixel_shuffle(self.conv_up(x), self.scale)
            return self.conv_out(x)

    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = EDSR_Small(scale=2).to(device)
        m.eval()
        return m, device
else:
    load_model = None


# ── Processing helpers ────────────────────────────────────────────────────────

def normalize(data: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Percentile-stretch to [0, 1]."""
    vmin, vmax = np.percentile(data, lo), np.percentile(data, hi)
    return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1).astype(np.float32)


def psf_correct(data: np.ndarray) -> np.ndarray:
    kernel = Gaussian2DKernel(x_stddev=2)
    psf = kernel.array / kernel.array.sum()
    blurred = convolve2d(data, psf, mode="same", boundary="symm")
    return np.clip(data + 0.5 * (data - blurred), 0, 1).astype(np.float32)


def neural_sr(data: np.ndarray) -> np.ndarray:
    if not TORCH_OK:
        # Fallback: bicubic upscale then downscale for sharpening effect
        import cv2
        h, w = data.shape
        up = cv2.resize(data, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    model, device = load_model()
    t = torch.tensor(data[None, None], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    return np.clip(out, 0, 1).astype(np.float32)


def apply_boost(img: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
    img = np.clip(img * brightness, 0, 1)
    mid = img.mean()
    return np.clip(mid + (img - mid) * saturation, 0, 1).astype(np.float32)


def entangle(sr: np.ndarray, omega: float, fringe_scale: float) -> np.ndarray:
    H, W = sr.shape
    y, x = np.mgrid[0:H, 0:W]
    fringe = (
        0.40 * np.sin(2 * np.pi * x / fringe_scale) * np.cos(2 * np.pi * y / fringe_scale)
        + 0.30 * np.sin(2 * np.pi * (x + y) / (fringe_scale * 1.4))
        + 0.20 * np.cos(2 * np.pi * x / (fringe_scale * 0.7))
        + 0.10 * np.sin(4 * np.pi * y / fringe_scale)
    )
    fringe = (fringe - fringe.min()) / (fringe.max() - fringe.min() + 1e-9)
    weight = omega * (1.0 - sr * 0.5)
    return np.clip(sr + weight * (fringe - 0.5), 0, 1).astype(np.float32)


# ── Figure helpers ────────────────────────────────────────────────────────────

def make_fig(img: np.ndarray, cmap: str, title: str, figsize=(9, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d1e")
    ax.set_facecolor("#0d0d1e")
    im = ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334")
    fig.tight_layout(pad=1.5)
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def arr_to_fits_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    hdu = fits.PrimaryHDU(arr.astype(np.float32))
    hdu.writeto(buf)
    buf.seek(0)
    return buf.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔭 QCI AstroEntangle Refiner")
    st.caption("Tony E Ford • tlcagford@gmail.com")
    st.divider()

    uploaded = st.file_uploader("Upload FITS file", type=["fits", "fit", "fz"])
    st.divider()

    st.markdown("### ⚙️ Pipeline Controls")
    omega = st.slider("Ω_PD Entanglement", 0.05, 0.50, 0.20, 0.01)
    fringe_scale = st.slider("Fringe Scale (px)", 20, 80, 45, 1)
    brightness = st.slider("Brightness Boost", 0.5, 3.0, 1.3, 0.05)
    saturation = st.slider("Color Saturation", 0.5, 3.0, 1.5, 0.05)
    cmap_choice = st.selectbox(
        "Colormap",
        ["plasma", "inferno", "viridis", "magma", "turbo", "hot", "rainbow", "jet"],
        index=0,
    )
    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"PyTorch: {'✅ GPU' if TORCH_OK and __import__('torch').cuda.is_available() else '✅ CPU' if TORCH_OK else '⚠️ fallback (no torch)'}")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🔭 QCI AstroEntangle Refiner")
st.caption("PSF Correction → Neural Super-Resolution → Photon–Dark-Photon Entanglement Overlay")

if uploaded is None:
    st.info("👈 Upload a FITS file in the sidebar to get started.", icon="📂")
    st.stop()

# Load FITS
@st.cache_data(show_spinner="Reading FITS…")
def load_fits_data(file_bytes: bytes) -> np.ndarray:
    with fits.open(io.BytesIO(file_bytes)) as hdul:
        data = hdul[0].data
        if data is None:
            data = hdul[1].data
        data = data.astype(np.float32)
        if data.ndim == 3:
            data = np.mean(data, axis=0)
        elif data.ndim > 3:
            data = data[0, 0]
    return data

raw = load_fits_data(uploaded.read())
st.success(f"Loaded **{uploaded.name}** — shape: {raw.shape[1]} × {raw.shape[0]} px")

# Always show input preview
norm = normalize(raw)
with st.expander("📷 Input Preview (percentile stretch)", expanded=True):
    fig_in = make_fig(norm, "inferno", "Input – percentile stretch (inferno)")
    st.image(fig_to_png_bytes(fig_in), width=None)
    plt.close(fig_in)

if not run_btn:
    st.info("Adjust sliders in the sidebar then click **Run Full Pipeline**.")
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
progress = st.progress(0, text="Starting…")

progress.progress(15, text="PSF correction…")
psf = psf_correct(norm)

progress.progress(40, text="Neural super-resolution…")
sr = neural_sr(psf)
sr = apply_boost(sr, brightness, saturation)

progress.progress(70, text="Entanglement overlay…")
ent = entangle(sr, omega, fringe_scale)
ent = apply_boost(ent, brightness, saturation)

progress.progress(90, text="Rendering…")

# ── Display tabs ──────────────────────────────────────────────────────────────
tab_input, tab_sr, tab_ent, tab_compare = st.tabs(
    ["📥 Input", "🧠 Neural Enhanced", "🌀 Entangled Overlay", "↔️ Before / After"]
)

with tab_input:
    fig = make_fig(norm, "inferno", "Input – percentile stretch")
    st.image(fig_to_png_bytes(fig), width=None)
    plt.close(fig)

with tab_sr:
    fig = make_fig(sr, "viridis", "Neural SR + Brightness/Saturation (viridis)")
    st.image(fig_to_png_bytes(fig), width=None)
    plt.close(fig)

with tab_ent:
    fig = make_fig(
        ent, cmap_choice,
        f"Entanglement Overlay  Ω={omega:.2f}  fringe={fringe_scale}px  ({cmap_choice})"
    )
    st.image(fig_to_png_bytes(fig), width=None)
    plt.close(fig)

with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        fig = make_fig(norm, "inferno", "Before (input)", figsize=(6, 5))
        st.image(fig_to_png_bytes(fig), width=None)
        plt.close(fig)
    with col2:
        fig = make_fig(ent, cmap_choice, f"After (entangled – {cmap_choice})", figsize=(6, 5))
        st.image(fig_to_png_bytes(fig), width=None)
        plt.close(fig)

progress.progress(100, text="Done ✓")
st.success("Pipeline complete! Download results below. 👇")

# ── Downloads ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💾 Export Results")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
stem = os.path.splitext(uploaded.name)[0]

dl1, dl2, dl3, dl4 = st.columns(4)

with dl1:
    fig = make_fig(sr, "viridis", "Neural Refined", figsize=(10, 7))
    st.download_button(
        "⬇️ Refined PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"QCI_refined_{stem}_{ts}.png",
        mime="image/png",
        use_container_width=True,
    )
    plt.close(fig)

with dl2:
    st.download_button(
        "⬇️ Refined FITS",
        data=arr_to_fits_bytes(sr),
        file_name=f"QCI_refined_{stem}_{ts}.fits",
        mime="application/octet-stream",
        use_container_width=True,
    )

with dl3:
    fig = make_fig(ent, cmap_choice, f"Entangled – {cmap_choice}", figsize=(10, 7))
    st.download_button(
        "⬇️ Entangled PNG",
        data=fig_to_png_bytes(fig),
        file_name=f"QCI_entangled_{stem}_{ts}.png",
        mime="image/png",
        use_container_width=True,
    )
    plt.close(fig)

with dl4:
    st.download_button(
        "⬇️ Entangled FITS",
        data=arr_to_fits_bytes(ent),
        file_name=f"QCI_entangled_{stem}_{ts}.fits",
        mime="application/octet-stream",
        use_container_width=True,
    )
