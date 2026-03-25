"""
QCI AstroEntangle Refiner – Streamlit Web App v4
Tony E Ford • tlcagford@gmail.com

Universal image input: FITS · JPG · PNG · TIFF · BMP · WEBP · CSV · DICOM/X-Ray
Full pipeline: Percentile Stretch → PSF Correction → Neural SR → Entanglement Overlay
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

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import pydicom
    DICOM_OK = True
except ImportError:
    DICOM_OK = False


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QCI AstroEntangle Refiner",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0b1a; }
[data-testid="stSidebar"]          { background: #10102a; }
h1,h2,h3,h4                        { color: #c8d8ff; }
.stTabs [data-baseweb="tab"]        { color: #9aacdd; }
.stTabs [aria-selected="true"]      { color: #fff; border-bottom: 2px solid #6688ff; }
</style>
""", unsafe_allow_html=True)


# ── Neural SR ─────────────────────────────────────────────────────────────────
if TORCH_OK:
    class EDSR_Small(nn.Module):
        def __init__(self, scale=2):
            super().__init__()
            self.scale   = scale
            self.conv1   = nn.Conv2d(1, 32, 3, padding=1)
            self.res     = nn.Sequential(*[self._rb() for _ in range(8)])
            self.conv_up = nn.Conv2d(32, 32 * scale**2, 3, padding=1)
            self.conv_out= nn.Conv2d(32, 1, 3, padding=1)
        def _rb(self):
            return nn.Sequential(
                nn.Conv2d(32,32,3,padding=1), nn.ReLU(True), nn.Conv2d(32,32,3,padding=1))
        def forward(self, x):
            x = F.relu(self.conv1(x)); r = x
            x = self.res(x) + r
            return self.conv_out(F.pixel_shuffle(self.conv_up(x), self.scale))

    @st.cache_resource(show_spinner=False)
    def load_sr_model():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = EDSR_Small(2).to(dev); m.eval()
        return m, dev


# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_TYPES = [
    "fits","fit","fz",
    "jpg","jpeg","png","tif","tiff","bmp","webp",
    "csv",
    "dcm",
]
FILE_LABELS = {
    "fits":"🌌 FITS","fit":"🌌 FITS","fz":"🌌 FITS",
    "jpg":"🖼️ Image","jpeg":"🖼️ Image","png":"🖼️ Image",
    "tif":"🖼️ Image","tiff":"🖼️ Image","bmp":"🖼️ Image","webp":"🖼️ Image",
    "csv":"📊 CSV","dcm":"🩻 DICOM",
}


# ── File loaders ──────────────────────────────────────────────────────────────

def load_fits_bytes(data: bytes) -> np.ndarray:
    with fits.open(io.BytesIO(data)) as h:
        arr = h[0].data
        if arr is None:
            arr = h[1].data
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)
        elif arr.ndim > 3:
            arr = arr[0, 0]
    return arr


def load_image_bytes(data: bytes) -> np.ndarray:
    if PIL_OK:
        img = PILImage.open(io.BytesIO(data)).convert("L")
        return np.array(img, dtype=np.float32)
    if CV2_OK:
        buf = np.frombuffer(data, np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    raise RuntimeError("Install Pillow or opencv-python-headless to read images.")


def load_csv_bytes(data: bytes) -> np.ndarray:
    if not PANDAS_OK:
        raise RuntimeError("Install pandas to read CSV files.")
    df = pd.read_csv(io.BytesIO(data), header=None)
    return df.values.astype(np.float32)


def load_dicom_bytes(data: bytes) -> np.ndarray:
    if DICOM_OK:
        ds  = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        return arr
    if PIL_OK:
        img = PILImage.open(io.BytesIO(data)).convert("L")
        return np.array(img, dtype=np.float32)
    raise RuntimeError("Add pydicom to requirements.txt for DICOM support.")


@st.cache_data(show_spinner="Reading file…")
def load_any(file_bytes: bytes, ext: str) -> np.ndarray:
    ext = ext.lower().lstrip(".")
    if ext in ("fits","fit","fz"):
        return load_fits_bytes(file_bytes)
    if ext in ("jpg","jpeg","png","tif","tiff","bmp","webp"):
        return load_image_bytes(file_bytes)
    if ext == "csv":
        return load_csv_bytes(file_bytes)
    if ext == "dcm":
        return load_dicom_bytes(file_bytes)
    raise ValueError(f"Unsupported file type: .{ext}")


# ── Processing ────────────────────────────────────────────────────────────────

def normalize(arr: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
    vmin, vmax = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1).astype(np.float32)


def psf_correct(data: np.ndarray) -> np.ndarray:
    kernel = Gaussian2DKernel(x_stddev=2)
    psf    = kernel.array / kernel.array.sum()
    blurred = convolve2d(data, psf, mode="same", boundary="symm")
    return np.clip(data + 0.5 * (data - blurred), 0, 1).astype(np.float32)


MAX_SR_DIM = 1024  # cap to avoid OOM on large images

def _resize_arr(arr, w, h):
    if CV2_OK:
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    if PIL_OK:
        pil = PILImage.fromarray((arr * 255).astype(np.uint8))
        pil = pil.resize((w, h), PILImage.LANCZOS)
        return np.array(pil, dtype=np.float32) / 255.0
    return arr

def neural_sr(data: np.ndarray) -> np.ndarray:
    H, W = data.shape
    scale_down = max(H, W) > MAX_SR_DIM
    if scale_down:
        factor = MAX_SR_DIM / max(H, W)
        sh, sw = int(H * factor), int(W * factor)
        data_in = _resize_arr(data, sw, sh)
    else:
        data_in = data

    if TORCH_OK:
        model, dev = load_sr_model()
        t = torch.tensor(data_in[None, None], dtype=torch.float32).to(dev)
        with torch.no_grad():
            out = model(t).squeeze().cpu().numpy()
        out = np.clip(out, 0, 1).astype(np.float32)
    elif CV2_OK:
        u8 = (data_in * 255).astype(np.uint8)
        bl = cv2.GaussianBlur(u8, (0, 0), 3)
        out = cv2.addWeighted(u8, 1.5, bl, -0.5, 0).astype(np.float32) / 255.0
    else:
        out = data_in.copy()

    if scale_down:
        if CV2_OK:
            out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
        elif PIL_OK:
            pil = PILImage.fromarray((out * 255).astype(np.uint8))
            pil = pil.resize((W, H), PILImage.LANCZOS)
            out = np.array(pil, dtype=np.float32) / 255.0

    return np.clip(out, 0, 1).astype(np.float32)


def boost(img: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
    img = np.clip(img * brightness, 0, 1)
    mid = img.mean()
    return np.clip(mid + (img - mid) * saturation, 0, 1).astype(np.float32)


def entangle(sr: np.ndarray, omega: float, fringe_scale: float) -> np.ndarray:
    H, W = sr.shape
    y, x = np.mgrid[0:H, 0:W]
    fringe = (
        0.40 * np.sin(2*np.pi*x / fringe_scale) * np.cos(2*np.pi*y / fringe_scale)
      + 0.30 * np.sin(2*np.pi*(x+y) / (fringe_scale*1.4))
      + 0.20 * np.cos(2*np.pi*x / (fringe_scale*0.7))
      + 0.10 * np.sin(4*np.pi*y / fringe_scale)
    )
    fringe = (fringe - fringe.min()) / (fringe.max() - fringe.min() + 1e-9)
    weight = omega * (1.0 - sr * 0.5)
    return np.clip(sr + weight * (fringe - 0.5), 0, 1).astype(np.float32)


# ── Figure helpers ────────────────────────────────────────────────────────────

def make_fig(img: np.ndarray, cmap: str, title: str, figsize=(9,6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0b0b1a")
    ax.set_facecolor("#0b0b1a")
    im = ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest", aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334")
    fig.tight_layout(pad=1.5)
    return fig


def fig_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    buf.seek(0); return buf.read()


def fits_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    fits.PrimaryHDU(arr.astype(np.float32)).writeto(buf)
    buf.seek(0); return buf.read()


def csv_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, arr, delimiter=",", fmt="%.6f")
    buf.seek(0); return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔭 QCI AstroEntangle Refiner")
    st.caption("Tony E Ford • tlcagford@gmail.com")
    st.divider()

    st.markdown("### 📂 Drop Any Image File")
    st.caption(
        "**FITS** · **JPG** · **PNG** · **TIFF** · **BMP** · **WEBP**\n\n"
        "**CSV** (2-D numeric array) · **DICOM / DCM** (X-Ray)"
    )
    uploaded = st.file_uploader(
        "Upload file",
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ⚙️ Pipeline Controls")
    omega        = st.slider("Ω_PD  Entanglement",   0.05, 0.50, 0.20, 0.01)
    fringe_scale = st.slider("Fringe Scale (px)",      20,   80,  45,   1)
    brightness   = st.slider("Brightness Boost",      0.5,  3.0, 1.3, 0.05)
    saturation   = st.slider("Color Saturation",      0.5,  3.0, 1.5, 0.05)
    cmap_choice  = st.selectbox(
        "Overlay Colormap",
        ["plasma","inferno","viridis","magma","turbo","hot","rainbow","jet"],
    )
    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        f"Torch: {'✅ GPU' if TORCH_OK and torch.cuda.is_available() else '✅ CPU' if TORCH_OK else '⚠️ fallback'} · "
        f"PIL: {'✅' if PIL_OK else '⚠️'} · "
        f"DICOM: {'✅' if DICOM_OK else '⚠️ add pydicom'}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔭 QCI AstroEntangle Refiner")
st.caption(
    "**Universal input** → Percentile Stretch → PSF Correction → "
    "Neural Super-Resolution → Photon–Dark-Photon Entanglement Overlay"
)

# ── No file yet: show format cards ───────────────────────────────────────────
if uploaded is None:
    st.info(
        "👈 **Drop any image file in the sidebar** to begin.\n\n"
        "Supported: **FITS · JPG · PNG · TIFF · BMP · WEBP · CSV · DICOM/X-Ray**",
        icon="📂",
    )
    cards = [
        ("🌌","FITS / FIT / FZ","Astronomy raw data"),
        ("🖼️","JPG · PNG · TIFF · BMP · WEBP","Standard image formats"),
        ("📊","CSV","2-D numeric array → heatmap"),
        ("🩻","DCM / DICOM","Medical / X-Ray imaging"),
    ]
    cols = st.columns(4)
    for col, (icon, fmt, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div style="background:#16163a;border-radius:12px;padding:16px;text-align:center">'
                f'<div style="font-size:2.2rem">{icon}</div>'
                f'<div style="color:#c8d8ff;font-weight:700;margin:6px 0">{fmt}</div>'
                f'<div style="color:#888;font-size:0.82rem">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ── Load file ─────────────────────────────────────────────────────────────────
ext        = uploaded.name.rsplit(".", 1)[-1].lower()
file_label = FILE_LABELS.get(ext, f"📄 .{ext}")
file_bytes = uploaded.read()

try:
    raw = load_any(file_bytes, ext)
except Exception as e:
    st.error(f"❌ Could not load file: {e}")
    st.stop()

st.success(
    f"{file_label} — **{uploaded.name}** loaded  |  "
    f"Shape: {raw.shape[1]} × {raw.shape[0]} px  |  "
    f"Range: {raw.min():.2f} – {raw.max():.2f}"
)

norm = normalize(raw)

with st.expander("📷 Input Preview (auto-stretched)", expanded=True):
    fig = make_fig(norm, "inferno", f"Input – {uploaded.name}  (inferno, percentile stretch)")
    st.image(fig_bytes(fig), use_container_width=True)
    plt.close(fig)

if not run_btn:
    st.info("Adjust sliders then click **🚀 Run Full Pipeline** in the sidebar.")
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
bar = st.progress(0, text="Initialising…")

bar.progress(10, text="PSF correction…")
psf = psf_correct(norm)

bar.progress(35, text="Neural super-resolution…")
sr = neural_sr(psf)
sr = boost(sr, brightness, saturation)

bar.progress(65, text="Building entanglement overlay…")
ent = entangle(sr, omega, fringe_scale)
ent = boost(ent, brightness, saturation)

bar.progress(85, text="Rendering tabs…")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t_in, t_sr, t_ent, t_ba, t_raw = st.tabs([
    "📥 Input",
    "🧠 Neural Enhanced",
    "🌀 Entangled Overlay",
    "↔️ Before / After",
    "🔢 Raw Stats",
])

with t_in:
    fig = make_fig(norm, "inferno", "Input – percentile stretch (inferno)")
    st.image(fig_bytes(fig), use_container_width=True)
    plt.close(fig)

with t_sr:
    fig = make_fig(sr, "viridis",
        f"Neural SR  brightness={brightness:.1f}  saturation={saturation:.1f}  (viridis)")
    st.image(fig_bytes(fig), use_container_width=True)
    plt.close(fig)

with t_ent:
    fig = make_fig(ent, cmap_choice,
        f"Entanglement Overlay — Ω={omega:.2f}  fringe={fringe_scale}px  ({cmap_choice})")
    st.image(fig_bytes(fig), use_container_width=True)
    plt.close(fig)

with t_ba:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before (input)**")
        fig = make_fig(norm, "inferno", "Before", figsize=(6,5))
        st.image(fig_bytes(fig), use_container_width=True)
        plt.close(fig)
    with c2:
        st.markdown(f"**After (entangled – {cmap_choice})**")
        fig = make_fig(ent, cmap_choice, "After", figsize=(6,5))
        st.image(fig_bytes(fig), use_container_width=True)
        plt.close(fig)

with t_raw:
    st.markdown("#### Array statistics")
    ca, cb, cc, cd = st.columns(4)
    ca.metric("Min",   f"{raw.min():.4f}")
    cb.metric("Max",   f"{raw.max():.4f}")
    cc.metric("Mean",  f"{raw.mean():.4f}")
    cd.metric("Shape", f"{raw.shape[1]}×{raw.shape[0]}")
    if PANDAS_OK:
        st.dataframe(pd.DataFrame(raw).describe().round(4), use_container_width=True)

bar.progress(100, text="Done ✓")
st.success("Pipeline complete! Download results below. 👇")

# ── Downloads ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 💾 Download Results")
ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
stem = os.path.splitext(uploaded.name)[0]

d1,d2,d3,d4,d5,d6 = st.columns(6)

with d1:
    fig = make_fig(norm, "inferno", "Input", figsize=(10,7))
    st.download_button("⬇️ Input PNG",
        fig_bytes(fig), f"QCI_input_{stem}_{ts}.png","image/png", use_container_width=True)
    plt.close(fig)
with d2:
    st.download_button("⬇️ Input FITS",
        fits_bytes(norm), f"QCI_input_{stem}_{ts}.fits","application/octet-stream",
        use_container_width=True)
with d3:
    fig = make_fig(sr, "viridis", "Refined", figsize=(10,7))
    st.download_button("⬇️ Refined PNG",
        fig_bytes(fig), f"QCI_refined_{stem}_{ts}.png","image/png", use_container_width=True)
    plt.close(fig)
with d4:
    st.download_button("⬇️ Refined FITS",
        fits_bytes(sr), f"QCI_refined_{stem}_{ts}.fits","application/octet-stream",
        use_container_width=True)
with d5:
    fig = make_fig(ent, cmap_choice, "Entangled", figsize=(10,7))
    st.download_button("⬇️ Entangled PNG",
        fig_bytes(fig), f"QCI_entangled_{stem}_{ts}.png","image/png", use_container_width=True)
    plt.close(fig)
with d6:
    st.download_button("⬇️ Entangled FITS",
        fits_bytes(ent), f"QCI_entangled_{stem}_{ts}.fits","application/octet-stream",
        use_container_width=True)

st.download_button(
    "⬇️ Entangled CSV (raw numeric array)",
    csv_bytes(ent),
    f"QCI_entangled_{stem}_{ts}.csv","text/csv",
)
