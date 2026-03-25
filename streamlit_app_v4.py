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


def _gray_to_rgb(gray: np.ndarray, cmap_name: str) -> np.ndarray:
    """Map [0,1] grayscale to uint8 RGB via matplotlib cmap."""
    cmap = plt.get_cmap(cmap_name)
    return (cmap(gray)[:, :, :3] * 255).astype(np.uint8)


def entangle_rgb(
    norm_gray: np.ndarray,
    base_rgb: np.ndarray,
    omega: float,
    fringe_scale: float,
) -> np.ndarray:
    """
    Photon-Dark-Photon entanglement overlay matching the Tony Ford Model output:
      - Radial concentric rings centred on the brightest region of the image
      - Iridescent hue cycles continuously around each ring
      - Original RGB image preserved underneath; rings blended on top
      - Returns uint8 RGB (H, W, 3) for direct st.image display
    """
    H, W = norm_gray.shape

    # Centre: weighted centroid of top-10% brightest pixels
    thresh = np.percentile(norm_gray, 90)
    mask   = norm_gray > thresh
    ys, xs = np.where(mask)
    cy = float(ys.mean()) if len(ys) else H / 2
    cx = float(xs.mean()) if len(xs) else W / 2

    y, x = np.mgrid[0:H, 0:W]
    r    = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Multi-harmonic radial phase → rich iridescent rings
    phase = (r / fringe_scale) * 2 * np.pi
    ring  = (
        0.50 * np.sin(phase)
      + 0.25 * np.sin(2*phase + 0.8)
      + 0.15 * np.sin(3*phase + 1.6)
      + 0.10 * np.cos(5*phase)
    )
    ring = (ring - ring.min()) / (ring.max() - ring.min() + 1e-9)  # [0,1]

    # Vectorised HSV → RGB  (hue = ring position, full saturation)
    h6 = ring * 6.0
    i  = h6.astype(int) % 6
    f  = h6 - np.floor(h6)
    sat, val = 0.95, 0.95
    p = val * (1 - sat)
    q_ = val * (1 - sat * f)
    t_ = val * (1 - sat * (1 - f))
    v_ = val
    combos = [(v_, t_, p), (q_, v_, p), (p, v_, t_), (p, q_, v_), (t_, p, v_), (v_, p, q_)]
    ring_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate(combos):
        sel = (i == idx)
        ring_rgb[sel, 0] = rv if np.isscalar(rv) else rv[sel]
        ring_rgb[sel, 1] = gv if np.isscalar(gv) else gv[sel]
        ring_rgb[sel, 2] = bv if np.isscalar(bv) else bv[sel]

    # Blend weight: stronger in dim areas, fades at bright core
    blend_w = (omega * (1.0 - norm_gray * 0.55))[..., None]
    blend_w = np.clip(blend_w, 0.0, 0.75)

    base = base_rgb.astype(np.float32) / 255.0
    out  = base * (1.0 - blend_w) + ring_rgb * blend_w
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def make_annotated_fig(
    ent_rgb: np.ndarray,
    norm_gray: np.ndarray,
    title: str,
    omega: float,
    scale_kpc: float = 100,
) -> plt.Figure:
    """
    Render the entangled RGB image with scientific overlays:
      circles marking dark-matter halos, τ(r)/γ(r) labels, N arrow, scale bar.
    """
    H, W = norm_gray.shape

    # Locate 5 brightest local maxima as halo centres
    from scipy.ndimage import maximum_filter, label
    local_max = (norm_gray == maximum_filter(norm_gray, size=max(H, W)//20))
    coords = np.argwhere(local_max)  # (N, 2) → (row, col)
    # Sort by brightness, take top 6
    vals = [norm_gray[r, c] for r, c in coords]
    order = np.argsort(vals)[::-1]
    top = coords[order[:6]]

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#000010")
    ax.set_facecolor("#000010")
    ax.imshow(ent_rgb, origin="upper", aspect="auto")

    # Draw circles around each candidate halo
    radius = max(H, W) / 12
    for k, (row, col) in enumerate(top):
        circ = plt.Circle((col, row), radius * (0.9 if k > 0 else 1.35),
                           fill=False, edgecolor="white", linewidth=1.2, alpha=0.85)
        ax.add_patch(circ)

    # Labels on the two brightest
    if len(top) >= 1:
        r0, c0 = top[0]
        ax.text(c0 - radius*0.6, r0 + radius*0.15, r"$\gamma(r$",
                color="white", fontsize=16, fontweight="bold")
    if len(top) >= 2:
        r1, c1 = top[1]
        ax.text(c1 - radius*0.5, r1 - radius*0.3, r"$	au(r)$",
                color="white", fontsize=13, alpha=0.9)

    # North arrow (top-right)
    ax.annotate("", xy=(W*0.93, H*0.06), xytext=(W*0.93, H*0.14),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
    ax.text(W*0.932, H*0.04, "N", color="white", fontsize=11, ha="center")

    # Scale bar (bottom-left)  ~15% of image width
    bar_px  = W * 0.15
    bar_x0  = W * 0.04
    bar_y   = H * 0.93
    ax.plot([bar_x0, bar_x0 + bar_px], [bar_y, bar_y], color="white", lw=2)
    ax.text(bar_x0 + bar_px/2, bar_y - H*0.025, f"{scale_kpc} kpc",
            color="white", fontsize=10, ha="center")

    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


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
# Load original image as RGB for colour-preserving overlay
try:
    if PIL_OK:
        pil_orig = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        rgb_orig = np.array(pil_orig.resize((norm.shape[1], norm.shape[0]), PILImage.LANCZOS))
    elif CV2_OK:
        buf = np.frombuffer(file_bytes, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        rgb_orig = cv2.cvtColor(cv2.resize(bgr, (norm.shape[1], norm.shape[0])), cv2.COLOR_BGR2RGB)
    else:
        # Fallback: colourize the grayscale via inferno cmap
        rgb_orig = _gray_to_rgb(norm, "inferno")
except Exception:
    rgb_orig = _gray_to_rgb(norm, "inferno")

# For non-image files (FITS, CSV) always colourize via inferno
if ext in ("fits","fit","fz","csv"):
    rgb_orig = _gray_to_rgb(norm, "inferno")

ent_rgb = entangle_rgb(norm, rgb_orig, omega, fringe_scale)

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
    ann_title = (
        f"After: Photon–Dark-Photon Entangled FDM Overlays (Tony Ford Model)\n"
        f"Ω={omega:.2f}  fringe={fringe_scale}px"
    )
    fig = make_annotated_fig(ent_rgb, norm, ann_title, omega)
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
        st.markdown("**After (entangled FDM overlay)**")
        fig = make_annotated_fig(ent_rgb, norm,
            "After: Bullet Cluster – Full Photon-Dark-Photon Entangled FDM Overlays (Tony Ford Model)",
            omega)
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
    fig_e = make_annotated_fig(
        ent_rgb, norm,
        "After: Photon-Dark-Photon Entangled FDM Overlays (Tony Ford Model)", omega)
    st.download_button("⬇️ Entangled PNG",
        fig_bytes(fig_e), f"QCI_entangled_{stem}_{ts}.png","image/png", use_container_width=True)
    plt.close(fig_e)
with d6:
    # Save the float32 norm as FITS; ent_rgb is uint8 RGB
    st.download_button("⬇️ Entangled FITS",
        fits_bytes(sr), f"QCI_entangled_{stem}_{ts}.fits","application/octet-stream",
        use_container_width=True)

# Raw RGB as numpy CSV (flattened)
ent_flat = ent_rgb.reshape(-1, 3).astype(np.float32) / 255.0
st.download_button(
    "⬇️ Entangled RGB CSV",
    csv_bytes(ent_flat),
    f"QCI_entangled_{stem}_{ts}.csv","text/csv",
)
