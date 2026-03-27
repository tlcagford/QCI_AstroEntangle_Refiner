import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import io
import base64
from datetime import datetime

st.set_page_config(page_title="QCAUS", page_icon="🌌", layout="wide")

# ============================================================================
# FORMULAS
# ============================================================================

def fdm_soliton(r, k=1.0):
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)

def pdp_field(img, omega=0.5, fringe=1.0):
    fft_img = fft2(img)
    fft_shift = fftshift(fft_img)
    rows, cols = img.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe))
    mixed = fft_shift * mask
    return np.abs(ifft2(fftshift(mixed)))

def process(img, omega=0.5, fringe=1.0, k=1.0, alpha=0.8, beta=1.0):
    size = min(img.shape)
    r = np.linspace(0, 3, size)
    fdm_profile = fdm_soliton(r, k=k)
    fdm_2d = np.outer(fdm_profile, fdm_profile)
    fdm_resized = zoom(fdm_2d, (img.shape[0]/size, img.shape[1]/size))
    pdp = pdp_field(img, omega, fringe)
    enhanced = img + alpha * fdm_resized + beta * pdp
    return np.clip(enhanced, 0, 1), fdm_resized, pdp

def add_scale(ax, width, kpc=100, scale=0.1):
    bar_px = kpc / scale
    rect = Rectangle((50, width - 60), bar_px, 8, linewidth=2, edgecolor='white', facecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(50 + bar_px/2, width - 72, f"{kpc} kpc", color='white', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

def download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{name}" style="text-decoration:none;">📥 Download</a>'

def load_image(file):
    ext = file.name.split('.')[-1].lower()
    data = file.read()
    try:
        img = Image.open(io.BytesIO(data))
        arr = np.array(img)
        if arr.ndim == 3:
            arr = np.mean(arr[:, :, :3], axis=2)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr, f"{ext.upper()}: {arr.shape}"
    except:
        return None, "Error loading image"

def sample_image():
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img = np.exp(-R**2 / 1.5**2)
    img += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
    img += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
    return img / img.max()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    omega = st.slider("Ω (Entanglement)", 0.0, 1.0, 0.5, 0.01)
    fringe = st.slider("λ (Fringe)", 0.1, 3.0, 1.5, 0.05)
    k = st.slider("k (Soliton)", 0.5, 3.0, 1.0, 0.05)
    alpha = st.slider("α (FDM)", 0.0, 2.0, 0.8, 0.05)
    beta = st.slider("β (PDP)", 0.0, 2.0, 1.0, 0.05)
    
    src = st.radio("Image", ["Sample", "Upload"])
    
    img = None
    img_name = "galaxy"
    scale = 0.1
    
    if src == "Sample":
        img = sample_image()
        img_name = "galaxy_cluster"
    else:
        up = st.file_uploader("Upload", type=['jpg', 'png'])
        if up:
            img, info = load_image(up)
            if img is not None:
                st.success(info)
                img_name = up.name.split('.')[0]
                scale = 100.0 / img.shape[1]
    
    scale = st.number_input("kpc/pixel", value=scale, format="%.4f")

# ============================================================================
# MAIN
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")

if img is not None:
    enhanced, fdm, pdp = process(img, omega, fringe, k, alpha, beta)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"qcaus_{img_name}_{ts}"
    
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ω", f"{omega:.2f}")
    c2.metric("λ", f"{fringe:.2f}")
    c3.metric("k", f"{k:.2f}")
    c4.metric("Max P", f"{np.max(enhanced):.3f}")
    
    # Before/After Comparison
    st.subheader("🔬 Before / After Comparison")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.imshow(img, cmap='gray', origin='upper')
    ax1.set_title("Before: Standard View", fontsize=12)
    ax1.axis('off')
    add_scale(ax1, img.shape[1], scale=scale)
    
    ax2.imshow(enhanced, cmap='plasma', origin='upper')
    ax2.set_title(f"After: Quantum View\nα={alpha:.2f}, β={beta:.2f}", fontsize=12)
    ax2.axis('off')
    add_scale(ax2, enhanced.shape[1], scale=scale)
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(download(fig, f"{base}_comparison.png"), unsafe_allow_html=True)
    plt.close(fig)
    
    # Full-Spectrum Composite
    st.subheader("🌈 Full-Spectrum Composite")
    
    rgb = np.zeros((*img.shape, 3))
    rgb[..., 0] = img / (img.max() + 1e-8)
    fdm_norm = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
    pdp_norm = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
    rgb[..., 1] = fdm_norm * 0.9
    rgb[..., 2] = pdp_norm * 0.9
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.clip(rgb, 0, 1), origin='upper')
    ax.set_title("Full-Spectrum Composite\nRed: Visible | Green: Dark Matter | Blue: Dark Photons")
    ax.axis('off')
    add_scale(ax, rgb.shape[1], scale=scale)
    st.pyplot(fig)
    st.markdown(download(fig, f"{base}_composite.png"), unsafe_allow_html=True)
    plt.close(fig)
    
    # Individual Fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌌 FDM Soliton Field")
        fig, ax = plt.subplots(figsize=(6, 6))
        fdm_viz = np.zeros((*fdm.shape, 3))
        fdm_viz[..., 1] = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
        ax.imshow(fdm_viz, origin='upper')
        ax.set_title(f"ρ(r) = ρ₀ [sin({k:.2f}r)/({k:.2f}r)]²")
        ax.axis('off')
        add_scale(ax, fdm.shape[1], scale=scale)
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_fdm.png"), unsafe_allow_html=True)
        plt.close(fig)
    
    with col2:
        st.subheader("🌀 PDP Quantum Field")
        fig, ax = plt.subplots(figsize=(6, 6))
        pdp_viz = np.zeros((*pdp.shape, 3))
        pdp_viz[..., 2] = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
        ax.imshow(pdp_viz, origin='upper')
        ax.set_title(f"ℒ_mix = (ε/2) F_μν F'^μν\nΩ={omega:.2f}, λ={fringe:.2f}")
        ax.axis('off')
        add_scale(ax, pdp.shape[1], scale=scale)
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_pdp.png"), unsafe_allow_html=True)
        plt.close(fig)
    
    # Metrics
    st.subheader("📊 Quantum Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Max FDM", f"{np.max(fdm):.3f}")
    m2.metric("Max PDP", f"{np.max(pdp):.3f}")
    corr = np.corrcoef(fdm.flatten(), pdp.flatten())[0, 1]
    m3.metric("Correlation", f"{corr:.3f}")

else:
    st.info("👈 Select or upload an image to begin")

# ============================================================================
# ABOUT
# ============================================================================

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Verified Formulas
    
    | Project | Formula |
    |---------|---------|
    | **FDM Soliton** | ρ(r) = ρ₀ [sin(kr)/(kr)]² |
    | **PDP Mixing** | ℒ_mix = (ε/2) F_μν F'^μν |
    | **Quantum State** | |Ψ⟩ = |Ψ_astro⟩ + α|Ψ_FDM⟩ + β|Ψ_PDP⟩ |
    
    **Color Mapping:** 🔴 Visible | 🟢 Dark Matter | 🔵 Dark Photons
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>© 2026 Tony E. Ford</div>", unsafe_allow_html=True)
