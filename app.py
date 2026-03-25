# QCI AstroEntangle Refiner – v14 FINAL WORKING
# COMPLETE REWRITE: Fixed display, proper soliton physics, visible outputs

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from astropy.io import fits
from scipy.ndimage import gaussian_filter, sobel, maximum_filter
from scipy.signal import convolve2d
from astropy.convolution import Gaussian2DKernel
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI AstroEntangle Refiner v14", page_icon="🔭")

# Light blue interface
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e3f2fd; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #0288d1; }
.stTitle, h1, h2, h3 { color: #01579b; }
[data-testid="stMetricValue"] { color: #01579b; }
</style>
""", unsafe_allow_html=True)

# ── PHYSICS FUNCTIONS ─────────────────────────────────────────────

def load_image(uploaded_file):
    """Load image from uploaded file"""
    ext = uploaded_file.name.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    if ext == "fits":
        with fits.open(io.BytesIO(data_bytes)) as h:
            img = h[0].data.astype(np.float32)
            if len(img.shape) > 2:
                img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
    else:
        img = PILImage.open(io.BytesIO(data_bytes)).convert("L")
        img = np.array(img, dtype=np.float32)
    
    # Normalize to [0,1]
    img = np.nan_to_num(img, nan=0.0)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    return img


def create_soliton_core(size, fringe, center_ratio=0.5):
    """
    Create FDM soliton core with [sin(kr)/kr]² profile
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w/2), int(h/2)
    
    # Distance from center normalized
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
    
    # Soliton scale depends on fringe
    r_s = 0.25 * (50.0 / max(fringe, 1))
    
    # k parameter
    k = np.pi / r_s
    kr = k * r
    
    # Soliton profile: ρ(r) = [sin(kr)/(kr)]²
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    # Normalize
    soliton = (soliton - soliton.min()) / (soliton.max() - soliton.min())
    
    # Add smoothing
    soliton = gaussian_filter(soliton, sigma=3)
    
    return soliton


def create_fringe_pattern(size, fringe, soliton):
    """
    Create visible dark photon fringe patterns
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w/2), int(h/2)
    
    # Normalized coordinates
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
    theta = np.arctan2(y - cy, x - cx)
    
    # Wave number (scaled for visibility)
    k = fringe / 15.0
    
    # Multiple wave patterns
    # 1. Radial waves
    radial = np.sin(k * 2 * np.pi * r * 3)
    
    # 2. Spiral waves
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    
    # 3. Angular waves
    angular = np.sin(k * 3 * theta)
    
    # Combine patterns
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    # Modulate by soliton envelope
    pattern = pattern * soliton
    
    # Normalize to [0,1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    
    return pattern


def create_dark_matter_map(size, data, fringe, soliton):
    """
    Create dark matter density map with halos
    """
    h, w = size
    
    # Smooth data for large-scale structure
    smoothed = gaussian_filter(data, sigma=10)
    
    # Density from gradients
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    density = np.sqrt(grad_x**2 + grad_y**2)
    
    # Combine with soliton core
    dm_map = soliton * 0.6 + density * 0.4
    
    # Add small halos at peaks
    try:
        # Find local maxima
        neighborhood = np.ones((15, 15))
        local_max = maximum_filter(smoothed, footprint=neighborhood) == smoothed
        peaks = np.argwhere(local_max & (smoothed > np.percentile(smoothed, 85)))
        
        y_grid, x_grid = np.ogrid[:h, :w]
        for yc, xc in peaks[:20]:
            r2 = (x_grid - xc)**2 + (y_grid - yc)**2
            halo = np.exp(-r2 / (2 * 100))
            dm_map += halo * smoothed[yc, xc] * 0.3
    except:
        pass
    
    # Normalize
    dm_map = (dm_map - dm_map.min()) / (dm_map.max() - dm_map.min() + 1e-9)
    
    return dm_map


def apply_pdp_entanglement(data, omega, fringe, brightness=1.2):
    """
    Apply photon-dark-photon entanglement with soliton physics
    """
    # Get image size
    h, w = data.shape
    
    # Create soliton core
    soliton = create_soliton_core((h, w), fringe)
    
    # Create fringe patterns
    fringe_pattern = create_fringe_pattern((h, w), fringe, soliton)
    
    # Create dark matter map
    dm_map = create_dark_matter_map((h, w), data, fringe, soliton)
    
    # Mixing strength
    mix = omega * 0.8
    
    # Entangle components
    result = data * (1 - mix * 0.3)
    result = result + fringe_pattern * mix * 0.5
    result = result + dm_map * mix * 0.3
    result = result + soliton * mix * 0.4
    
    # Apply brightness
    result = result * brightness
    
    # Final clamp
    result = np.clip(result, 0, 1)
    
    return result, soliton, fringe_pattern, dm_map


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v14")
    st.markdown("### Photon-Dark-Photon Entangled FDM")
    st.markdown("*Working Display Version*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    st.markdown("### Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    st.markdown("---")
    st.caption("Tony Ford Model | v14 - Fully Working")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled FDM with Soliton Core Waves*")
st.markdown("---")

if uploaded is not None:
    # Load and process
    with st.spinner("Processing image..."):
        # Load
        img = load_image(uploaded)
        
        # Simple preprocessing
        img_blur = gaussian_filter(img, sigma=1)
        img_sharp = img + (img - img_blur) * 0.5
        img_sharp = np.clip(img_sharp, 0, 1)
        
        # Apply PDP entanglement
        result, soliton, fringe_pattern, dm_map = apply_pdp_entanglement(
            img_sharp, omega, fringe, brightness
        )
    
    # Display parameters
    st.success(f"✅ Processing Complete | Ω={omega:.2f} | Fringe={fringe}")
    
    # ── DISPLAY IMAGES ─────────────────────────────────────────────
    st.markdown("### 📊 Results")
    
    # Row 1: Main pipeline
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img, cmap="gray")
        ax.set_title("Original", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(img_sharp, cmap="inferno")
        ax.set_title("Enhanced", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col3:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(result, cmap="inferno")
        ax.set_title("PDP Entangled", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col4:
        # RGB composite
        rgb = np.stack([
            result,
            result * 0.4 + fringe_pattern * 0.6,
            result * 0.3 + dm_map * 0.7
        ], axis=-1)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(np.clip(rgb, 0, 1))
        ax.set_title("RGB Composite", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    st.markdown("### 🌌 FDM Physics Components")
    
    # Row 2: Physics components
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(soliton, cmap="hot")
        ax.set_title("FDM Soliton Core", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        
        # Radial profile
        h, w = soliton.shape
        cx, cy = w//2, h//2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        radii = np.arange(0, min(h, w)//2, 3)
        profile = []
        for rad in radii:
            mask = (r >= rad) & (r < rad + 3)
            if np.any(mask):
                profile.append(np.mean(soliton[mask]))
            else:
                profile.append(0)
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(radii[:len(profile)], profile, 'r-', linewidth=2)
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Density")
        ax.set_title("Soliton Profile [sin(kr)/kr]²")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(fringe_pattern, cmap="plasma")
        ax.set_title(f"Dark Photon Field (fringe={fringe})", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        
        # Fringe detail zoom
        h, w = fringe_pattern.shape
        zh, zw = h//2 - 60, w//2 - 60
        zoom = fringe_pattern[zh:zh+120, zw:zw+120]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(zoom, cmap="plasma")
        ax.set_title("Fringe Detail (Zoom)", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dm_map, cmap="viridis")
        ax.set_title("Dark Matter Density", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
        
        # Power spectrum
        fft = np.fft.fft2(fringe_pattern)
        fft_shift = np.fft.fftshift(fft)
        power = np.log10(np.abs(fft_shift) + 1)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(power, cmap="hot")
        ax.set_title("Frequency Spectrum", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    st.markdown("### 📊 Before vs After")
    
    # Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis("off")
    
    ax2.imshow(result, cmap="inferno")
    ax2.set_title(f"PDP Entangled (Ω={omega:.2f}, Fringe={fringe})", fontsize=14)
    ax2.axis("off")
    
    st.pyplot(fig)
    plt.close(fig)
    
    # ── METRICS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Soliton Peak", f"{soliton.max():.3f}")
    
    with col_m2:
        st.metric("Fringe Contrast", f"{fringe_pattern.std():.3f}")
    
    with col_m3:
        st.metric("DM Density", f"{dm_map.mean():.3f}")
    
    with col_m4:
        mixing = omega * 0.8
        st.metric("PDP Mixing", f"{mixing:.2f}")
    
    # ── DOWNLOAD ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        buf = io.BytesIO()
        plt.imsave(buf, result, cmap="inferno")
        st.download_button("📸 PDP Image", buf.getvalue(), "pdp_result.png")
    
    with col_d2:
        buf = io.BytesIO()
        plt.imsave(buf, soliton, cmap="hot")
        st.download_button("⭐ Soliton Core", buf.getvalue(), "soliton.png")
    
    with col_d3:
        buf = io.BytesIO()
        plt.imsave(buf, fringe_pattern, cmap="plasma")
        st.download_button("🌊 Fringe Pattern", buf.getvalue(), "fringe.png")

else:
    st.info("✨ **Upload an image to start**\n\n"
            "This app applies Photon-Dark-Photon Entanglement with FDM Soliton Physics:\n\n"
            "• **FDM Soliton Core**: Ground state of fuzzy dark matter [sin(kr)/kr]²\n"
            "• **Dark Photon Fringes**: Wave interference patterns\n"
            "• **Dark Matter Map**: Substructure from gravitational potential\n"
            "• **RGB Composite**: Visual separation of components\n\n"
            "*Try Ω=0.7, Fringe=65 for optimal visibility*")
    
    # Show example expectations
    st.markdown("---")
    st.markdown("### 📋 Expected Output")
    st.markdown("""
    - **Soliton Core**: Bright central peak with [sin(kr)/kr]² profile
    - **Fringe Patterns**: Visible wave interference rings
    - **Dark Matter Halos**: Clumpy substructure around peaks
    - **Enhanced Image**: Original structure + dark sector effects
    """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v14** | Fully Working Display | Tony Ford Model")
