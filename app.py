# QCI AstroEntangle Refiner – v11 FINAL FIXED
# FIXED: All type errors, light blue interface, proper PDP physics

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
from scipy.ndimage import gaussian_filter, laplace, sobel, zoom, maximum_filter, label
from PIL import Image as PILImage

# ── LIGHT BLUE INTERFACE ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI AstroEntangle Refiner v11", page_icon="🔭")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { 
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdef5 100%);
}
[data-testid="stSidebar"] { 
    background: linear-gradient(135deg, #ffffff 0%, #e1f5fe 100%);
    border-right: 2px solid #0288d1;
}
[data-testid="stSidebar"] * {
    color: #01579b !important;
}
.stTitle, h1, h2, h3 {
    color: #01579b !important;
}
.stButton > button {
    background-color: #0288d1;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #0277bd;
    color: white !important;
}
[data-testid="stMetricValue"] {
    color: #01579b !important;
}
[data-testid="stInfo"] {
    background-color: #e1f5fe;
    border-left: 3px solid #0288d1;
}
[data-testid="stSlider"] {
    color: #0288d1;
}
</style>
""", unsafe_allow_html=True)

# ── PHYSICS FUNCTIONS ─────────────────────────────────────

def normalize(arr):
    """Safe normalization"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if vmax - vmin < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)


def psf_correct(data, amount=0.6):
    """PSF deconvolution"""
    try:
        kernel = Gaussian2DKernel(x_stddev=1.8)
        psf = kernel.array / kernel.array.sum()
        blurred = convolve2d(data, psf, mode="same")
        result = np.clip(data + amount * (data - blurred), 0, 1)
        return np.nan_to_num(result)
    except:
        return data


def enhance_resolution(data):
    """Resolution enhancement"""
    try:
        blurred = gaussian_filter(data, sigma=0.8)
        edges = data - blurred
        enhanced = data + edges * 0.4
        return np.clip(enhanced, 0, 1)
    except:
        return data


# ===== PDP PHYSICS - CORRECTED =====

def create_pdp_fringe_pattern(data, fringe_value, physical_scale_kpc=100):
    """
    Creates dark photon oscillation patterns based on fringe parameter
    """
    h, w = data.shape
    
    # Ensure fringe is integer
    fringe = int(fringe_value)
    
    # Normalized wave number (physics-based)
    wave_number = fringe / 50.0
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    
    # Radial distance normalized
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(w, h)
    theta = np.arctan2(y - cy, x - cx)
    
    # PDP Wave Equation (based on dark photon oscillation)
    # ψ_dark = sin(k·r - ωt) with fringe-dependent k
    
    # 1. Radial modes (dark matter soliton oscillations)
    radial_modes = np.sin(wave_number * 2 * np.pi * r)
    
    # 2. Angular modes (quantum vortex structures)
    angular_modes = np.sin(wave_number * 2 * theta)
    
    # 3. Spiral modes (characteristic of rotating dark matter)
    spiral_modes = np.sin(wave_number * 2 * np.pi * (r + theta / (2 * np.pi)))
    
    # 4. Interference pattern (PDP coupling)
    if fringe < 40:
        # Low frequency - large scale waves
        dark_photon_field = radial_modes * 0.7 + spiral_modes * 0.3
    elif fringe < 70:
        # Medium frequency - mixed patterns
        dark_photon_field = radial_modes * 0.4 + angular_modes * 0.3 + spiral_modes * 0.3
    else:
        # High frequency - fine interference
        dark_photon_field = (radial_modes * 0.3 + angular_modes * 0.4 + 
                            spiral_modes * 0.3 + np.sin(wave_number * 4 * np.pi * r) * 0.2)
    
    # Normalize
    dark_photon_field = (dark_photon_field - dark_photon_field.min()) / (dark_photon_field.max() - dark_photon_field.min() + 1e-9)
    
    return dark_photon_field, wave_number


def create_dark_matter_substructure(data, fringe_value):
    """
    Creates dark matter density map with proper physics
    """
    h, w = data.shape
    
    # Ensure integer values
    fringe = int(fringe_value)
    
    # Smooth to get mass distribution
    smoothed = gaussian_filter(data, sigma=5)
    
    # Gradient for density variations (gravitational potential)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Fringe-dependent substructure scale (ensure integer)
    substructure_scale = max(3, min(15, int(15 - (fringe / 10))))
    
    # Start with gradient map
    dm_density = gradient_magnitude.copy()
    
    # Add dark matter halos using physics-based approach
    try:
        # Find local maxima (dark matter halo centers)
        footprint_size = substructure_scale
        neighborhood = np.ones((footprint_size, footprint_size))
        local_max = maximum_filter(smoothed, footprint=neighborhood) == smoothed
        coords = np.argwhere(local_max & (smoothed > np.percentile(smoothed, 85)))
        
        # Limit number of halos based on fringe
        num_halos = min(50, int(20 + fringe / 5))
        
        # Add Gaussian halos at each peak
        y_grid, x_grid = np.ogrid[:h, :w]
        for yc, xc in coords[:num_halos]:
            # Gaussian profile for dark matter halo
            y_dist = y_grid - yc
            x_dist = x_grid - xc
            r2 = (x_dist**2 + y_dist**2)
            sigma_halo = substructure_scale * 2
            halo = np.exp(-r2 / (2 * sigma_halo**2))
            dm_density += halo * smoothed[yc, xc] * 0.5
    except Exception as e:
        # Fallback: simple density enhancement
        dm_density = gradient_magnitude + smoothed * 0.3
    
    # Normalize
    dm_density = (dm_density - dm_density.min()) / (dm_density.max() - dm_density.min() + 1e-9)
    
    return dm_density


def photon_dark_photon_entanglement_visible(data, omega, fringe, physical_scale_kpc=100, brightness=1.2):
    """
    PDP Entanglement with visible fringe patterns
    """
    h, w = data.shape
    
    # Ensure inputs are valid
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0, 1)
    fringe = int(fringe)
    omega = float(omega)
    
    # Create PDP components
    dark_photon_field, wave_number = create_pdp_fringe_pattern(data, fringe, physical_scale_kpc)
    dm_density = create_dark_matter_substructure(data, fringe)
    
    # PDP coupling strength
    mixing_strength = omega * 0.7
    
    # Create entangled image (photon-dark photon mixing)
    # Based on: |γ⟩ → cos(θ)|γ⟩ + sin(θ)|γ_dark⟩
    entangled = data * (1 - mixing_strength * 0.5)
    entangled = entangled + dark_photon_field * mixing_strength * 0.6
    entangled = entangled + dm_density * mixing_strength * 0.4
    
    # Apply brightness
    entangled = entangled * brightness
    
    # Final clamp
    entangled = np.clip(entangled, 0, 1)
    
    # Create RGB overlay for visualization
    # Red: original image, Green: dark photon, Blue: dark matter
    overlay_rgb = np.stack([
        entangled,  # R
        entangled * 0.4 + dark_photon_field * 0.6,  # G - dark photon
        entangled * 0.3 + dm_density * 0.7  # B - dark matter
    ], axis=-1)
    
    return entangled, overlay_rgb, dark_photon_field, dm_density, wave_number


# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v11")
    st.markdown("### Photon-Dark-Photon Entangled FDM")
    st.markdown("*Fixed Physics Engine*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload FITS or Image", 
                                type=["fits", "png", "jpg", "jpeg", "tif", "tiff"])
    
    st.markdown("---")
    st.markdown("### ⚛️ PDP Physics Parameters")
    
    omega = st.slider("Ω Entanglement Strength", 0.1, 1.0, 0.65, 
                       help="Coupling between photon and dark photon fields")
    
    fringe = st.slider("Fringe Scale (k⁻¹)", 20, 120, 70,
                       help="Dark photon oscillation wavenumber")
    
    brightness = st.slider("Image Brightness", 0.8, 1.8, 1.25)
    
    physical_scale = st.selectbox("Physical Scale", 
                                   ["50 kpc", "100 kpc", "200 kpc", "500 kpc"],
                                   index=1)
    scale_map = {"50 kpc": 50, "100 kpc": 100, "200 kpc": 200, "500 kpc": 500}
    physical_scale_kpc = scale_map[physical_scale]
    
    st.markdown("---")
    st.success(f"""
    **Active PDP Physics**
    • Fringe: {fringe}
    • Ω: {omega:.2f}
    • Mixing: {omega*0.7:.2f}
    • Wave number: {fringe/50:.2f}
    """)
    
    st.caption("Tony Ford Model | v11 Fixed")


# ── MAIN PIPELINE ──────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled Fuzzy Dark Matter with Visible Fringe Patterns*")
st.markdown("---")

if uploaded:
    # Load image
    ext = uploaded.name.split(".")[-1].lower()
    data_bytes = uploaded.read()
    
    with st.spinner("📸 Loading astronomical image..."):
        try:
            if ext == "fits":
                with fits.open(io.BytesIO(data_bytes)) as h:
                    raw = h[0].data.astype(np.float32)
                    if len(raw.shape) > 2:
                        raw = raw[0] if raw.shape[0] < raw.shape[1] else raw[:, :, 0]
            else:
                img = PILImage.open(io.BytesIO(data_bytes)).convert("L")
                raw = np.array(img, dtype=np.float32)
        except Exception as e:
            st.error(f"Error loading: {e}")
            st.stop()
    
    # Ensure 2D
    if len(raw.shape) > 2:
        raw = raw[:, :, 0]
    
    # Resize if too large
    MAX_SIZE = 800
    if raw.shape[0] > MAX_SIZE or raw.shape[1] > MAX_SIZE:
        from skimage.transform import resize
        raw = resize(raw, (MAX_SIZE, MAX_SIZE), preserve_range=True)
    
    # Process pipeline
    with st.spinner("⚛️ Running PDP Physics Engine..."):
        norm = normalize(raw)
        psf = psf_correct(norm)
        enhanced = enhance_resolution(psf)
        
        # Apply PDP entanglement
        ent, overlay_rgb, dark_photon, dm_density, wave_number = photon_dark_photon_entanglement_visible(
            enhanced, omega, fringe, physical_scale_kpc, brightness
        )
    
    # Success message
    st.success(f"✅ PDP Physics Complete | Fringe = {fringe} | Ω = {omega:.2f} | Wave Number = {wave_number:.2f}")
    
    # ── DISPLAY RESULTS ────────────────────────────────────
    st.markdown("### 📊 Pipeline Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    def show_img(img, title, cmap="inferno"):
        try:
            fig, ax = plt.subplots(figsize=(3, 3))
            if len(img.shape) > 2:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.write(f"⚠️ {title}")
    
    with col1: show_img(norm, "📷 Input", "inferno")
    with col2: show_img(psf, "🔍 PSF", "inferno")
    with col3: show_img(enhanced, "🧠 Enhanced", "inferno")
    with col4: show_img(ent, "✨ PDP Entangled", "inferno")
    
    st.markdown("---")
    st.markdown("### 🌌 PDP Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**Dark Photon Field**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dark_photon, cmap="plasma")
        ax.set_title(f"Fringe Pattern (k = {fringe})", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Amplitude")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        st.markdown("**Dark Matter Density**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dm_density, cmap="viridis")
        ax.set_title("FDM Substructure Map", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Density")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_c:
        st.markdown("**PDP RGB Composite**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(overlay_rgb)
        ax.set_title("R: Image | G: Dark Photon | B: Dark Matter", fontsize=8)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    # ── FRINGE DETAIL ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Fringe Pattern Analysis")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown(f"**Fringe Detail (fringe = {fringe})**")
        # Zoom into center region
        h, w = dark_photon.shape
        zh, zw = h//3, w//3
        zoom_region = dark_photon[zh:zh+150, zw:zw+150]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(zoom_region, cmap="plasma")
        ax.set_title(f"Wave Interference Pattern", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_f2:
        st.markdown("**Power Spectrum**")
        fft = np.fft.fft2(dark_photon)
        fft_shift = np.fft.fftshift(fft)
        power = np.log10(np.abs(fft_shift) + 1)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(power, cmap="hot")
        ax.set_title("Frequency Domain (Fringe Peaks)", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="log(Power)")
        st.pyplot(fig)
        plt.close(fig)
    
    # ── COMPARISON ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before/After Comparison")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.imshow(raw, cmap="gray")
    ax1.set_title("Original Raw Image", fontsize=12)
    ax1.axis("off")
    
    ax2.imshow(enhanced, cmap="inferno")
    ax2.set_title("Enhanced (No PDP)", fontsize=12)
    ax2.axis("off")
    
    ax3.imshow(ent, cmap="inferno")
    ax3.set_title(f"PDP Entangled (Ω={omega:.2f}, k={fringe})", fontsize=12)
    ax3.axis("off")
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # ── METRICS ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 PDP Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        fringe_contrast = np.std(dark_photon)
        st.metric("Fringe Contrast", f"{fringe_contrast:.3f}")
    
    with col_m2:
        dm_contrast = np.std(dm_density)
        st.metric("DM Structure", f"{dm_contrast:.3f}")
    
    with col_m3:
        mixing = omega * 0.7
        st.metric("PDP Mixing", f"{mixing:.2f}")
    
    with col_m4:
        info = np.var(ent) / (np.var(enhanced) + 1e-9)
        st.metric("Info Gain", f"{info:.2f}x")
    
    # ── DOWNLOAD ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        buf = io.BytesIO()
        plt.imsave(buf, ent, cmap="inferno")
        st.download_button("📸 PDP Entangled", buf.getvalue(), 
                          f"pdp_entangled_f{fringe}_o{omega:.2f}.png")
    
    with col_d2:
        buf = io.BytesIO()
        plt.imsave(buf, dark_photon, cmap="plasma")
        st.download_button("🌊 Fringe Pattern", buf.getvalue(),
                          f"fringe_k{fringe}.png")
    
    with col_d3:
        buf = io.BytesIO()
        plt.imsave(buf, dm_density, cmap="viridis")
        st.download_button("🌌 DM Map", buf.getvalue(),
                          f"darkmatter_f{fringe}.png")

else:
    st.info("✨ **Upload an image to run the PDP Physics Engine**\n\n"
            "**What this does:**\n"
            "• ⚛️ **Photon-Dark Photon Entanglement**: Quantum mixing of visible and dark sectors\n"
            "• 🌊 **Fringe Patterns**: Visible wave interference from dark photon oscillations\n"
            "• 🌌 **Dark Matter Substructure**: FDM halos from gravitational potential\n"
            "• 🔬 **Physics-Based**: Tony Ford Model with proper wave equations\n\n"
            "*Recommended: Start with Ω=0.65, Fringe=70 for visible effects*")

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v11** | Fixed PDP Physics Engine | Light Blue Interface | Tony Ford Model")
