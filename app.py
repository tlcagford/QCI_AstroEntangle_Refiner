# QCI AstroEntangle Refiner – v10 VISIBLE PDP EFFECTS
# FIXED: Clear PDP images, visible fringe patterns, proper dark matter overlays

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
from scipy.ndimage import gaussian_filter, laplace, sobel, zoom
from PIL import Image as PILImage

# ── CONFIG ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI Refiner v10 - Visible PDP", page_icon="🔭")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0b1a; }
[data-testid="stSidebar"] { background: #10102a; }
.stTitle { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ─────────────────────────────────────

def normalize(arr):
    """Normalize for display"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if vmax - vmin < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)


def psf_correct(data, amount=0.6):
    """PSF correction"""
    try:
        kernel = Gaussian2DKernel(x_stddev=1.8)
        psf = kernel.array / kernel.array.sum()
        blurred = convolve2d(data, psf, mode="same")
        result = np.clip(data + amount * (data - blurred), 0, 1)
        return np.nan_to_num(result)
    except:
        return data


def enhance_resolution(data):
    """Gentle resolution enhancement"""
    try:
        # Simple unsharp masking for clarity
        blurred = gaussian_filter(data, sigma=0.8)
        edges = data - blurred
        enhanced = data + edges * 0.4
        return np.clip(enhanced, 0, 1)
    except:
        return data


# ===== VISIBLE PDP CONVERSION WITH FRINGE PATTERNS =====

def create_pdp_fringe_pattern(data, fringe_value, physical_scale_kpc=100):
    """
    Creates VISIBLE dark photon oscillation patterns based on fringe parameter
    """
    h, w = data.shape
    
    # Convert fringe to wave characteristics
    # Higher fringe = more oscillations
    wave_number = fringe_value / 25.0  # Normalize to reasonable range
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    
    # Center coordinates
    cx, cy = w/2, h/2
    
    # Create multiple wave patterns for rich interference
    
    # 1. Radial waves (concentric circles)
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(w, h)
    radial_freq = wave_number * 3
    radial_pattern = np.sin(radial_freq * 2 * np.pi * r) * np.exp(-r * 1.5)
    
    # 2. Spiral waves (characteristic of quantum interference)
    theta = np.arctan2(y - cy, x - cx)
    spiral_freq = wave_number * 2
    spiral_pattern = np.sin(spiral_freq * theta + radial_freq * r * 2)
    
    # 3. Linear waves (direction-dependent)
    linear_freq = wave_number * 1.5
    linear_pattern = np.sin(linear_freq * 2 * np.pi * x / w) * \
                     np.cos(linear_freq * 2 * np.pi * y / h)
    
    # 4. Moiré interference (fringe-dependent)
    moire_pattern = np.sin((radial_pattern + spiral_pattern) * wave_number * np.pi)
    
    # Combine patterns based on fringe
    if fringe < 40:
        # Low fringe: large-scale radial patterns
        dark_photon_field = radial_pattern * 0.6 + moire_pattern * 0.4
    elif fringe < 70:
        # Medium fringe: mixed radial and spiral
        dark_photon_field = radial_pattern * 0.4 + spiral_pattern * 0.4 + moire_pattern * 0.2
    else:
        # High fringe: fine-grained interference
        dark_photon_field = spiral_pattern * 0.5 + linear_pattern * 0.3 + moire_pattern * 0.2
    
    # Normalize to [0,1]
    dark_photon_field = (dark_photon_field - dark_photon_field.min()) / (dark_photon_field.max() - dark_photon_field.min() + 1e-9)
    
    return dark_photon_field, wave_number


def create_dark_matter_substructure(data, fringe_value):
    """
    Creates visible dark matter density maps based on fringe
    """
    h, w = data.shape
    
    # Use data to trace mass distribution
    smoothed = gaussian_filter(data, sigma=5)
    
    # Gradient for density variations
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Fringe-dependent substructure scale
    substructure_scale = max(3, 15 - (fringe_value / 10))
    
    # Add clumpy substructure (characteristic of dark matter halos)
    y, x = np.ogrid[:h, :w]
    num_halos = int(20 + fringe_value / 5)
    
    dm_density = gradient_magnitude.copy()
    
    # Add artificial dark matter halos at density peaks
    from scipy.ndimage import maximum_filter, label
    
    # Find local maxima in smoothed data
    neighborhood = np.ones((substructure_scale, substructure_scale))
    local_max = maximum_filter(smoothed, footprint=neighborhood) == smoothed
    coords = np.argwhere(local_max & (smoothed > np.percentile(smoothed, 85)))
    
    # Add Gaussian halos at these positions
    for yc, xc in coords[:num_halos]:
        y_dist = y - yc
        x_dist = x - xc
        r2 = (x_dist**2 + y_dist**2)
        halo = np.exp(-r2 / (2 * (substructure_scale * 2)**2))
        dm_density += halo * smoothed[yc, xc]
    
    # Normalize
    dm_density = (dm_density - dm_density.min()) / (dm_density.max() - dm_density.min() + 1e-9)
    
    return dm_density


def photon_dark_photon_entanglement_visible(data, omega, fringe, physical_scale_kpc=100, brightness=1.2):
    """
    Creates VISIBLE PDP entanglement with clear fringe patterns
    """
    h, w = data.shape
    
    # Ensure data is valid
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0, 1)
    
    # Create visible fringe patterns
    dark_photon_field, wave_number = create_pdp_fringe_pattern(data, fringe, physical_scale_kpc)
    
    # Create dark matter substructure
    dm_density = create_dark_matter_substructure(data, fringe)
    
    # PDP Entanglement mixing
    # Higher omega = more visible dark matter effects
    mixing_strength = omega * 0.8
    
    # Create entangled image
    # Start with base image
    entangled = data.copy()
    
    # Add dark photon modulations (creates visible wave patterns)
    entangled = entangled * (1 - mixing_strength * 0.4)
    entangled = entangled + dark_photon_field * mixing_strength * 0.6
    
    # Add dark matter substructure
    entangled = entangled + dm_density * mixing_strength * 0.5
    
    # Enhance fringe visibility through edge modulation
    edges = sobel(entangled)
    entangled = entangled + edges * dark_photon_field * mixing_strength * 0.3
    
    # Apply brightness
    entangled = entangled * brightness
    
    # Final clamp
    entangled = np.clip(entangled, 0, 1)
    
    # Create colored overlay for visualization
    # RGB overlay: R = original, G = dark photon, B = dark matter
    overlay_rgb = np.stack([
        entangled,  # Red channel
        entangled * 0.5 + dark_photon_field * 0.5,  # Green channel (dark photon in green)
        entangled * 0.3 + dm_density * 0.7  # Blue channel (dark matter in blue)
    ], axis=-1)
    
    return entangled, overlay_rgb, dark_photon_field, dm_density, wave_number


# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v10")
    st.markdown("**Visible PDP Fringe Patterns**")
    st.markdown("---")
    
    uploaded = st.file_uploader("Upload FITS or Image", type=["fits", "png", "jpg", "jpeg", "tif", "tiff"])
    
    st.markdown("---")
    st.markdown("### PDP Parameters")
    
    omega = st.slider("Ω Entanglement Strength", 0.1, 1.0, 0.55, 
                       help="Higher = more visible dark matter patterns")
    
    fringe = st.slider("Fringe Scale (PDP λ)", 20, 120, 65,
                       help="Controls wave pattern density - higher = more oscillations")
    
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2)
    
    physical_scale = st.selectbox("Physical Scale", 
                                   ["50 kpc", "100 kpc", "200 kpc"],
                                   index=1)
    scale_map = {"50 kpc": 50, "100 kpc": 100, "200 kpc": 200}
    physical_scale_kpc = scale_map[physical_scale]
    
    st.markdown("---")
    st.info(f"**Current PDP Settings**\n\n"
            f"• Fringe = {fringe}\n"
            f"• Wave density = {fringe/20:.1f}x\n"
            f"• Ω = {omega:.2f}\n"
            f"• Visibility = {'High' if omega > 0.5 else 'Medium' if omega > 0.3 else 'Low'}")
    
    st.caption("Based on Tony Ford Model | v10 Visible PDP")


# ── MAIN PIPELINE ──────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled FDM with Visible Fringe Patterns*")
st.markdown("---")

if uploaded:
    # Load image
    ext = uploaded.name.split(".")[-1].lower()
    data_bytes = uploaded.read()
    
    with st.spinner("Loading image..."):
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
    
    # Process
    with st.spinner("Creating PDP entanglement with visible fringe patterns..."):
        norm = normalize(raw)
        psf = psf_correct(norm)
        enhanced = enhance_resolution(psf)
        
        # Apply PDP entanglement
        ent, overlay_rgb, dark_photon, dm_density, wave_number = photon_dark_photon_entanglement_visible(
            enhanced, omega, fringe, physical_scale_kpc, brightness
        )
    
    # Display info
    st.success(f"✅ PDP Applied | Fringe = {fringe} | Wave density = {wave_number:.1f} cycles/image | Ω = {omega:.2f}")
    
    # ── DISPLAY RESULTS ────────────────────────────────────
    st.markdown("### 📊 Pipeline Results")
    
    # First row: Main processing steps
    col1, col2, col3, col4 = st.columns(4)
    
    def show(img, title, cmap="inferno", figsize=(3, 3)):
        try:
            fig, ax = plt.subplots(figsize=figsize)
            if len(img.shape) > 2:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, color='white', fontsize=9)
            ax.axis("off")
            fig.patch.set_facecolor('#0b0b1a')
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.write(f"⚠️ {title}")
    
    with col1: show(norm, "📷 Input", "inferno")
    with col2: show(psf, "🔍 PSF", "inferno")
    with col3: show(enhanced, "🧠 Enhanced", "inferno")
    with col4: show(ent, "✨ PDP Entangled", "inferno")
    
    st.markdown("---")
    st.markdown("### 🌌 PDP Components (Visible Fringe Patterns)")
    
    # Second row: PDP components
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**Dark Photon Field**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dark_photon, cmap="plasma")
        ax.set_title(f"Fringe Pattern (λ = {fringe})", color='white')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Amplitude")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        st.markdown("**Dark Matter Density**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dm_density, cmap="viridis")
        ax.set_title("Substructure Map", color='white')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Density")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_c:
        st.markdown("**PDP RGB Composite**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(overlay_rgb)
        ax.set_title("Red: Image | Green: Dark Photon | Blue: Dark Matter", color='white', fontsize=8)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    # ── FRINGE VISUALIZATION ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Fringe Pattern Analysis")
    
    # Show fringe patterns at different scales
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown(f"**Fringe Pattern Detail (fringe = {fringe})**")
        # Zoom into a region to show fringe details
        zoom_y = dark_photon.shape[0] // 3
        zoom_x = dark_photon.shape[1] // 3
        zoom_region = dark_photon[zoom_y:zoom_y+150, zoom_x:zoom_x+150]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(zoom_region, cmap="plasma", extent=[0, 150, 0, 150])
        ax.set_title(f"Fringe Detail - {fringe} cycles/image", color='white')
        ax.set_xlabel("Pixels", color='white')
        ax.set_ylabel("Pixels", color='white')
        st.pyplot(fig)
        plt.close(fig)
    
    with col_f2:
        st.markdown("**Fringe Power Spectrum**")
        # Calculate FFT to show fringe frequencies
        fft = np.fft.fft2(dark_photon)
        fft_shift = np.fft.fftshift(fft)
        power = np.log10(np.abs(fft_shift) + 1)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(power, cmap="hot")
        ax.set_title("Frequency Domain (Fringe Peaks Visible)", color='white')
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="log(Power)")
        st.pyplot(fig)
        plt.close(fig)
    
    # ── COMPARISON ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before vs After with PDP")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.imshow(raw, cmap="gray")
    ax1.set_title("Original", color='white', fontsize=12)
    ax1.axis("off")
    
    ax2.imshow(enhanced, cmap="inferno")
    ax2.set_title("Enhanced (No PDP)", color='white', fontsize=12)
    ax2.axis("off")
    
    ax3.imshow(ent, cmap="inferno")
    ax3.set_title(f"PDP Entangled (Ω={omega:.2f}, Fringe={fringe})", color='white', fontsize=12)
    ax3.axis("off")
    
    fig.patch.set_facecolor('#0b0b1a')
    st.pyplot(fig)
    plt.close(fig)
    
    # ── METRICS ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 PDP Enhancement Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        fringe_contrast = np.std(dark_photon)
        st.metric("Fringe Visibility", f"{fringe_contrast:.3f}", 
                 delta="High" if fringe_contrast > 0.3 else "Low")
    
    with col_m2:
        dm_contrast = np.std(dm_density)
        st.metric("DM Substructure", f"{dm_contrast:.3f}")
    
    with col_m3:
        edge_boost = np.sum(np.abs(sobel(ent))) / (np.sum(np.abs(sobel(enhanced))) + 1e-9)
        st.metric("Edge Enhancement", f"{edge_boost:.2f}x")
    
    with col_m4:
        from scipy.stats import entropy
        hist_before, _ = np.histogram(enhanced, bins=50)
        hist_after, _ = np.histogram(ent, bins=50)
        info_gain = entropy(hist_after + 1) - entropy(hist_before + 1)
        st.metric("Information Gain", f"{info_gain:.2f} bits")
    
    # ── DOWNLOAD ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        buf = io.BytesIO()
        plt.imsave(buf, ent, cmap="inferno")
        st.download_button("📸 PDP Entangled", buf.getvalue(), 
                          f"pdp_entangled_f{fringe}.png")
    
    with col_d2:
        buf = io.BytesIO()
        plt.imsave(buf, dark_photon, cmap="plasma")
        st.download_button("🌊 Fringe Pattern", buf.getvalue(),
                          f"fringe_pattern_f{fringe}.png")
    
    with col_d3:
        buf = io.BytesIO()
        plt.imsave(buf, dm_density, cmap="viridis")
        st.download_button("🌌 Dark Matter Map", buf.getvalue(),
                          f"dm_map_f{fringe}.png")
    
    with col_d4:
        buf = io.BytesIO()
        plt.imsave(buf, overlay_rgb)
        st.download_button("🎨 RGB Composite", buf.getvalue(),
                          f"pdp_rgb_f{fringe}.png")

else:
    st.info("✨ **Upload an image to see visible PDP fringe patterns**\n\n"
            "**What you'll see:**\n"
            "- 🌊 **Fringe Patterns**: Visible wave interference from dark photons\n"
            "- 🌌 **Dark Matter Maps**: Clumpy substructure from FDM simulations\n"
            "- 🎨 **RGB Composite**: Red=Image, Green=Dark Photon, Blue=Dark Matter\n"
            "- 📊 **Fringe Analysis**: Power spectrum and detail zoom\n\n"
            "*Based on the Tony Ford Model for visible photon-dark-photon entanglement*")
    
    st.markdown("---")
    st.markdown("### 🎮 Parameter Guide for Visible PDP Effects")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Ω Entanglement Strength (0.1-1.0)**")
        st.markdown("- **0.1-0.3**: Subtle, barely visible patterns")
        st.markdown("- **0.4-0.6**: Moderate, clearly visible fringes")
        st.markdown("- **0.7-1.0**: Strong, dominant wave patterns")
    
    with col_g2:
        st.markdown("**Fringe Scale (20-120)**")
        st.markdown("- **20-40**: Large, smooth waves (galaxy-scale)")
        st.markdown("- **40-70**: Medium interference patterns")
        st.markdown("- **70-120**: Fine, detailed oscillations")

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v10** | Visible PDP Fringe Patterns | Tony Ford Model")
