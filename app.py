# QCI AstroEntangle Refiner – v12 WITH SOLITON WAVES
# ADDED: FDM soliton core physics, visible wave structures

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
from scipy.ndimage import gaussian_filter, laplace, sobel, zoom, maximum_filter
from scipy.special import jv  # Bessel functions for soliton profiles
from PIL import Image as PILImage

# ── LIGHT BLUE INTERFACE ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI AstroEntangle Refiner v12", page_icon="🔭")

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
}
[data-testid="stMetricValue"] {
    color: #01579b !important;
}
</style>
""", unsafe_allow_html=True)

# ── SOLITON PHYSICS FUNCTIONS ─────────────────────────────────────

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


# ===== FDM SOLITON PHYSICS =====

def create_fdm_soliton(data, fringe_value, physical_scale_kpc=100):
    """
    Creates FDM soliton core - the ground state of fuzzy dark matter
    Soliton profile: ρ(r) ∝ [sin(kr)/(kr)]^2 for 3D, or Bessel for 2D projection
    """
    h, w = data.shape
    fringe = int(fringe_value)
    
    # Create coordinate grid centered on image
    y, x = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(w, h)
    
    # Soliton scale radius (depends on fringe - higher fringe = smaller soliton)
    # Based on FDM physics: r_soliton ∝ 1/m_fdm, fringe scales with mass
    soliton_scale = 0.15 * (50.0 / max(fringe, 1))  # Larger fringe = smaller core
    
    # FDM soliton profile (Schrödinger-Poisson ground state)
    # 2D projection of 3D soliton: ρ(r) ∝ [sin(kr)/(kr)]^2
    k_soliton = np.pi / soliton_scale
    kr = k_soliton * r
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Soliton density profile (dimensionless)
        soliton_profile = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    # Add secondary soliton ring (characteristic of FDM interference)
    ring_profile = np.where(kr > 1e-6, (np.sin(2*kr) / (2*kr))**2 * 0.3, 0.3)
    
    # Combine main soliton and ring
    soliton_core = soliton_profile * 0.8 + ring_profile * 0.2
    
    # Normalize
    soliton_core = (soliton_core - soliton_core.min()) / (soliton_core.max() - soliton_core.min() + 1e-9)
    
    # Apply Gaussian smoothing for realistic appearance
    soliton_core = gaussian_filter(soliton_core, sigma=2)
    
    return soliton_core, soliton_scale


def create_pdp_fringe_pattern_with_soliton(data, fringe_value, physical_scale_kpc=100):
    """
    Creates dark photon oscillation patterns with soliton modulation
    """
    h, w = data.shape
    fringe = int(fringe_value)
    
    # Wave number
    wave_number = fringe / 40.0
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    cx, cy = w/2, h/2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(w, h)
    theta = np.arctan2(y - cy, x - cx)
    
    # Create soliton envelope
    soliton, soliton_scale = create_fdm_soliton(data, fringe, physical_scale_kpc)
    
    # PDP Wave Equation with soliton modulation
    # ψ_dark = soliton(r) * [sin(kr) + spiral modes]
    
    # 1. Radial waves (modulated by soliton)
    radial_modes = np.sin(wave_number * 4 * np.pi * r) * soliton
    
    # 2. Spiral waves (dark matter vortex structures)
    spiral_modes = np.sin(wave_number * 2 * np.pi * (r + theta / (2 * np.pi))) * soliton
    
    # 3. Quantum interference fringes (characteristic of soliton oscillations)
    interference = np.sin(wave_number * 6 * np.pi * r) * np.cos(wave_number * 2 * theta)
    interference = interference * soliton
    
    # Combine based on fringe
    if fringe < 40:
        dark_photon_field = radial_modes * 0.6 + interference * 0.4
    elif fringe < 70:
        dark_photon_field = radial_modes * 0.4 + spiral_modes * 0.4 + interference * 0.2
    else:
        dark_photon_field = spiral_modes * 0.5 + interference * 0.3 + radial_modes * 0.2
    
    # Add soliton core as base brightness
    dark_photon_field = dark_photon_field * (0.5 + 0.5 * soliton)
    
    # Normalize
    dark_photon_field = (dark_photon_field - dark_photon_field.min()) / (dark_photon_field.max() - dark_photon_field.min() + 1e-9)
    
    return dark_photon_field, wave_number, soliton, soliton_scale


def create_dark_matter_substructure_with_soliton(data, fringe_value):
    """
    Creates dark matter density map with soliton core and substructure
    """
    h, w = data.shape
    fringe = int(fringe_value)
    
    # Smooth to get mass distribution
    smoothed = gaussian_filter(data, sigma=5)
    
    # Gradient for density variations
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create soliton core
    soliton, soliton_scale = create_fdm_soliton(data, fringe)
    
    # Start with gradient map plus soliton
    dm_density = gradient_magnitude * 0.5 + soliton * 0.5
    
    # Add substructure halos
    substructure_scale = max(3, int(15 - (fringe / 10)))
    
    try:
        footprint_size = min(substructure_scale, 15)
        neighborhood = np.ones((footprint_size, footprint_size))
        local_max = maximum_filter(smoothed, footprint=neighborhood) == smoothed
        coords = np.argwhere(local_max & (smoothed > np.percentile(smoothed, 85)))
        
        num_halos = min(40, int(15 + fringe / 8))
        y_grid, x_grid = np.ogrid[:h, :w]
        
        for yc, xc in coords[:num_halos]:
            y_dist = y_grid - yc
            x_dist = x_grid - xc
            r2 = (x_dist**2 + y_dist**2)
            sigma_halo = substructure_scale * 2
            halo = np.exp(-r2 / (2 * sigma_halo**2))
            dm_density += halo * smoothed[yc, xc] * 0.3
    except:
        pass
    
    # Normalize
    dm_density = (dm_density - dm_density.min()) / (dm_density.max() - dm_density.min() + 1e-9)
    
    return dm_density, soliton


def photon_dark_photon_entanglement_with_soliton(data, omega, fringe, physical_scale_kpc=100, brightness=1.2):
    """
    PDP Entanglement with FDM soliton waves
    """
    h, w = data.shape
    
    # Ensure inputs are valid
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0, 1)
    fringe = int(fringe)
    omega = float(omega)
    
    # Create PDP components with soliton
    dark_photon_field, wave_number, soliton, soliton_scale = create_pdp_fringe_pattern_with_soliton(
        data, fringe, physical_scale_kpc
    )
    dm_density, dm_soliton = create_dark_matter_substructure_with_soliton(data, fringe)
    
    # Combine solitons (use the one from DM for consistency)
    combined_soliton = np.clip(soliton + dm_soliton * 0.5, 0, 1)
    
    # PDP coupling strength
    mixing_strength = omega * 0.8
    
    # Create entangled image with soliton enhancement
    entangled = data * (1 - mixing_strength * 0.4)
    entangled = entangled + dark_photon_field * mixing_strength * 0.5
    entangled = entangled + dm_density * mixing_strength * 0.3
    
    # Add soliton core enhancement (characteristic FDM feature)
    entangled = entangled + combined_soliton * mixing_strength * 0.4
    
    # Apply brightness
    entangled = entangled * brightness
    
    # Final clamp
    entangled = np.clip(entangled, 0, 1)
    
    # Create RGB overlay for visualization
    # Red: original, Green: dark photon + soliton, Blue: dark matter + soliton
    overlay_rgb = np.stack([
        entangled,  # R
        entangled * 0.4 + dark_photon_field * 0.4 + combined_soliton * 0.2,  # G
        entangled * 0.3 + dm_density * 0.5 + combined_soliton * 0.2  # B
    ], axis=-1)
    
    return entangled, overlay_rgb, dark_photon_field, dm_density, combined_soliton, wave_number, soliton_scale


# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v12")
    st.markdown("### Photon-Dark-Photon Entangled FDM")
    st.markdown("*With FDM Soliton Physics*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload FITS or Image", 
                                type=["fits", "png", "jpg", "jpeg", "tif", "tiff"])
    
    st.markdown("---")
    st.markdown("### ⚛️ PDP Physics Parameters")
    
    omega = st.slider("Ω Entanglement Strength", 0.1, 1.0, 0.70, 
                       help="Coupling between photon and dark photon fields")
    
    fringe = st.slider("Fringe Scale (k⁻¹)", 20, 120, 65,
                       help="Dark photon wavenumber - controls soliton size")
    
    brightness = st.slider("Image Brightness", 0.8, 1.8, 1.2)
    
    physical_scale = st.selectbox("Physical Scale", 
                                   ["50 kpc", "100 kpc", "200 kpc", "500 kpc"],
                                   index=1)
    scale_map = {"50 kpc": 50, "100 kpc": 100, "200 kpc": 200, "500 kpc": 500}
    physical_scale_kpc = scale_map[physical_scale]
    
    st.markdown("---")
    st.success(f"""
    **FDM Soliton Physics Active**
    • Fringe: {fringe}
    • Ω: {omega:.2f}
    • Soliton scale: {50/fringe:.2f} r_s
    • Wave number: {fringe/40:.2f}
    """)
    
    st.caption("Tony Ford Model | v12 - Soliton Waves Added")


# ── MAIN PIPELINE ──────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled FDM with Soliton Core Waves*")
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
    with st.spinner("⚛️ Running FDM Soliton Physics Engine..."):
        norm = normalize(raw)
        psf = psf_correct(norm)
        enhanced = enhance_resolution(psf)
        
        # Apply PDP entanglement with soliton
        ent, overlay_rgb, dark_photon, dm_density, soliton, wave_number, soliton_scale = photon_dark_photon_entanglement_with_soliton(
            enhanced, omega, fringe, physical_scale_kpc, brightness
        )
    
    # Success message
    st.success(f"✅ FDM Soliton Physics Complete | Fringe = {fringe} | Ω = {omega:.2f} | Soliton Scale = {soliton_scale:.3f}")
    
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
    st.markdown("### 🌌 FDM Soliton Physics Components")
    
    # Display soliton separately
    st.markdown("#### ⚛️ FDM Soliton Core")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(soliton, cmap="plasma")
    ax.set_title(f"FDM Soliton Core (Scale = {soliton_scale:.3f} r_s)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Soliton Density")
    st.pyplot(fig)
    plt.close(fig)
    
    # Three columns for other components
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**Dark Photon Field**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dark_photon, cmap="plasma")
        ax.set_title(f"Oscillation Pattern (k={fringe})", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)
        plt.close(fig)
    
    with col_b:
        st.markdown("**Dark Matter Density**")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(dm_density, cmap="viridis")
        ax.set_title("FDM Substructure + Soliton", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
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
    
    # ── FRINGE ANALYSIS WITH SOLITON ────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Fringe + Soliton Analysis")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown(f"**Wave Interference (fringe = {fringe})**")
        h, w = dark_photon.shape
        zh, zw = h//3, w//3
        zoom_region = dark_photon[zh:zh+150, zw:zw+150]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(zoom_region, cmap="plasma")
        ax.set_title("Fringe Detail with Soliton Modulation", fontsize=10)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    
    with col_f2:
        st.markdown("**Soliton Radial Profile**")
        # Get radial profile of soliton
        h, w = soliton.shape
        cx, cy = w//2, h//2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_flat = r.flatten()
        soliton_flat = soliton.flatten()
        
        # Bin by radius
        bins = np.linspace(0, max(w, h)//2, 50)
        r_binned = []
        soliton_binned = []
        for i in range(len(bins)-1):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.sum(mask) > 0:
                r_binned.append((bins[i] + bins[i+1])/2)
                soliton_binned.append(np.mean(soliton_flat[mask]))
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r_binned, soliton_binned, 'b-', linewidth=2)
        ax.set_xlabel("Radius (pixels)", fontsize=10)
        ax.set_ylabel("Soliton Density", fontsize=10)
        ax.set_title(f"FDM Soliton Profile (r_s ≈ {soliton_scale:.2f})", fontsize=10)
        ax.grid(True, alpha=0.3)
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
    ax3.set_title(f"PDP + FDM Soliton (Ω={omega:.2f}, k={fringe})", fontsize=12)
    ax3.axis("off")
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # ── METRICS ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 FDM Soliton Metrics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Soliton Peak", f"{soliton.max():.3f}")
    
    with col_m2:
        st.metric("Soliton Scale", f"{soliton_scale:.3f}")
    
    with col_m3:
        mixing = omega * 0.8
        st.metric("PDP Mixing", f"{mixing:.2f}")
    
    with col_m4:
        fringe_contrast = np.std(dark_photon)
        st.metric("Fringe Contrast", f"{fringe_contrast:.3f}")
    
    # ── DOWNLOAD ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        buf = io.BytesIO()
        plt.imsave(buf, ent, cmap="inferno")
        st.download_button("📸 PDP Entangled", buf.getvalue(), 
                          f"pdp_soliton_f{fringe}.png")
    
    with col_d2:
        buf = io.BytesIO()
        plt.imsave(buf, soliton, cmap="plasma")
        st.download_button("⭐ FDM Soliton", buf.getvalue(),
                          f"soliton_core_f{fringe}.png")
    
    with col_d3:
        buf = io.BytesIO()
        plt.imsave(buf, dark_photon, cmap="plasma")
        st.download_button("🌊 Fringe Pattern", buf.getvalue(),
                          f"fringe_k{fringe}.png")
    
    with col_d4:
        buf = io.BytesIO()
        plt.imsave(buf, dm_density, cmap="viridis")
        st.download_button("🌌 DM Map", buf.getvalue(),
                          f"darkmatter_f{fringe}.png")

else:
    st.info("✨ **Upload an image to see FDM Soliton Waves**\n\n"
            "**What's new in v12:**\n"
            "• ⚛️ **FDM Soliton Core**: Ground state of fuzzy dark matter\n"
            "• 🌊 **Soliton-Modulated Fringes**: Wave patterns enhanced by soliton\n"
            "• 📈 **Radial Profile**: Shows the characteristic soliton density profile\n"
            "• 🔬 **Tony Ford Model**: Complete PDP + FDM physics\n\n"
            "*The soliton appears as a bright central core with [sin(kr)/kr]² profile*")

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v12** | FDM Soliton Physics | Light Blue Interface | Tony Ford Model")
