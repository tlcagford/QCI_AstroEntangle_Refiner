# QCI AstroEntangle Refiner – v9 BRIGHTNESS FIXED
# FIXED: Brightness preservation, proper contrast, shape handling

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
st.set_page_config(layout="wide", page_title="QCI Refiner v9 - Brightness Fixed", page_icon="🔭")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0b1a; }
[data-testid="stSidebar"] { background: #10102a; }
.stTitle { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ─────────────────────────────────────

def normalize(arr, preserve_brightness=True):
    """Normalize while preserving overall brightness"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    if preserve_brightness:
        # Use median-based normalization to preserve brightness
        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
    else:
        vmin, vmax = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    
    if vmax - vmin < 1e-9:
        return np.zeros_like(arr)
    
    normalized = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
    
    # Apply gamma correction to prevent darkening
    gamma = 0.9
    normalized = np.power(normalized, gamma)
    
    return normalized


def psf_correct(data, amount=0.5):
    """Gentle PSF correction to avoid over-sharpening"""
    try:
        kernel = Gaussian2DKernel(x_stddev=1.5)
        psf = kernel.array / kernel.array.sum()
        blurred = convolve2d(data, psf, mode="same")
        # Reduced amount to prevent edge artifacts
        result = np.clip(data + amount * (data - blurred), 0, 1)
        return np.nan_to_num(result)
    except Exception as e:
        return data


# ===== SIMPLE BUT EFFECTIVE SUPER-RESOLUTION =====
def neural_sr_real(data):
    """Gentle super-resolution that preserves brightness"""
    try:
        # First, gently enhance edges
        blurred = gaussian_filter(data, sigma=0.8)
        edges = data - blurred
        # Add edges back with lower weight
        enhanced = data + edges * 0.3
        
        # Simple upscaling if needed (but preserve original size)
        if data.shape[0] < 500:
            enhanced = zoom(enhanced, 1.5, order=2)
            # Trim to reasonable size
            if enhanced.shape[0] > 800:
                enhanced = enhanced[:800, :800]
        
        return np.clip(np.nan_to_num(enhanced), 0, 1)
    except:
        return data


# ===== CORRECTED PDP CONVERSION WITH BRIGHTNESS PRESERVATION =====

def pdp_fringe_conversion(fringe_value, image_size, physical_scale_kpc=100):
    """Convert fringe slider to physical dark photon oscillation frequency"""
    base_wavelength_kpc = 30.0  # Increased for better visibility
    physical_wavelength_kpc = base_wavelength_kpc * (50.0 / max(fringe_value, 1))
    wave_number = (physical_scale_kpc / physical_wavelength_kpc) * (image_size / 100.0)
    mixing_angle = np.clip(fringe_value / 100.0, 0.15, 0.85)  # Reduced range
    
    return wave_number, physical_wavelength_kpc, mixing_angle


def photon_dark_photon_entanglement_corrected(data, omega, fringe, physical_scale_kpc=100, brightness=1.2):
    """
    Fixed: Preserves original brightness while adding subtle PDP effects
    """
    h, w = data.shape
    
    # Ensure data is valid and preserve original brightness
    original_mean = np.mean(data)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    data = np.clip(data, 0, 1)
    
    # Convert fringe to physical PDP parameters
    wave_number, physical_wavelength, mixing_angle = pdp_fringe_conversion(
        fringe, max(h, w), physical_scale_kpc
    )
    
    # Reduce entanglement strength for more subtle effect
    effective_omega = omega * 0.6  # Reduced for less dramatic effect
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    
    # 1. DARK PHOTON FIELD (more subtle)
    kx = wave_number * 2 * np.pi / w
    ky = wave_number * 2 * np.pi / h * 0.7
    
    dark_photon_field = np.sin(kx * x + ky * y) * np.cos(kx * x * 0.6 - ky * y * 0.4)
    
    # Gentle radial wave (reduced amplitude)
    r = np.sqrt((x - w/2)**2 + (y - h/2)**2) / max(w, h)
    radial_wave = np.sin(wave_number * 2 * np.pi * r) * np.exp(-r * 2) * 0.5
    dark_photon_field = (dark_photon_field + radial_wave) / 1.5
    
    # 2. DARK MATTER DENSITY (subtle)
    potential = gaussian_filter(data, sigma=6)
    dm_density = np.abs(np.gradient(np.gradient(potential))[0] + np.gradient(np.gradient(potential))[1])
    dm_density = dm_density / (dm_density.max() + 1e-9)
    dm_density = dm_density * 0.3  # Reduce intensity
    
    # 3. PDP ENTANGLEMENT MIXING (preserve original)
    # Start with original data
    entangled_field = data.copy()
    
    # Add subtle enhancements
    coupling = effective_omega * mixing_angle * 0.4  # Reduced coupling
    
    # Add dark photon modulation (preserve brightness)
    entangled_field = entangled_field * (1 - coupling * 0.3)
    entangled_field = entangled_field + dark_photon_field * coupling * 0.2
    
    # Add very subtle DM enhancement
    entangled_field = entangled_field + dm_density * coupling * 0.15
    
    # 4. GENTLE WAVE INTERFERENCE
    grad_x = sobel(entangled_field, axis=0)
    grad_y = sobel(entangled_field, axis=1)
    wave_interference = (np.sqrt(grad_x**2 + grad_y**2)) * mixing_angle * 0.15
    
    # Combine with original data bias
    final = entangled_field + wave_interference
    
    # Apply brightness adjustment
    final = final * brightness
    
    # Restore original brightness level
    final_mean = np.mean(final)
    if final_mean > 0:
        final = final * (original_mean / final_mean)
    
    final = np.clip(final, 0, 1)
    final = np.nan_to_num(final)
    
    # Create overlay (more subtle)
    overlay = np.clip(dark_photon_field * 0.4 + dm_density * 0.3, 0, 0.6)
    
    return final, overlay, dark_photon_field, dm_density, physical_wavelength


# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v9")
    st.markdown("**Photon-Dark-Photon Entangled FDM**")
    st.markdown("*Brightness-Preserved Version*")
    st.markdown("---")
    
    uploaded = st.file_uploader("Upload FITS or Image", type=["fits", "png", "jpg", "jpeg", "tif", "tiff"])
    
    st.markdown("---")
    st.markdown("### Parameters")
    
    omega = st.slider("Ω Entanglement Strength", 0.05, 0.6, 0.25, 
                       help="Controls coupling between baryonic and dark matter")
    
    fringe = st.slider("Fringe Scale (PDP λ)", 30, 100, 55,
                        help="Dark photon oscillation wavelength")
    
    brightness = st.slider("Brightness", 0.8, 1.8, 1.15,
                           help="Final image brightness")
    
    physical_scale = st.selectbox("Image Physical Scale", 
                                   ["50 kpc", "100 kpc", "200 kpc", "500 kpc"],
                                   index=1)
    scale_map = {"50 kpc": 50, "100 kpc": 100, "200 kpc": 200, "500 kpc": 500}
    physical_scale_kpc = scale_map[physical_scale]
    
    st.caption("Based on Tony Ford Model | v9 Brightness Fixed")
    st.success("✓ Preserves original brightness\n✓ Subtle enhancements\n✓ No darkening")


# ── MAIN PIPELINE ──────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled Fuzzy Dark Matter Overlays*")
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
            st.error(f"Error loading image: {e}")
            st.stop()
    
    # Ensure 2D and reasonable size
    if len(raw.shape) > 2:
        raw = raw[:, :, 0]
    
    MAX_SIZE = 1024
    if raw.shape[0] > MAX_SIZE or raw.shape[1] > MAX_SIZE:
        raw = raw[:MAX_SIZE, :MAX_SIZE]
    
    # Store original for reference
    original_raw = raw.copy()
    
    # Process pipeline
    with st.spinner("Processing with brightness-preserved PDP conversion..."):
        # Normalize but preserve characteristics
        norm = normalize(raw, preserve_brightness=True)
        
        # PSF correction (gentle)
        psf = psf_correct(norm, amount=0.4)
        
        # Super-resolution (gentle)
        sr = neural_sr_real(psf)
        
        # Ensure same size as original after SR
        if sr.shape != norm.shape:
            from skimage.transform import resize
            sr = resize(sr, norm.shape, preserve_range=True)
        
        # Entanglement with brightness preservation
        ent, overlay, dark_photon, dm_density, phys_wavelength = photon_dark_photon_entanglement_corrected(
            sr, omega, fringe, physical_scale_kpc, brightness
        )
        
        # Final safety
        ent = np.clip(np.nan_to_num(ent), 0, 1)
        overlay = np.clip(np.nan_to_num(overlay), 0, 0.8)
        
        # Ensure all arrays are 2D
        for arr_name in ['norm', 'psf', 'sr', 'ent', 'overlay']:
            arr = locals()[arr_name]
            if len(arr.shape) > 2:
                locals()[arr_name] = arr[:, :, 0]
    
    # Display PDP info
    st.info(f"📡 **PDP Parameters**: Dark photon λ = **{phys_wavelength:.1f} kpc** | "
            f"Fringe = {fringe} | Scale = {physical_scale_kpc} kpc | "
            f"Ω = {omega:.2f}")
    
    # ── DISPLAY RESULTS ────────────────────────────────────
    st.markdown("### Pipeline Results")
    
    # Create 2x2 grid for better visibility
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    def show_fig(img, title, cmap="inferno", size=(6, 5)):
        """Safe image display with larger size"""
        try:
            fig, ax = plt.subplots(figsize=size)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, color='white', fontsize=12, pad=10)
            ax.axis("off")
            fig.patch.set_facecolor('#0b0b1a')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not display {title}")
    
    with col1:
        show_fig(norm, "📷 Input (Normalized)", size=(6, 5))
    with col2:
        show_fig(psf, "🔍 PSF Corrected", size=(6, 5))
    with col3:
        show_fig(sr, "🧠 Neural SR", size=(6, 5))
    with col4:
        show_fig(ent, "✨ PDP Entangled", size=(6, 5))
    
    # ── DARK MATTER OVERLAY ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌌 Dark Matter Substructure (Subtle Overlay)")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            # Show base image
            ax.imshow(ent, cmap="inferno", alpha=0.7)
            # Overlay dark matter in cyan with low opacity
            overlay_colored = ax.imshow(overlay, cmap="cool", alpha=0.4)
            ax.set_title("Photon-Dark-Photon Entangled Overlay\n(Cyan = Dark Matter)", 
                        color='white', fontsize=12)
            ax.axis("off")
            plt.colorbar(overlay_colored, ax=ax, fraction=0.046, label="Dark Matter Density")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Overlay error: {e}")
    
    with col_b:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(dark_photon, cmap="plasma")
            ax.set_title("Dark Photon Oscillation Field\n(Quantum Interference)", 
                        color='white', fontsize=12)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, label="Field Amplitude")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Field display error: {e}")
    
    # ── COMPARISON ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before/After Comparison")
    
    try:
        from skimage.transform import resize
        if norm.shape != ent.shape:
            norm_resized = resize(norm, ent.shape, preserve_range=True)
        else:
            norm_resized = norm
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.imshow(original_raw, cmap="gray")
        ax1.set_title("Original Raw", color='white', fontsize=12)
        ax1.axis("off")
        
        ax2.imshow(norm_resized, cmap="inferno")
        ax2.set_title("Before Processing", color='white', fontsize=12)
        ax2.axis("off")
        
        ax3.imshow(ent, cmap="inferno")
        ax3.set_title("After (PDP Entangled)", color='white', fontsize=12)
        ax3.axis("off")
        
        fig.patch.set_facecolor('#0b0b1a')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Comparison error: {e}")
    
    # ── METRICS ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Enhancement Metrics")
    
    try:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            # Calculate mean brightness preservation
            original_brightness = np.mean(original_raw)
            final_brightness = np.mean(ent)
            brightness_ratio = final_brightness / (original_brightness + 1e-9)
            st.metric("Brightness Preservation", f"{brightness_ratio:.2f}x", 
                     delta="Good" if 0.8 < brightness_ratio < 1.2 else "Adjusted")
        
        with metric_col2:
            # Edge enhancement (should be moderate)
            from scipy.ndimage import sobel
            original_edges = np.sum(np.abs(sobel(norm_resized)))
            final_edges = np.sum(np.abs(sobel(ent)))
            edge_ratio = final_edges / (original_edges + 1e-9)
            st.metric("Edge Enhancement", f"{edge_ratio:.2f}x",
                     delta="Subtle" if edge_ratio < 2.5 else "Strong")
        
        with metric_col3:
            # Contrast (should be similar or slightly improved)
            original_contrast = np.std(norm_resized)
            final_contrast = np.std(ent)
            contrast_ratio = final_contrast / (original_contrast + 1e-9)
            st.metric("Contrast", f"{contrast_ratio:.2f}x",
                     delta="Preserved" if 0.8 < contrast_ratio < 1.3 else "Changed")
            
    except Exception as e:
        st.warning("Could not calculate all metrics")
    
    # ── DOWNLOAD ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        buf1 = io.BytesIO()
        plt.imsave(buf1, ent, cmap="inferno")
        st.download_button("📸 Download Entangled Image", buf1.getvalue(), 
                          f"pdp_entangled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    with col_d2:
        buf2 = io.BytesIO()
        plt.imsave(buf2, overlay, cmap="cool")
        st.download_button("🌌 Download Dark Matter Overlay", buf2.getvalue(),
                          f"dm_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    with col_d3:
        buf3 = io.BytesIO()
        plt.imsave(buf3, dark_photon, cmap="plasma")
        st.download_button("⚡ Download Dark Photon Field", buf3.getvalue(),
                          f"dark_photon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

else:
    st.info("✨ **Upload a FITS or image file to begin**\n\n"
            "**New in v9:**\n"
            "- ✓ Preserves original brightness\n"
            "- ✓ Subtle, natural-looking enhancements\n"
            "- ✓ No darkening artifacts\n"
            "- ✓ Realistic dark matter overlays\n\n"
            "*Based on the Tony Ford Model for dark matter substructure visualization*")
    
    # Show parameter guide
    with st.expander("📖 Parameter Guide"):
        st.markdown("""
        **Ω Entanglement Strength (0.05-0.6)**
        - Lower values = subtle dark matter effects
        - Higher values = more visible quantum interference
        
        **Fringe Scale (30-100)**
        - Lower = larger, smoother wave patterns
        - Higher = finer, more detailed interference
        
        **Brightness (0.8-1.8)**
        - Adjust final image brightness
        - Default 1.15 usually works well
        
        **Physical Scale**
        - Match to your image's actual size
        - 100 kpc is typical for galaxy clusters
        """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v9** | Brightness-Preserved PDP Conversion | Tony Ford Model")
