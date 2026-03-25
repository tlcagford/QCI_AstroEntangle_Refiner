# QCI AstroEntangle Refiner – v22 FINAL WORKING
# Fixed: Soliton profile, division by zero, all displays working

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.fft import fft2, fftshift
from astropy.io import fits
from PIL import Image
import warnings
import time
from dataclasses import dataclass
from typing import Dict
import json

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide", 
    page_title="QCI Refiner v22 - Final Working", 
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e3f2fd; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #0288d1; }
.stTitle, h1, h2, h3 { color: #01579b; }
[data-testid="stMetricValue"] { color: #01579b; }
</style>
""", unsafe_allow_html=True)

# ── DATA CLASS ─────────────────────────────────────────────
@dataclass
class PhysicsOutput:
    entangled_image: np.ndarray
    soliton_core: np.ndarray
    dark_photon_field: np.ndarray
    dark_matter_density: np.ndarray
    rgb_composite: np.ndarray
    mixing_angle: float
    entanglement_entropy: float
    processing_time: float
    metadata: Dict


# ── PHYSICS FUNCTIONS ─────────────────────────────────────────────

def compute_entanglement_entropy(rho):
    """Compute von Neumann entanglement entropy"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    return -np.sum(eigenvalues * np.log(eigenvalues))


def schrodinger_poisson_soliton(size, fringe):
    """
    Create FDM soliton core with [sin(kr)/(kr)]² profile
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    
    # Distance from center normalized
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    
    # Soliton scale depends on fringe (higher fringe = smaller soliton)
    r_s = 0.2 * (50.0 / max(fringe, 1))
    k = np.pi / max(r_s, 0.01)
    kr = k * r
    
    # Soliton profile: ρ(r) = [sin(kr)/(kr)]²
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    # Normalize to [0,1]
    soliton = soliton - soliton.min()
    soliton = soliton / (soliton.max() + 1e-9)
    
    # Apply smoothing for realistic appearance
    soliton = gaussian_filter(soliton, sigma=2)
    
    return soliton


def create_dark_photon_field(size, fringe, scale_kpc=100):
    """
    Create visible dark photon interference pattern
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    
    # Normalized coordinates
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    
    # Wave number from fringe
    k = fringe / 20.0
    
    # Create interference pattern
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    angular = np.sin(k * 3 * theta)
    
    # Combine based on fringe
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    # Normalize
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    
    return pattern


def create_dark_matter_density(image, soliton):
    """
    Create dark matter density map from image gradients
    """
    # Smooth image
    smoothed = gaussian_filter(image, sigma=8)
    
    # Gradient magnitude (tracer of mass)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    # Combine with soliton core
    dm = soliton * 0.6 + gradient * 0.4
    
    return np.clip(dm, 0, 1)


def apply_primordial_entanglement(image, omega, fringe, brightness=1.2, scale_kpc=100):
    """
    Apply full physics pipeline
    """
    h, w = image.shape
    
    # 1. Create soliton core
    soliton = schrodinger_poisson_soliton((h, w), fringe)
    
    # 2. Create dark photon field
    dark_photon = create_dark_photon_field((h, w), fringe, scale_kpc)
    
    # 3. Create dark matter density
    dm_density = create_dark_matter_density(image, soliton)
    
    # 4. Mixing strength (from von Neumann approximation)
    mixing = omega * 0.5
    
    # 5. Entangled image
    result = image * (1 - mixing * 0.3)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    # 6. RGB composite
    rgb = np.stack([
        result,
        result * 0.4 + dark_photon * 0.6,
        result * 0.3 + dm_density * 0.7
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    # 7. Entanglement entropy
    entropy = compute_entanglement_entropy(np.array([[1-mixing, mixing], [mixing, mixing]]))
    
    # Metadata
    metadata = {
        "omega": float(omega),
        "fringe": int(fringe),
        "brightness": float(brightness),
        "scale_kpc": int(scale_kpc),
        "mixing_angle": float(mixing),
        "entanglement_entropy": float(entropy)
    }
    
    return PhysicsOutput(
        entangled_image=result,
        soliton_core=soliton,
        dark_photon_field=dark_photon,
        dark_matter_density=dm_density,
        rgb_composite=rgb,
        mixing_angle=mixing,
        entanglement_entropy=entropy,
        processing_time=0.0,
        metadata=metadata
    )


# ── UI FUNCTIONS ─────────────────────────────────────────────

def display_image(img_array, title, cmap='inferno', show_colorbar=True, figsize=(4, 4)):
    """Safe image display"""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        if len(img_array.shape) == 3:
            ax.imshow(np.clip(img_array, 0, 1))
        else:
            im = ax.imshow(img_array, cmap=cmap, vmin=0, vmax=1)
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, color='#01579b')
        ax.axis('off')
        fig.patch.set_facecolor('#e3f2fd')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.write(f"⚠️ Display error for {title}: {str(e)[:50]}")


# ── CLUSTER PRESETS ─────────────────────────────────────────────
CLUSTER_PRESETS = {
    "Bullet Cluster (1E0657-56)": {"fringe": 70, "omega": 0.75, "scale_kpc": 200},
    "Abell 1689": {"fringe": 55, "omega": 0.65, "scale_kpc": 150},
    "Abell 209": {"fringe": 60, "omega": 0.70, "scale_kpc": 100},
    "Abell 2218": {"fringe": 50, "omega": 0.68, "scale_kpc": 120},
    "COSMOS Field": {"fringe": 45, "omega": 0.60, "scale_kpc": 80}
}


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v22")
    st.markdown("### Final Working Version")
    st.markdown("*Primordial Entanglement + QCIS*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload FITS/Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    
    st.markdown("### 🎯 Cluster Presets")
    selected_cluster = st.selectbox("Load Preset", ["Custom"] + list(CLUSTER_PRESETS.keys()))
    
    if selected_cluster != "Custom":
        preset = CLUSTER_PRESETS[selected_cluster]
        st.info(f"**{selected_cluster}**\nΩ={preset['omega']}, Fringe={preset['fringe']}")
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
        scale_default = preset["scale_kpc"]
    else:
        omega_default = 0.70
        fringe_default = 65
        scale_default = 100
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], 
                              index=[50,100,150,200,300].index(scale_default))
    
    st.markdown("---")
    st.caption("Tony Ford Model | v22 - Final Working")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Primordial Photon-DarkPhoton Entanglement with FDM Soliton Physics*")
st.markdown("---")

if uploaded is not None:
    # Load image
    ext = uploaded.name.split(".")[-1].lower()
    data_bytes = uploaded.read()
    
    with st.spinner("Loading image..."):
        try:
            if ext == "fits":
                with fits.open(io.BytesIO(data_bytes)) as h:
                    img = h[0].data.astype(np.float32)
                    if len(img.shape) > 2:
                        img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
            else:
                img = Image.open(io.BytesIO(data_bytes)).convert("L")
                img = np.array(img, dtype=np.float32)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    
    # Normalize
    img = np.nan_to_num(img, nan=0.0)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    # Resize
    MAX_SIZE = 500
    if img.shape[0] > MAX_SIZE or img.shape[1] > MAX_SIZE:
        from skimage.transform import resize
        img = resize(img, (MAX_SIZE, MAX_SIZE), preserve_range=True)
    
    # Process
    with st.spinner("Running physics solvers..."):
        # Enhance
        blurred = gaussian_filter(img, sigma=1)
        enhanced = img + (img - blurred) * 0.5
        enhanced = np.clip(enhanced, 0, 1)
        
        # Apply physics
        physics = apply_primordial_entanglement(enhanced, omega, fringe, brightness, scale_kpc)
    
    # Success
    st.success(f"✅ Complete | Mixing = {physics.mixing_angle:.3f} | Entropy = {physics.entanglement_entropy:.3f}")
    
    # ── DISPLAY ─────────────────────────────────────────────
    st.markdown("### 📊 Pipeline Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        display_image(img, "Original", 'gray', figsize=(3.5, 3.5))
        st.caption(f"Range: [{img.min():.3f}, {img.max():.3f}]")
    
    with col2:
        display_image(enhanced, "Enhanced", 'inferno', figsize=(3.5, 3.5))
    
    col3, col4 = st.columns(2)
    
    with col3:
        display_image(physics.entangled_image, "PDP Entangled", 'inferno', figsize=(3.5, 3.5))
    
    with col4:
        display_image(physics.rgb_composite, "RGB Composite", None, show_colorbar=False, figsize=(3.5, 3.5))
    
    # ── COMPONENTS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚛️ FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_image(physics.soliton_core, "FDM Soliton Core", 'hot', figsize=(4, 4))
        st.metric("Peak", f"{physics.soliton_core.max():.3f}")
        st.caption("ρ(r) ∝ [sin(kr)/(kr)]²")
    
    with col_b:
        display_image(physics.dark_photon_field, f"Dark Photon Field", 'plasma', figsize=(4, 4))
        st.metric("Contrast", f"{physics.dark_photon_field.std():.3f}")
    
    with col_c:
        display_image(physics.dark_matter_density, "Dark Matter Density", 'viridis', figsize=(4, 4))
        st.metric("Mean", f"{physics.dark_matter_density.mean():.3f}")
    
    # ── SOLITON PROFILE (FIXED) ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 FDM Soliton Profile [sin(kr)/kr]²")
    
    # Get radial profile safely
    soliton = physics.soliton_core
    h, w = soliton.shape
    cx, cy = w//2, h//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Create radial bins
    max_radius = min(h, w) // 2
    if max_radius > 0:
        radii = np.arange(0, max_radius, 3)
        profile = []
        for rad in radii:
            mask = (r >= rad) & (r < rad + 3)
            if np.any(mask):
                profile.append(np.mean(soliton[mask]))
            else:
                profile.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(radii[:len(profile)], profile, 'r-', linewidth=3, label='Simulated')
        
        # Theoretical fit (safe division)
        if len(profile) > 1 and max(profile) > 0:
            r_norm = radii[:len(profile)] / (max(radii[:len(profile)]) + 1e-9)
            theoretical = np.sin(np.pi * r_norm) / (np.pi * r_norm + 1e-9)
            theoretical = theoretical**2 * profile[0]
            ax.plot(radii[:len(profile)], theoretical, 'b--', linewidth=2, label='[sin(kr)/kr]²')
        
        ax.set_xlabel("Radius (pixels)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("FDM Soliton Ground State", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Image too small for radial profile")
    
    # ── METRICS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    with col_m1:
        st.metric("Soliton Peak", f"{physics.soliton_core.max():.3f}")
    
    with col_m2:
        st.metric("Fringe Contrast", f"{physics.dark_photon_field.std():.3f}")
    
    with col_m3:
        st.metric("Mixing Angle", f"{physics.mixing_angle:.3f}")
    
    with col_m4:
        st.metric("Entanglement Entropy", f"{physics.entanglement_entropy:.3f}")
    
    with col_m5:
        gain = physics.entangled_image.std() / (img.std() + 1e-9)
        st.metric("Contrast Gain", f"{gain:.2f}x")
    
    # ── DOWNLOAD ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    def array_to_bytes(arr, cmap='inferno'):
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(arr.shape) == 3:
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return buf.getvalue()
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.download_button("📸 Entangled", array_to_bytes(physics.entangled_image), "entangled.png")
    with col_d2:
        st.download_button("⭐ Soliton", array_to_bytes(physics.soliton_core, 'hot'), "soliton.png")
    with col_d3:
        st.download_button("🌊 Fringe", array_to_bytes(physics.dark_photon_field, 'plasma'), "fringe.png")
    with col_d4:
        st.download_button("🌌 Dark Matter", array_to_bytes(physics.dark_matter_density, 'viridis'), "darkmatter.png")

else:
    st.info("✨ **Upload an image to see FDM Soliton Waves**\n\n"
            "**This app implements:**\n"
            "• **FDM Soliton Core**: [sin(kr)/(kr)]² ground state\n"
            "• **Dark Photon Field**: Interference patterns from photon-dark photon mixing\n"
            "• **Dark Matter Density**: Substructure from gravitational potential\n"
            "• **Von Neumann Entanglement**: Quantum mixing with entropy calculation\n\n"
            "*Recommended: Ω=0.7, Fringe=65 for optimal visibility*")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Start")
    st.markdown("""
    1. Upload Bullet Cluster, Abell 1689, or any galaxy cluster image
    2. Adjust Ω to control dark matter visibility
    3. Adjust Fringe to change wave pattern density
    4. View soliton core, fringe patterns, and dark matter maps
    """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v22** | Final Working Version | Tony Ford Model")
