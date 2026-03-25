"""
QCI AstroEntangle Refiner v4 - Enhanced Edition (No Plotly)
Full PDP physics, PSF corrections, neural enhancements, and professional overlays
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Import physics engine
from pdp_physics_working import PhotonDarkPhotonModel, H, HBAR, C, ALPHA, M_E, EV, K_B, EPS0

# Page configuration
st.set_page_config(
    page_title="QCI AstroEntangle Refiner v4",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌌 QCI AstroEntangle Refiner v4</div>', unsafe_allow_html=True)
st.markdown("### Photon-Dark Photon Quantum Entanglement + Gravitational Lensing")
st.markdown("---")

# ============================================================================
# Helper Functions
# ============================================================================

def load_image(uploaded_file):
    """Load image from various formats"""
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['fits', 'fit']:
            hdul = fits.open(uploaded_file)
            image_data = hdul[0].data
            if len(image_data.shape) == 3:
                image_data = image_data[0]
            if len(image_data.shape) == 2:
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-10)
            else:
                return None, 0.05
            try:
                pixel_scale = abs(hdul[0].header['CDELT1']) * 3600
            except:
                pixel_scale = 0.05
            return image_data, pixel_scale
        else:
            img = Image.open(uploaded_file)
            if img.mode == 'RGB':
                img = img.convert('L')
            image_data = np.array(img) / 255.0
            return image_data, 0.05
    return None, None

def create_overlay(original_img, overlay_map, color='red', transparency=0.5):
    """Create a colored overlay on top of the original image"""
    if len(original_img.shape) == 2:
        original_rgb = np.stack([original_img] * 3, axis=-1)
    else:
        original_rgb = original_img.copy()
    
    overlay_norm = (overlay_map - overlay_map.min()) / (overlay_map.max() - overlay_map.min() + 1e-10)
    overlay_rgb = np.zeros_like(original_rgb)
    
    if color == 'red':
        overlay_rgb[..., 0] = overlay_norm
    elif color == 'blue':
        overlay_rgb[..., 2] = overlay_norm
    elif color == 'yellow':
        overlay_rgb[..., 0] = overlay_norm
        overlay_rgb[..., 1] = overlay_norm
    
    blended = (1 - transparency) * original_rgb + transparency * overlay_rgb
    blended = np.clip(blended, 0, 1)
    return blended

def apply_lensing(image, kappa_map, strength=1.0):
    """Apply gravitational lensing effect"""
    lensed = image * (1 + strength * kappa_map)
    return np.clip(lensed, 0, 1)

# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.markdown("## 🎮 Simulation Controls")
    
    cluster = st.selectbox("Select Cluster", ["Bullet Cluster", "Abell 520", "Abell 2744", "Custom"])
    
    st.markdown("---")
    st.markdown("### 🟣 Dark Photon Parameters")
    
    log_eps = st.slider("log₁₀(ε)", -12.0, -5.0, -8.0, 0.1)
    mixing_epsilon = 10**log_eps
    
    log_mass = st.slider("log₁₀(m_dark [eV])", -14.0, -10.0, -12.0, 0.1)
    dark_photon_mass = 10**log_mass
    
    st.markdown("---")
    st.markdown("### 💡 Photon Energy")
    photon_energy = st.slider("Photon Energy (eV)", 1.0, 10000.0, 1000.0, 10.0)
    
    st.markdown("---")
    st.markdown("### 🌌 Gravitational Lensing")
    lensing_strength = st.slider("Lensing Strength (κ scale)", 0.0, 2.0, 1.0, 0.05)
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Corrections")
    
    apply_psf = st.checkbox("Apply PSF Correction", value=True)
    if apply_psf:
        psf_fwhm = st.slider("PSF FWHM (arcsec)", 0.01, 0.5, 0.05, 0.01)
    else:
        psf_fwhm = 0.05
    
    apply_neural = st.checkbox("Apply Neural Enhancement", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Overlay Settings")
    overlay_transparency = st.slider("Overlay Transparency", 0.1, 0.9, 0.5, 0.05)

# ============================================================================
# Main Content
# ============================================================================

st.markdown("## 📤 Data Input")

uploaded_file = st.file_uploader(
    "Upload Image (FITS, PNG, JPG, TIFF, etc.)",
    type=['fits', 'fit', 'png', 'jpg', 'jpeg', 'tif', 'tiff']
)

if uploaded_file is not None:
    with st.spinner("Loading image..."):
        image_data, pixel_scale = load_image(uploaded_file)
    
    if image_data is not None:
        st.success(f"✅ Loaded {uploaded_file.name} | Size: {image_data.shape}")
        
        engine = PhotonDarkPhotonModel()
        
        cluster_params = {
            "Bullet Cluster": {"redshift": 0.296, "distance_mpc": 430, "velocity": 2000000},
            "Abell 520": {"redshift": 0.201, "distance_mpc": 390, "velocity": 1800000},
            "Abell 2744": {"redshift": 0.308, "distance_mpc": 450, "velocity": 2200000},
            "Custom": {"redshift": 0.2, "distance_mpc": 400, "velocity": 2000000}
        }
        
        params = cluster_params.get(cluster, cluster_params["Bullet Cluster"])
        
        with st.spinner("Computing quantum entanglement..."):
            metadata = engine.initialize_from_image(
                image_data=image_data,
                pixel_scale_arcsec=pixel_scale,
                dark_photon_mass_eV=dark_photon_mass,
                mixing_epsilon=mixing_epsilon,
                relative_velocity=params["velocity"],
                redshift=params["redshift"],
                distance_mpc=params["distance_mpc"],
                E_photon_eV=photon_energy,
                apply_psf=apply_psf,
                apply_neural=apply_neural,
                psf_fwhm_arcsec=psf_fwhm
            )
        
        original_norm = engine.original_norm if hasattr(engine, 'original_norm') else image_data
        conversion_map = engine.get_conversion_map()
        entanglement_map = engine.get_entanglement_map()
        
        lensing_map = gaussian_filter(conversion_map, sigma=5) * (1 + 0.5 * conversion_map)
        lensing_map = lensing_map / lensing_map.max()
        lensed_image = apply_lensing(entanglement_map, lensing_map, lensing_strength)
        
        # Display images
        st.markdown("---")
        st.markdown("## 🔬 Before vs After")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📸 Original**")
            st.image(original_norm, use_column_width=True, clamp=True)
        
        with col2:
            st.markdown("**🔄 After PDP Conversion**")
            st.image(entanglement_map, use_column_width=True, clamp=True)
        
        with col3:
            st.markdown("**🌌 After PDP + Lensing**")
            st.image(lensed_image, use_column_width=True, clamp=True)
        
        # Overlays
        st.markdown("---")
        st.markdown("## 🎨 Overlay Views")
        
        pdp_overlay = create_overlay(original_norm, conversion_map, 'red', overlay_transparency)
        lensing_overlay = create_overlay(original_norm, lensing_map, 'blue', overlay_transparency)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original + PDP Conversion (red)**")
            st.image(pdp_overlay, use_column_width=True, clamp=True)
        
        with col2:
            st.markdown("**Original + Lensing (blue)**")
            st.image(lensing_overlay, use_column_width=True, clamp=True)
        
        # Metrics
        st.markdown("---")
        st.markdown("## 📊 Physics Metrics")
        
        mean_conversion = np.mean(conversion_map)
        correlation = np.corrcoef(conversion_map.flatten(), lensing_map.flatten())[0,1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entropy", f"{metadata['entropy']:.3f} bits")
        with col2:
            st.metric("Concurrence", f"{metadata['concurrence']:.4f}")
        with col3:
            st.metric("Purity", f"{metadata['purity']:.4f}")
        with col4:
            st.metric("Avg Conversion", f"{mean_conversion:.2%}")
        
        st.success("✅ Analysis complete!")
        
else:
    st.info("👈 Please upload an image to begin analysis")

st.markdown("---")
st.markdown("QCI AstroEntangle Refiner v4 | Enhanced with PSF Corrections & Neural Enhancements")
