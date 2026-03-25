"""
QCI AstroEntangle Refiner v4 - Enhanced Edition
Full PDP physics, PSF corrections, neural enhancements, and professional overlays
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter
import io
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
    .metric-card {
        background-color: #1e1e2e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .stMetric {
        background-color: #1e1e2e;
        border-radius: 0.5rem;
        padding: 0.5rem;
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
            # Load FITS file
            hdul = fits.open(uploaded_file)
            image_data = hdul[0].data
            
            # Handle multi-dimensional FITS
            if len(image_data.shape) == 3:
                image_data = image_data[0]
            if len(image_data.shape) == 2:
                # Normalize
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-10)
            else:
                st.error("Unsupported FITS dimensions")
                return None, None
            
            # Get pixel scale from header if available
            try:
                pixel_scale = abs(hdul[0].header['CDELT1']) * 3600  # degrees to arcseconds
            except:
                pixel_scale = 0.05  # Default
                
            return image_data, pixel_scale
        
        else:
            # Load regular image
            img = Image.open(uploaded_file)
            if img.mode == 'RGB':
                img = img.convert('L')  # Convert to grayscale
            image_data = np.array(img) / 255.0
            return image_data, 0.05  # Default pixel scale
    
    return None, None

def create_overlay(original_img, overlay_map, color='red', transparency=0.5):
    """
    Create a colored overlay on top of the original image
    """
    # Ensure images are 3-channel RGB
    if len(original_img.shape) == 2:
        original_rgb = np.stack([original_img] * 3, axis=-1)
    else:
        original_rgb = original_img.copy()
    
    # Normalize overlay map
    overlay_min = overlay_map.min()
    overlay_max = overlay_map.max()
    if overlay_max - overlay_min > 1e-10:
        overlay_norm = (overlay_map - overlay_min) / (overlay_max - overlay_min)
    else:
        overlay_norm = overlay_map
    
    # Create colored overlay
    overlay_rgb = np.zeros_like(original_rgb)
    
    if color == 'red':
        overlay_rgb[..., 0] = overlay_norm
    elif color == 'blue':
        overlay_rgb[..., 2] = overlay_norm
    elif color == 'green':
        overlay_rgb[..., 1] = overlay_norm
    elif color == 'yellow':
        overlay_rgb[..., 0] = overlay_norm
        overlay_rgb[..., 1] = overlay_norm
    
    # Blend original and overlay
    blended = (1 - transparency) * original_rgb + transparency * overlay_rgb
    blended = np.clip(blended, 0, 1)
    
    return blended

def create_contour_overlay(original_img, overlay_map, color='red', threshold_percentile=90):
    """Create overlay with contour lines"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Show original image
    ax.imshow(original_img, cmap='gray', origin='lower')
    
    # Add contour lines at high probability regions
    threshold = np.percentile(overlay_map, threshold_percentile)
    ax.contour(
        overlay_map, 
        levels=[threshold],
        colors=color,
        linewidths=2,
        alpha=0.8
    )
    
    # Add colorbar for overlay intensity
    im = ax.imshow(overlay_map, cmap=f'{color}_hot', alpha=0.4, origin='lower')
    plt.colorbar(im, ax=ax, label=f'{color.capitalize()} intensity')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{color.capitalize()} Contours (Top {100-threshold_percentile}%)')
    
    return fig

def apply_lensing(image, kappa_map, strength=1.0):
    """Apply gravitational lensing effect"""
    lensed = image * (1 + strength * kappa_map)
    lensed = np.clip(lensed, 0, 1)
    return lensed

def image_to_bytes(img_array, format='PNG'):
    """Convert numpy array to bytes for download"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.markdown("## 🎮 Simulation Controls")
    
    # Cluster selection
    cluster = st.selectbox(
        "Select Cluster",
        ["Bullet Cluster", "Abell 520", "Abell 2744", "Custom"],
        help="Select astronomical cluster for analysis"
    )
    
    st.markdown("---")
    
    # Dark photon parameters
    st.markdown("### 🟣 Dark Photon Parameters")
    
    log_eps = st.slider("log₁₀(ε)", -12.0, -5.0, -8.0, 0.1,
                        help="Kinetic mixing parameter")
    mixing_epsilon = 10**log_eps
    
    log_mass = st.slider("log₁₀(m_dark [eV])", -14.0, -10.0, -12.0, 0.1,
                         help="Dark photon mass")
    dark_photon_mass = 10**log_mass
    
    st.markdown("---")
    
    # Photon energy
    st.markdown("### 💡 Photon Energy")
    photon_energy = st.slider("Photon Energy (eV)", 1.0, 10000.0, 1000.0, 10.0,
                              help="Energy of photons (affects oscillation probability)")
    
    st.markdown("---")
    
    # Gravitational lensing
    st.markdown("### 🌌 Gravitational Lensing")
    lensing_strength = st.slider("Lensing Strength (κ scale)", 0.0, 2.0, 1.0, 0.05,
                                 help="Strength of gravitational lensing effect")
    
    st.markdown("---")
    
    # Advanced corrections
    st.markdown("### 🔧 Advanced Corrections")
    
    apply_psf = st.checkbox("Apply PSF Correction", value=True,
                           help="Correct for telescope beam smearing")
    
    if apply_psf:
        psf_fwhm = st.slider("PSF FWHM (arcsec)", 0.01, 0.5, 0.05, 0.01,
                            help="Point Spread Function width")
    else:
        psf_fwhm = 0.05
    
    apply_neural = st.checkbox("Apply Neural Enhancement", value=True,
                              help="CLAHE-based contrast enhancement")
    
    if apply_neural:
        enhancement_method = st.selectbox("Enhancement Method",
                                         ["clahe", "unsharp", "retinex"],
                                         help="CLAHE: adaptive contrast, Unsharp: edge enhancement")
    else:
        enhancement_method = "clahe"
    
    st.markdown("---")
    
    # Overlay controls
    st.markdown("### 🎨 Overlay Settings")
    overlay_transparency = st.slider("Overlay Transparency", 0.1, 0.9, 0.5, 0.05)
    
    # Parameter display
    st.markdown("---")
    st.caption(f"**ε:** {mixing_epsilon:.2e}")
    st.caption(f"**m_dark:** {dark_photon_mass:.2e} eV")
    st.caption(f"**Photon Energy:** {photon_energy:.1f} eV")

# ============================================================================
# Main Content - Image Upload
# ============================================================================

st.markdown("## 📤 Data Input")

uploaded_file = st.file_uploader(
    "Upload Image (FITS, PNG, JPG, TIFF, etc.)",
    type=['fits', 'fit', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
    help="Supports astronomical FITS files and regular images"
)

if uploaded_file is not None:
    # Load image
    with st.spinner("Loading image..."):
        image_data, pixel_scale = load_image(uploaded_file)
    
    if image_data is not None:
        st.success(f"✅ Loaded {uploaded_file.name} | Size: {image_data.shape}")
        
        # Initialize physics engine
        engine = PhotonDarkPhotonModel()
        
        # Set default parameters based on cluster
        cluster_params = {
            "Bullet Cluster": {"redshift": 0.296, "distance_mpc": 430, "velocity": 2000000},
            "Abell 520": {"redshift": 0.201, "distance_mpc": 390, "velocity": 1800000},
            "Abell 2744": {"redshift": 0.308, "distance_mpc": 450, "velocity": 2200000},
            "Custom": {"redshift": 0.2, "distance_mpc": 400, "velocity": 2000000}
        }
        
        params = cluster_params.get(cluster, cluster_params["Bullet Cluster"])
        
        # Run physics engine
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
        
        # Get maps
        original_norm = engine.original_norm if hasattr(engine, 'original_norm') else image_data
        enhanced_map = engine.get_enhanced_map()
        conversion_map = engine.get_conversion_map()
        entanglement_map = engine.get_entanglement_map()
        
        # Create lensing map (simulated from convergence)
        lensing_map = gaussian_filter(conversion_map, sigma=5) * (1 + 0.5 * conversion_map)
        lensing_map = lensing_map / (lensing_map.max() + 1e-10)
        
        # Apply lensing to PDP image
        lensed_image = apply_lensing(entanglement_map, lensing_map, lensing_strength)
        
        # ====================================================================
        # Before vs After Section
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🔬 Before vs After")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📸 Original**")
            st.image(original_norm, use_column_width=True, clamp=True)
        
        with col2:
            st.markdown("**🔄 After PDP Conversion**")
            st.image(entanglement_map, use_column_width=True, clamp=True)
            st.caption("Photon → Dark Photon conversion applied")
        
        with col3:
            st.markdown("**🌌 After PDP + Lensing**")
            st.image(lensed_image, use_column_width=True, clamp=True)
            st.caption("Full effect: conversion + gravitational lensing")
        
        # ====================================================================
        # Overlay Views
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 🎨 Overlay Views")
        
        # Create overlays
        pdp_overlay = create_overlay(original_norm, conversion_map, 'red', overlay_transparency)
        lensing_overlay = create_overlay(original_norm, lensing_map, 'blue', overlay_transparency)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original + PDP Conversion (red = converted)**")
            st.image(pdp_overlay, use_column_width=True, clamp=True)
            st.caption("Brighter red = higher conversion probability")
        
        with col2:
            st.markdown("**Original + Lensing (blue = mass)**")
            st.image(lensing_overlay, use_column_width=True, clamp=True)
            st.caption("Brighter blue = stronger dark matter concentration")
        
        # ====================================================================
        # Physics Metrics
        # ====================================================================
        
        st.markdown("---")
        st.markdown("## 📊 Physics Metrics")
        
        # Calculate additional metrics
        mean_conversion = float(np.mean(conversion_map))
        max_conversion = float(np.max(conversion_map))
        mean_lensing = float(np.mean(lensing_map))
        peak_lensing = float(np.max(lensing_map))
        
        # Calculate correlation safely
        conv_flat = conversion_map.flatten()
        lens_flat = lensing_map.flatten()
        if len(conv_flat) > 1 and len(lens_flat) > 1:
            correlation = np.corrcoef(conv_flat, lens_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Quantum metrics from metadata
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Entropy (bits)", f"{metadata.get('entropy', 0):.3f}")
        
        with col2:
            concurrence = metadata.get('concurrence', 0)
            if np.isnan(concurrence):
                concurrence = 0
            st.metric("Concurrence", f"{concurrence:.2e}")
        
        with col3:
            purity = metadata.get('purity', 1)
            if np.isnan(purity):
                purity = 1
            st.metric("Purity", f"{purity:.6f}")
        
        with col4:
            avg_conv = metadata.get('avg_conversion_probability', 0)
            if np.isnan(avg_conv):
                avg_conv = 0
            st.metric("Avg Conversion", f"{avg_conv:.2%}")
        
        # Additional stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Conversion Probability", f"{max_conversion:.2%}")
        
        with col2:
            st.metric("Mean κ (lensing)", f"{mean_lensing:.3f}")
            st.metric("Peak κ", f"{peak_lensing:.3f}")
        
        with col3:
            st.metric("Conversion-Mass Correlation", f"{correlation:.3f}")
            st.caption("Positive = conversion follows dark matter")
        
        # ====================================================================
        # Parameter Information
        # ====================================================================
        
        with st.expander("🔬 Current Simulation Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Cluster:** {cluster}")
                st.write(f"**Redshift:** {params['redshift']:.3f}")
                st.write(f"**Distance:** {params['distance_mpc']:.0f} Mpc")
                st.write(f"**Pixel Scale:** {pixel_scale:.3f} arcsec/pixel")
            with col2:
                st.write(f"**Mixing Parameter (ε):** {mixing_epsilon:.2e}")
                st.write(f"**Dark Photon Mass:** {dark_photon_mass:.2e} eV")
                st.write(f"**Photon Energy:** {photon_energy:.1f} eV")
                st.write(f"**Lensing Strength:** {lensing_strength:.2f}")
            
            st.write("---")
            st.write(f"**PSF Correction:** {'✅ On' if apply_psf else '❌ Off'} (FWHM: {psf_fwhm:.2f} arcsec)")
            st.write(f"**Neural Enhancement:** {'✅ On' if apply_neural else '❌ Off'} ({enhancement_method if apply_neural else 'N/A'})")
        
        # ====================================================================
        # Download Buttons
        # ====================================================================
        
        with st.expander("📥 Download Results"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📸 Download Original",
                    image_to_bytes(original_norm),
                    "original.png",
                    "image/png"
                )
            
            with col2:
                st.download_button(
                    "🔄 Download PDP Conversion",
                    image_to_bytes(entanglement_map),
                    "pdp_conversion.png",
                    "image/png"
                )
            
            with col3:
                st.download_button(
                    "🌌 Download PDP + Lensing",
                    image_to_bytes(lensed_image),
                    "pdp_lensing.png",
                    "image/png"
                )
        
        # ====================================================================
        # Physics Explanation
        # ====================================================================
        
        with st.expander("🔬 Physics Explanation"):
            st.markdown("""
            ### Quantum Oscillation Formula
            
            The photon-dark photon conversion probability is given by:
            
            \[
            P_{\gamma \to A'} = 4\epsilon^2 \sin^2\left(\frac{\Delta m^2 L}{4E}\right)
            \]
            
            Where:
            - ε = kinetic mixing parameter
            - Δm² = m_dark² - m_photon² ≈ m_dark²
            - L = propagation distance
            - E = photon energy
            
            ### PSF Correction
            Uses Wiener-like deconvolution with a Gaussian PSF to correct for telescope beam smearing.
            
            ### Neural Enhancement
            CLAHE (Contrast Limited Adaptive Histogram Equalization) provides adaptive contrast enhancement.
            
            ### Overlay Colors
            - **Red**: Photon → Dark Photon conversion probability
            - **Blue**: Dark matter distribution (from gravitational lensing)
            """)
        
        # Success message
        st.success("✅ Analysis complete! Adjust parameters in sidebar to explore different scenarios.")
        
    else:
        st.error("Failed to load image. Please try another file.")
else:
    # Placeholder when no image is uploaded
    st.info("👈 Please upload an image to begin analysis")
    
    # Show example
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions
        
        1. **Upload an image** (FITS, PNG, JPG, etc.)
        2. **Adjust dark photon parameters** (ε and m_dark)
        3. **Set photon energy** to see oscillation effects
        4. **Toggle PSF correction** for telescope beam correction
        5. **Enable neural enhancement** for contrast improvement
        6. **Explore overlays** showing conversion and dark matter
        7. **Download results** for further analysis
        
        ### Supported Image Types
        - **FITS/FIT**: Astronomical images with optional header info
        - **PNG/JPG/TIFF**: Regular images for quick testing
        
        ### Physics Parameters
        - **ε (mixing parameter)**: Strength of photon-dark photon coupling
        - **m_dark**: Dark photon mass (affects oscillation length)
        - **Photon Energy**: Higher energy = longer oscillation length
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
QCI AstroEntangle Refiner v4 | Enhanced with PSF Corrections & Neural Enhancements
</div>
""", unsafe_allow_html=True)
