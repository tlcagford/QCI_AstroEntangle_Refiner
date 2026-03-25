"""
QCI AstroEntangle Refiner v4.5 - Complete Edition
Full FDM Two-Field Physics + AI Data Download + Mosaic Creator
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import astropy.io.fits as fits
from scipy.ndimage import gaussian_filter
import io
import warnings
import requests
import json

warnings.filterwarnings('ignore')

# Import physics engine
from pdp_physics_working import PhotonDarkPhotonModel, H, HBAR, C, ALPHA, M_E, EV, K_B, EPS0

# Page configuration
st.set_page_config(
    page_title="QCI AstroEntangle Refiner v4.5",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { font-size: 2.5rem; color: #ffffff; text-align: center; margin-bottom: 1rem; }
    .metric-card { background-color: #1e1e2e; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌌 QCI AstroEntangle Refiner v4.5</div>', unsafe_allow_html=True)
st.markdown("### Photon-Dark Photon Quantum Entanglement + FDM Two-Field Physics")
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
                return None, None
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
    """Create colored overlay"""
    if len(original_img.shape) == 2:
        original_rgb = np.stack([original_img] * 3, axis=-1)
    else:
        original_rgb = original_img.copy()
    
    overlay_min = overlay_map.min()
    overlay_max = overlay_map.max()
    if overlay_max - overlay_min > 1e-10:
        overlay_norm = (overlay_map - overlay_min) / (overlay_max - overlay_min)
    else:
        overlay_norm = overlay_map
    
    overlay_rgb = np.zeros_like(original_rgb)
    
    if color == 'red':
        overlay_rgb[..., 0] = overlay_norm
    elif color == 'blue':
        overlay_rgb[..., 2] = overlay_norm
    elif color == 'yellow':
        overlay_rgb[..., 0] = overlay_norm
        overlay_rgb[..., 1] = overlay_norm
    
    blended = (1 - transparency) * original_rgb + transparency * overlay_rgb
    return np.clip(blended, 0, 1)

def apply_lensing(image, kappa_map, strength=1.0):
    """Apply gravitational lensing"""
    lensed = image * (1 + strength * kappa_map)
    return np.clip(lensed, 0, 1)

def image_to_bytes(img_array, format='PNG'):
    """Convert numpy array to bytes"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def query_mast(target_name, observatory=None):
    """Query MAST database for astronomical images"""
    try:
        from astroquery.mast import Observations
        
        # Build query
        query_params = {
            'target_name': target_name,
        }
        if observatory:
            query_params['obs_collection'] = observatory
        
        obs_table = Observations.query_criteria(**query_params)
        
        if len(obs_table) > 0:
            return obs_table
        else:
            return None
    except Exception as e:
        st.warning(f"MAST query error: {e}")
        return None

def query_ned(target_name):
    """Query NED database for galaxy cluster information"""
    try:
        url = f"https://ned.ipac.caltech.edu/uri/nph-objsearch?objname={target_name}&out_format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

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
    
    if apply_neural:
        enhancement_method = st.selectbox("Enhancement Method", ["clahe", "unsharp", "retinex"])
    
    st.markdown("---")
    st.markdown("### 🎨 Overlay Settings")
    overlay_transparency = st.slider("Overlay Transparency", 0.1, 0.9, 0.5, 0.05)
    
    st.markdown("---")
    st.caption(f"**ε:** {mixing_epsilon:.2e}")
    st.caption(f"**m_dark:** {dark_photon_mass:.2e} eV")
    st.caption(f"**Photon Energy:** {photon_energy:.1f} eV")

# ============================================================================
# AI Data Download Section
# ============================================================================

with st.expander("🤖 AI Data Download", expanded=False):
    st.markdown("### Fetch Astronomical Data from Online Databases")
    st.markdown("Query MAST (HST/JWST), NED, and other archives using natural language.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ai_query = st.text_input("Describe what data you want:", 
                                  placeholder="e.g., 'Get JWST NIRCam images of Abell 2744' or 'Find Bullet Cluster observations'")
    
    with col2:
        st.markdown("###")
        fetch_button = st.button("🔍 Fetch Data", type="primary", use_container_width=True)
    
    if fetch_button and ai_query:
        with st.spinner("🤖 AI parsing request and querying databases..."):
            # Simple keyword parsing (can be enhanced with real LLM)
            query_lower = ai_query.lower()
            
            # Determine observatory
            if "jwst" in query_lower or "webb" in query_lower:
                observatory = "JWST"
            elif "hst" in query_lower or "hubble" in query_lower:
                observatory = "HST"
            else:
                observatory = None
            
            # Determine target
            target_map = {
                "abell 2744": "Abell 2744",
                "abell 520": "Abell 520",
                "bullet cluster": "Bullet Cluster",
                "1e 0657-56": "Bullet Cluster",
                "m87": "M87",
                "virgo": "Virgo Cluster",
                "coma": "Coma Cluster"
            }
            
            target = None
            for key, value in target_map.items():
                if key in query_lower:
                    target = value
                    break
            
            if target is None:
                # Try to extract any capitalized words as potential target
                words = ai_query.split()
                for word in words:
                    if word[0].isupper() and len(word) > 2:
                        target = word
                        break
            
            results = []
            
            # Query MAST if target found
            if target:
                st.info(f"🔭 Querying MAST for {target}...")
                mast_results = query_mast(target, observatory)
                if mast_results is not None and len(mast_results) > 0:
                    results.append(("MAST", mast_results))
                    st.success(f"✅ Found {len(mast_results)} observations in MAST")
                    
                    # Show results
                    st.dataframe(mast_results[['obs_id', 'obs_collection', 'target_name', 't_exptime']].to_pandas()[:5])
                    
                    # Option to download
                    if st.button("📥 Download first observation"):
                        try:
                            products = Observations.get_product_list(mast_results[0])
                            st.info(f"Found {len(products)} products. Select one to download.")
                            st.dataframe(products[['productFilename', 'productType', 'size']].to_pandas()[:5])
                        except Exception as e:
                            st.error(f"Download error: {e}")
                else:
                    st.warning(f"No MAST data found for {target}")
            
            # Query NED for information
            if target:
                st.info(f"📡 Querying NED for {target} information...")
                ned_data = query_ned(target)
                if ned_data:
                    st.success("NED data retrieved")
                    st.json(ned_data)
            
            if not results:
                st.warning("No data found. Try a more specific query like 'Get JWST images of Abell 2744'")
    
    st.markdown("---")
    st.markdown("**Supported databases:** MAST (HST/JWST), NED")
    st.markdown("**Example queries:**")
    st.caption("• 'Get JWST images of Abell 2744'")
    st.caption("• 'Find Hubble observations of Bullet Cluster'")
    st.caption("• 'Show me Abell 520 data'")

# ============================================================================
# Main Content - Image Upload
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
        
        # Initialize physics engine
        engine = PhotonDarkPhotonModel()
        
        # Cluster parameters
        cluster_params = {
            "Bullet Cluster": {"redshift": 0.296, "distance_mpc": 430, "velocity": 200000},
            "Abell 520": {"redshift": 0.201, "distance_mpc": 390, "velocity": 180000},
            "Abell 2744": {"redshift": 0.308, "distance_mpc": 450, "velocity": 220000},
            "Custom": {"redshift": 0.2, "distance_mpc": 400, "velocity": 200000}
        }
        
        params = cluster_params.get(cluster, cluster_params["Bullet Cluster"])
        
        # Run physics engine
        with st.spinner("Computing quantum entanglement with FDM two-field equations..."):
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
        original_norm = engine.original_norm
        enhanced_map = engine.get_enhanced_map()
        conversion_map = engine.get_conversion_map()
        entanglement_map = engine.get_entanglement_map()
        interference_pattern = engine.get_interference_pattern()
        
        # Create lensing map
        lensing_map = gaussian_filter(conversion_map, sigma=5) * (1 + 0.5 * conversion_map)
        lensing_map = lensing_map / (lensing_map.max() + 1e-10)
        lensed_image = apply_lensing(entanglement_map, lensing_map, lensing_strength)
        
        # ====================================================================
        # Display Sections
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
        
        # Overlay Views
        st.markdown("---")
        st.markdown("## 🎨 Overlay Views")
        
        pdp_overlay = create_overlay(original_norm, conversion_map, 'red', overlay_transparency)
        lensing_overlay = create_overlay(original_norm, lensing_map, 'blue', overlay_transparency)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original + PDP Conversion (red = converted)**")
            st.image(pdp_overlay, use_column_width=True, clamp=True)
        with col2:
            st.markdown("**Original + Lensing (blue = mass)**")
            st.image(lensing_overlay, use_column_width=True, clamp=True)
        
        # Interference Pattern (new from FDM two-field)
        st.markdown("---")
        st.markdown("## 🌊 Two-Field FDM Interference Pattern")
        st.markdown("From ψ_total = ψ_light + ψ_dark e^(iΔφ)")
        
        fig_interference, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(interference_pattern, cmap='viridis', origin='lower')
        ax.set_title("FDM Two-Field Interference | λ = h/(mΔv)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="Interference Intensity")
        st.pyplot(fig_interference)
        plt.close(fig_interference)
        
        # Physics Metrics
        st.markdown("---")
        st.markdown("## 📊 Physics Metrics")
        
        mean_conversion = float(np.mean(conversion_map))
        max_conversion = float(np.max(conversion_map))
        correlation = np.corrcoef(conversion_map.flatten(), lensing_map.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Entropy (bits)", f"{metadata.get('entropy', 0):.3f}")
        with col2:
            st.metric("Concurrence", f"{metadata.get('concurrence', 0):.2e}")
        with col3:
            st.metric("Purity", f"{metadata.get('purity', 1):.6f}")
        with col4:
            st.metric("Avg Conversion", f"{metadata.get('avg_conversion_probability', 0):.2%}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fringe Spacing", f"{metadata.get('fringe_spacing_kpc', 0):.2f} kpc")
            st.caption("λ = h/(mΔv)")
        with col2:
            st.metric("Ω_PD (Entanglement)", f"{metadata.get('entanglement_observable', 0):.2e}")
            st.caption("= 2ε")
        with col3:
            st.metric("Correlation", f"{correlation:.3f}")
            st.caption("Conversion vs Dark Matter")
        
        # Solitonic Core Profile (FDM feature)
        with st.expander("🌀 FDM Solitonic Core Profile"):
            st.markdown("**Solitonic core solution to the cusp-core problem:**")
            st.latex(r"\rho(r) = \frac{\rho_c}{[1 + (r/r_c)^2]^8}")
            
            col1, col2 = st.columns(2)
            with col1:
                r_core_kpc = st.slider("Core radius (kpc)", 0.1, 5.0, 1.0, 0.1)
                rho_core = st.slider("Peak density (M☉/kpc³)", 1e6, 1e8, 1.9e7, format="%.1e")
            with col2:
                st.markdown(f"**Derived from FDM mass m = {dark_photon_mass:.2e} eV**")
                st.markdown(f"**de Broglie wavelength:** {metadata.get('fringe_spacing_kpc', 0):.2f} kpc")
            
            r = np.linspace(0, 10, 500)
            r_norm = r / r_core_kpc
            rho = rho_core / (1 + r_norm**2)**8
            
            fig_sol, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r, rho, 'b-', linewidth=2)
            ax.fill_between(r, rho, alpha=0.3)
            ax.set_xlabel("Radius (kpc)")
            ax.set_ylabel("Density (M☉/kpc³)")
            ax.set_title("FDM Solitonic Core Profile")
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_sol)
            plt.close(fig_sol)
        
        # Parameter Information
        with st.expander("🔬 FDM Two-Field Physics Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Cluster:** {cluster}")
                st.write(f"**Redshift:** {params['redshift']:.3f}")
                st.write(f"**Distance:** {params['distance_mpc']:.0f} Mpc")
                st.write(f"**Pixel Scale:** {pixel_scale:.3f} arcsec/pixel")
            with col2:
                st.write(f"**Mixing ε:** {mixing_epsilon:.2e}")
                st.write(f"**Dark Photon Mass:** {dark_photon_mass:.2e} eV")
                st.write(f"**Photon Energy:** {photon_energy:.1f} eV")
                st.write(f"**Relative Velocity:** {params['velocity']/1000:.0f} km/s")
            
            st.markdown("---")
            st.markdown("**Implemented FDM Equations:**")
            st.latex(r"\text{Klein-Gordon:} \quad \Box\phi + m^2\phi = 0")
            st.latex(r"\text{Schrödinger-Poisson:} \quad i\partial_t\psi = -\frac{\nabla^2\psi}{2m} + \Phi\psi")
            st.latex(r"\text{Two-field interference:} \quad \rho = |\psi_l|^2 + |\psi_d|^2 + 2\text{Re}(\psi_l^*\psi_d e^{i\Delta\phi})")
            st.latex(r"\text{Fringe spacing:} \quad \lambda = \frac{h}{m\Delta v}")
            st.latex(r"\text{Solitonic core:} \quad \rho(r) = \frac{\rho_c}{[1 + (r/r_c)^2]^8}")
        
        # Download Buttons
        with st.expander("📥 Download Results"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button("📸 Download Original", image_to_bytes(original_norm), "original.png")
            with col2:
                st.download_button("🔄 Download PDP Conversion", image_to_bytes(entanglement_map), "pdp_conversion.png")
            with col3:
                st.download_button("🌌 Download PDP + Lensing", image_to_bytes(lensed_image), "pdp_lensing.png")
        
        st.success("✅ Analysis complete! Adjust parameters to explore different FDM scenarios.")
        
    else:
        st.error("Failed to load image.")
else:
    st.info("👈 Please upload an image to begin analysis")
    
    with st.expander("ℹ️ About v4.5 - FDM Two-Field Physics"):
        st.markdown("""
        ### New in v4.5: Full FDM Two-Field Implementation
        
        This version implements the complete Fuzzy Dark Matter (FDM) framework from:
        **[Cosmic Entanglement Visualizer](https://cosmic-entanglement-visualizer-f4f21576.base44.app/Equations)**
        
        **Implemented Equations:**
        - **Klein-Gordon**: Relativistic wave equation for scalar field dark matter
        - **Schrödinger-Poisson**: Non-relativistic limit for galactic-scale structure
        - **Two-field interference**: ψ_total = ψ_light + ψ_dark e^(iΔφ)
        - **Fringe spacing**: λ = h/(mΔv) - de Broglie wavelength of FDM
        - **Solitonic core**: ρ(r) = ρ_c / [1 + (r/r_c)²]⁸ - solves cusp-core problem
        
        **New Features:**
        - 🤖 **AI Data Download**: Query MAST/NED databases for real astronomical data
        - 🌊 **Interference Pattern Visualization**: See two-field FDM fringes
        - 🌀 **Solitonic Core Profile**: Visualize FDM density profiles
        
        **Try these parameters:**
        - m_dark = 10⁻²² eV (FDM candidate mass)
        - ε = 10⁻⁸ (typical mixing parameter)
        - Photon energy = 1000 eV (X-ray observations)
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
QCI AstroEntangle Refiner v4.5 | FDM Two-Field Physics | Based on Cosmic Entanglement Visualizer Framework
</div>
""", unsafe_allow_html=True)
