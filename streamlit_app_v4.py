"""
QCI AstroEntangle Refiner v4.5 - Light Theme Edition
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
import re
import urllib.request
import json

warnings.filterwarnings('ignore')

# Import physics engine
from pdp_physics_working import PhotonDarkPhotonModel, H, HBAR, C, ALPHA, M_E, EV, K_B, EPS0

# ============================================================================
# LIGHT THEME CSS
# ============================================================================

st.set_page_config(
    page_title="QCI AstroEntangle Refiner v4.5",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for LIGHT THEME
st.markdown("""
<style>
    /* Main background - light gradient */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    /* Subheader */
    .sub-header {
        text-align: center;
        color: #495057;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Sidebar - light background */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #212529;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1e3c72;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30,60,114,0.3);
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #1e3c72;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1e3c72;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        color: #495057;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        color: #1e3c72;
        font-weight: 500;
    }
    .streamlit-expanderContent {
        background-color: #fafbfc;
        border-radius: 0 0 8px 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #495057;
        border: 1px solid #e9ecef;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Images */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        background-color: #ffffff;
        padding: 4px;
    }
    
    /* Captions */
    .stCaption {
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Success/Info/Warning boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #1e3c72 !important;
    }
    
    /* Selectbox */
    [data-baseweb="select"] {
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    /* Number input */
    [data-baseweb="input"] {
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    /* Dividers */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dee2e6, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌌 QCI AstroEntangle Refiner v4.5</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Photon-Dark Photon Quantum Entanglement + FDM Two-Field Physics</div>', unsafe_allow_html=True)
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

# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.markdown("## 🎮 Simulation Controls")
    
    cluster = st.selectbox("Select Cluster", ["Bullet Cluster", "Abell 520", "Abell 2744", "Abell 1689", "Custom"])
    
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
    else:
        enhancement_method = "clahe"
    
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
    st.markdown("Query MAST (HST/JWST), NED, and other archives.")
    
    # Quick select buttons
    st.markdown("**Quick Select:**")
    quick_cols = st.columns(4)
    quick_targets = ["Bullet Cluster", "Abell 2744", "Abell 520", "M87"]
    
    for idx, target in enumerate(quick_targets):
        with quick_cols[idx]:
            if st.button(f"🔭 {target}", key=f"quick_{target}"):
                st.session_state.ai_query = f"Get images of {target}"
                st.rerun()
    
    ai_query = st.text_input("Describe what data you want:", 
                              value=st.session_state.get('ai_query', ''),
                              placeholder="e.g., 'Get JWST NIRCam images of Abell 2744'")
    
    if st.button("🔍 Fetch Data", type="primary"):
        if ai_query:
            with st.spinner("🔭 Querying astronomical databases..."):
                query_lower = ai_query.lower()
                
                target_list = ["Bullet Cluster", "1E 0657-56", "Abell 2744", "Abell 520", 
                               "Abell 1689", "Abell 209", "M87", "Virgo Cluster", "Coma Cluster"]
                
                target = None
                for t in target_list:
                    if t.lower() in query_lower:
                        target = t
                        break
                
                if target is None:
                    patterns = [r'of\s+([A-Z][a-z0-9\s]+)', r'images?\s+of\s+([A-Z][a-z0-9\s]+)']
                    for pattern in patterns:
                        match = re.search(pattern, ai_query)
                        if match:
                            target = match.group(1).strip()
                            break
                
                observatory = None
                if any(x in query_lower for x in ['jwst', 'webb']):
                    observatory = "JWST"
                elif any(x in query_lower for x in ['hst', 'hubble']):
                    observatory = "HST"
                
                st.info(f"Searching: **{target or 'any'}** | Observatory: **{observatory or 'any'}**")
                
                if target:
                    try:
                        from astroquery.mast import Observations
                        
                        criteria = {'target_name': target}
                        if observatory:
                            criteria['obs_collection'] = observatory
                        
                        obs_table = Observations.query_criteria(**criteria)
                        
                        if obs_table is not None and len(obs_table) > 0:
                            st.success(f"✅ Found **{len(obs_table)}** observations!")
                            
                            display_df = obs_table[['obs_id', 'obs_collection', 'target_name', 't_exptime', 'instrument_name']].to_pandas()
                            st.dataframe(display_df.head(10), use_container_width=True)
                            
                            st.markdown(f"[🔗 Open in MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={{\\\"filters\\\":[{{\\\"paramName\\\":\\\"target_name\\\",\\\"values\\\":[\\\"{target}\\\"]}}]}})")
                        else:
                            st.warning(f"No data found for {target}")
                    except Exception as e:
                        st.error(f"MAST query error: {e}")
                        st.markdown(f"[🔗 Search MAST manually for {target}](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={{\\\"filters\\\":[{{\\\"paramName\\\":\\\"target_name\\\",\\\"values\\\":[\\\"{target}\\\"]}}]}})")
                else:
                    st.info("Please specify a target name (e.g., 'Bullet Cluster', 'Abell 2744')")
        else:
            st.warning("Please enter a query or use Quick Select")

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
        
        engine = PhotonDarkPhotonModel()
        
        cluster_params = {
            "Bullet Cluster": {"redshift": 0.296, "distance_mpc": 430, "velocity": 200000},
            "Abell 520": {"redshift": 0.201, "distance_mpc": 390, "velocity": 180000},
            "Abell 2744": {"redshift": 0.308, "distance_mpc": 450, "velocity": 220000},
            "Abell 1689": {"redshift": 0.183, "distance_mpc": 380, "velocity": 190000},
            "Custom": {"redshift": 0.2, "distance_mpc": 400, "velocity": 200000}
        }
        
        params = cluster_params.get(cluster, cluster_params["Bullet Cluster"])
        
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
        
        original_norm = engine.original_norm
        conversion_map = engine.get_conversion_map()
        entanglement_map = engine.get_entanglement_map()
        interference_pattern = engine.get_interference_pattern()
        
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
        
        # Interference Pattern
        st.markdown("---")
        st.markdown("## 🌊 Two-Field FDM Interference Pattern")
        st.markdown("From ψ_total = ψ_light + ψ_dark e^(iΔφ)")
        
        fig_int, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(interference_pattern, cmap='viridis', origin='lower')
        ax.set_title("FDM Two-Field Interference | λ = h/(mΔv)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="Interference Intensity")
        st.pyplot(fig_int)
        plt.close(fig_int)
        
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
        
        # Solitonic Core Profile
        with st.expander("🌀 FDM Solitonic Core Profile"):
            st.markdown("**Solitonic core solution to the cusp-core problem:**")
            st.latex(r"\rho(r) = \frac{\rho_c}{[1 + (r/r_c)^2]^8}")
            
            col1, col2 = st.columns(2)
            with col1:
                r_core_kpc = st.slider("Core radius (kpc)", 0.1, 5.0, 1.0, 0.1)
                rho_core = st.slider("Peak density (M☉/kpc³)", 1e6, 1e8, 1.9e7, format="%.1e")
            
            r = np.linspace(0, 10, 500)
            r_norm = r / r_core_kpc
            rho = rho_core / (1 + r_norm**2)**8
            
            fig_sol, ax = plt.subplots(figsize=(8, 5))
            ax.plot(r, rho, 'b-', linewidth=2)
            ax.fill_between(r, rho, alpha=0.3, color='#1e3c72')
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
        ### 🌟 QCI AstroEntangle Refiner v4.5 - Light Theme Edition
        
        **New in v4.5: Full FDM Two-Field Implementation**
        
        This version implements the complete Fuzzy Dark Matter (FDM) framework from the Cosmic Entanglement Visualizer.
        
        **Implemented Equations:**
        - **Klein-Gordon**: Relativistic wave equation for scalar field dark matter
        - **Schrödinger-Poisson**: Non-relativistic limit for galactic-scale structure
        - **Two-field interference**: ψ_total = ψ_light + ψ_dark e^(iΔφ)
        - **Fringe spacing**: λ = h/(mΔv) - de Broglie wavelength of FDM
        - **Solitonic core**: ρ(r) = ρ_c / [1 + (r/r_c)²]⁸ - solves cusp-core problem
        
        **Features:**
        - 🤖 AI Data Download - Query MAST/NED databases
        - 🌊 Two-field Interference Pattern Visualization
        - 🌀 Solitonic Core Profile Calculator
        - 🎨 Light Theme Interface - Easier on the eyes
        - 📥 Download all results as PNG
        
        **Try these parameters:**
        - m_dark = 10⁻²² eV (FDM candidate mass)
        - ε = 10⁻⁸ (typical mixing parameter)
        - Photon energy = 1000 eV (X-ray observations)
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
QCI AstroEntangle Refiner v4.5 | Light Theme | FDM Two-Field Physics | Based on Cosmic Entanglement Visualizer Framework
</div>
""", unsafe_allow_html=True)
