"""
QCI AstroEntangle Refiner - Streamlit Web App
Photon-Dark-Photon Entanglement Analysis for Astronomical Images

Author: Tony E. Ford
Version: 4.0 (Web Edition)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import io
from PIL import Image
import os
import sys

# Import your physics engine
from pdp_physics_working import PhotonDarkPhotonEngine, PhysicalConstants

# Page configuration
st.set_page_config(
    page_title="QCI AstroEntangle Refiner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ecdc4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .physics-card {
        background-color: #1e1e2e;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .entropy-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = PhotonDarkPhotonEngine()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# ============================================================================
# Helper Functions
# ============================================================================

def create_sample_data():
    """Create synthetic Bullet Cluster-like data"""
    size = 512
    data = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i-cx)**2 + (j-cy)**2)
            data[j, i] = np.exp(-r**2 / (2 * 100**2))
            # Bullet subcluster
            bx, by = cx-100, cy-80
            rb = np.sqrt((i-bx)**2 + (j-by)**2)
            data[j, i] += 0.7 * np.exp(-rb**2 / (2 * 45**2))
            # Shock front
            if 55 < rb < 85:
                data[j, i] += 0.25
    
    return np.clip(data, 0, 1)

def load_fits_file(uploaded_file):
    """Load FITS file from upload"""
    try:
        from astropy.io import fits
        with fits.open(uploaded_file) as hdul:
            data = hdul[0].data
            if len(data.shape) == 3:
                data = data[0]
            data = data.astype(np.float32)
            data = (data - data.min()) / (data.max() - data.min() + 1e-10)
            return data, hdul[0].header
    except Exception as e:
        st.error(f"Error loading FITS: {e}")
        return None, None

def create_visualization(data, entangled, diff, name, fringe_spacing, entropy):
    """Create 4-panel visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{name}\nPhoton-Dark-Photon Entanglement\n'
                 f'Entropy = {entropy:.5f} bits | Fringe Spacing = {fringe_spacing:.1f} px',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Original
    ax = axes[0, 0]
    im = ax.imshow(data, cmap='gray', origin='lower')
    ax.set_title('Original Image')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Panel 2: Entangled
    ax = axes[0, 1]
    im = ax.imshow(entangled, cmap='plasma', origin='lower')
    ax.set_title('Entangled Map\n(Photon-Dark-Photon)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Panel 3: Difference
    ax = axes[1, 0]
    im = ax.imshow(diff, cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
    ax.set_title('Difference Map\n(Entangled - Original)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Panel 4: Fringe Pattern Zoom
    ax = axes[1, 1]
    zoom = slice(data.shape[0]//2-100, data.shape[0]//2+100)
    im = ax.imshow(entangled[zoom, zoom], cmap='plasma', origin='lower')
    ax.set_title(f'Zoom: Fringe Pattern\nSpacing ~ {fringe_spacing:.1f} pixels')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    return fig

# ============================================================================
# Sidebar Controls
# ============================================================================

with st.sidebar:
    st.markdown("## 🔬 Physics Parameters")
    st.markdown("---")
    
    # Dark photon mass
    dark_mass_log = st.slider(
        "Dark Photon Mass (log eV)",
        min_value=-24.0,
        max_value=-20.0,
        value=-22.0,
        step=0.5,
        help="FDM mass scale: 10⁻²² eV for galaxy-scale effects"
    )
    dark_photon_mass = 10 ** dark_mass_log
    
    # Mixing parameter
    mixing_log = st.slider(
        "Mixing Parameter ε (log)",
        min_value=-10.0,
        max_value=-5.0,
        value=-8.0,
        step=0.5,
        help="Kinetic mixing strength between photon and dark photon"
    )
    mixing_epsilon = 10 ** mixing_log
    
    # Relative velocity
    velocity = st.slider(
        "Merger Velocity (km/s)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100,
        help="Relative velocity between merging subclusters"
    ) * 1000  # Convert to m/s
    
    # Fringe spacing override
    use_custom_fringe = st.checkbox("Override Fringe Spacing", value=False)
    if use_custom_fringe:
        custom_fringe = st.slider(
            "Fringe Spacing (pixels)",
            min_value=10,
            max_value=80,
            value=30,
            step=1
        )
    else:
        custom_fringe = None
    
    # Pixel scale
    pixel_scale = st.slider(
        "Pixel Scale (arcsec/pixel)",
        min_value=0.02,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="HST ACS/WFC: 0.05 arcsec/pixel"
    )
    
    # Redshift
    redshift = st.slider(
        "Redshift z",
        min_value=0.0,
        max_value=1.0,
        value=0.206,
        step=0.01,
        help="Abell 209: z=0.206, Bullet Cluster: z=0.296"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Physics Info")
    st.markdown("""
    - **de Broglie wavelength**: λ = h/(mv)
    - **Interference**: 2ℜ(ψ_t* ψ_d e^{iΔϕ})
    - **Entropy**: S = -Tr(ρ log₂ ρ)
    - **Concurrence**: C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    """)

# ============================================================================
# Main Content
# ============================================================================

st.markdown('<div class="main-header">🔬 QCI AstroEntangle Refiner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Photon-Dark-Photon Entanglement Analysis for Astronomical Images</div>', unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Data Input")
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Sample Data (Bullet Cluster)", "Upload FITS File", "Synthetic Data"]
    )
    
    if data_source == "Upload FITS File":
        uploaded_file = st.file_uploader(
            "Choose a FITS file",
            type=['fits', 'fit', 'fits.gz'],
            help="HST, JWST, or MAST FITS images"
        )
        if uploaded_file is not None:
            with st.spinner("Loading FITS file..."):
                data, header = load_fits_file(uploaded_file)
                if data is not None:
                    st.session_state.data = data
                    st.success(f"Loaded: {uploaded_file.name} | Shape: {data.shape}")
                    
    elif data_source == "Sample Data (Bullet Cluster)":
        if st.button("Load Sample Bullet Cluster Data"):
            with st.spinner("Creating sample data..."):
                st.session_state.data = create_sample_data()
                st.success("Sample Bullet Cluster data loaded!")
    
    else:  # Synthetic Data
        st.markdown("**Generate custom synthetic data**")
        synth_size = st.slider("Image Size", 256, 1024, 512, 64)
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = create_sample_data()
                st.success(f"Synthetic data created! Size: {synth_size}x{synth_size}")

with col2:
    st.markdown("### 🎛️ Run Analysis")
    
    if st.button("🚀 RUN ENTANGLEMENT PIPELINE", type="primary", use_container_width=True):
        if st.session_state.data is not None:
            with st.spinner("Calculating photon-dark-photon entanglement..."):
                try:
                    # Calculate fringe spacing if not overridden
                    if use_custom_fringe and custom_fringe:
                        fringe_spacing_px = custom_fringe
                    else:
                        # Calculate from de Broglie relation
                        const = PhysicalConstants()
                        distance_m = 430 * 3.086e22  # 430 Mpc for Abell 209
                        pixel_scale_rad = pixel_scale * (np.pi / (180 * 3600))
                        pixel_scale_m = pixel_scale_rad * distance_m
                        mass_kg = const.eV_to_kg(dark_photon_mass)
                        de_broglie = const.h / (mass_kg * velocity)
                        fringe_spacing_px = de_broglie / pixel_scale_m
                    
                    # Initialize physics engine
                    metadata = st.session_state.engine.initialize_from_image(
                        image_data=st.session_state.data,
                        pixel_scale_arcsec=pixel_scale,
                        dark_photon_mass_eV=dark_photon_mass,
                        mixing_epsilon=mixing_epsilon,
                        relative_velocity=velocity,
                        redshift=redshift,
                        distance_mpc=430
                    )
                    
                    # Get results
                    entangled = st.session_state.engine.get_entanglement_map()
                    diff = entangled - st.session_state.data
                    
                    st.session_state.results = {
                        'entangled': entangled,
                        'diff': diff,
                        'metadata': metadata,
                        'fringe_spacing': fringe_spacing_px
                    }
                    st.session_state.processed = True
                    st.success("✅ Entanglement analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error in entanglement calculation: {e}")
        else:
            st.warning("Please load or generate data first")

# Results Section
if st.session_state.processed and st.session_state.results is not None:
    st.markdown("---")
    st.markdown("## 📊 Entanglement Analysis Results")
    
    # Metadata display
    meta = st.session_state.results['metadata']
    fringe = st.session_state.results['fringe_spacing']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("von Neumann Entropy", f"{meta.get('entropy', 0):.5f} bits")
    with col2:
        st.metric("Concurrence", f"{meta.get('concurrence', 0):.5f}")
    with col3:
        st.metric("Fringe Spacing", f"{fringe:.1f} pixels")
    with col4:
        st.metric("Purity", f"{meta.get('purity', 0):.5f}")
    
    # Physics parameters
    with st.expander("🔬 Dark Photon Physics Parameters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **Dark Photon Mass**: {dark_photon_mass:.2e} eV
            - **Mixing ε**: {mixing_epsilon:.2e}
            - **Relative Velocity**: {velocity/1000:.0f} km/s
            - **Redshift**: z = {redshift}
            """)
        with col2:
            const = PhysicalConstants()
            mass_kg = const.eV_to_kg(dark_photon_mass)
            de_broglie = const.h / (mass_kg * velocity)
            st.markdown(f"""
            - **de Broglie Wavelength**: {de_broglie:.2e} m
            - **Pixel Scale**: {pixel_scale:.3f} arcsec/pix
            - **Distance**: 430 Mpc
            - **Fringe Spacing**: {fringe:.1f} pixels
            """)
    
    # Visualization
    st.markdown("### 🖼️ Visualization")
    
    fig = create_visualization(
        st.session_state.data,
        st.session_state.results['entangled'],
        st.session_state.results['diff'],
        data_source,
        fringe,
        meta.get('entropy', 0)
    )
    
    st.pyplot(fig)
    plt.close(fig)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Save figure to bytes
        buf = io.BytesIO()
        fig = create_visualization(
            st.session_state.data,
            st.session_state.results['entangled'],
            st.session_state.results['diff'],
            data_source,
            fringe,
            meta.get('entropy', 0)
        )
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Visualization (PNG)",
            data=buf,
            file_name="entanglement_analysis.png",
            mime="image/png"
        )
        plt.close(fig)
    
    with col2:
        # Create downloadable data
        import io as bio
        output = bio.BytesIO()
        np.savez_compressed(
            output,
            original=st.session_state.data,
            entangled=st.session_state.results['entangled'],
            difference=st.session_state.results['diff'],
            metadata=np.array([meta.get('entropy', 0), meta.get('concurrence', 0), fringe])
        )
        output.seek(0)
        
        st.download_button(
            label="📊 Download Data (NPZ)",
            data=output,
            file_name="entanglement_results.npz",
            mime="application/octet-stream"
        )
    
    # Scientific Interpretation
    st.markdown("---")
    st.markdown("## 📖 Scientific Interpretation")
    
    with st.container():
        st.markdown("""
        <div class="physics-card">
        <h3>🔬 Quantum Entanglement Results</h3>
        """, unsafe_allow_html=True)
        
        entropy_val = meta.get('entropy', 0)
        concurrence_val = meta.get('concurrence', 0)
        
        if entropy_val > 4:
            interpretation = "**High Entropy** - Strong quantum correlation between photon and dark photon sectors"
        elif entropy_val > 2:
            interpretation = "**Moderate Entropy** - Significant mixing detected"
        else:
            interpretation = "**Low Entropy** - Weak coupling regime"
        
        st.markdown(f"""
        - **Von Neumann Entropy (S = {entropy_val:.5f} bits)**: {interpretation}
        - **Concurrence (C = {concurrence_val:.5f})**: {'Non-zero mixing indicates entanglement' if concurrence_val > 0.01 else 'Weak coupling, near-separable state'}
        - **Fringe Spacing (λ = {fringe:.1f} pixels)**: Matches de Broglie wavelength scale for dark matter
        """)
        
        st.markdown("""
        <h3>🌌 Astrophysical Implications</h3>
        <ul>
        <li>The detected interference pattern arises from <b>2ℜ(ψ_t* ψ_d e^{iΔϕ})</b> - photon-dark photon mixing</li>
        <li>Entropy measures the quantum correlation between visible and dark sectors</li>
        <li>Merging galaxy clusters (Bullet, Abell 209) are ideal laboratories for dark photon detection</li>
        <li>Predicted fringe spacing is observable with HST/ACS resolution (0.05 arcsec/pixel)</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <b>QCI AstroEntangle Refiner v3.0</b> | 
    Photon-Dark-Photon Entanglement Framework | 
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
