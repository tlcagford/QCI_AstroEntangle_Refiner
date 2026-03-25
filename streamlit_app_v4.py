"""
QCI AstroEntangle Refiner v4.0
Photon-Dark-Photon Entanglement Analysis
Enhanced with 3D visualization, batch processing, and interactive parameter exploration

Author: Tony E. Ford
Version: 4.0
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import io
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys
import base64
from pathlib import Path

# Import physics engine
from pdp_physics_working import PhotonDarkPhotonEngine, PhysicalConstants

# Page configuration
st.set_page_config(
    page_title="QCI AstroEntangle Refiner v4.0",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for v4.0
st.markdown("""
<style>
    /* Dark theme with neon accents */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .version-badge {
        text-align: center;
        color: #4ecdc4;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .physics-card {
        background: rgba(30, 30, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(78, 205, 196, 0.3);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: rgba(255, 107, 107, 0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 3px solid #ff6b6b;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6b6b;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #888;
    }
    
    .formula {
        font-family: monospace;
        background: #1e1e2e;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #4ecdc4;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = PhotonDarkPhotonEngine()
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# ============================================================================
# Header
# ============================================================================

st.markdown('<div class="main-header">🔬 QCI AstroEntangle Refiner</div>', unsafe_allow_html=True)
st.markdown('<div class="version-badge">Version 4.0 | Photon-Dark-Photon Entanglement Framework</div>', unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Mode selection
    mode = st.radio(
        "Mode",
        ["🔬 Single Analysis", "📊 Batch Processing", "📈 Parameter Explorer", "🎓 Tutorial"],
        help="Select analysis mode"
    )
    
    st.markdown("---")
    
    # Physics Parameters
    st.markdown("### ⚛️ Dark Photon Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        dark_mass_log = st.slider(
            "Mass (log eV)",
            min_value=-24.0,
            max_value=-20.0,
            value=-22.0,
            step=0.1,
            help="10⁻²² eV = FDM scale"
        )
    with col2:
        mixing_log = st.slider(
            "Mixing ε (log)",
            min_value=-12.0,
            max_value=-5.0,
            value=-8.0,
            step=0.2,
            help="Kinetic mixing strength"
        )
    
    velocity = st.slider(
        "Merger Velocity (km/s)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100,
        help="Relative velocity of merging subclusters"
    ) * 1000
    
    st.markdown("### 🌌 Observational Parameters")
    
    pixel_scale = st.slider(
        "Pixel Scale (arcsec/pixel)",
        min_value=0.02,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="HST ACS/WFC: 0.05 arcsec/pixel"
    )
    
    redshift = st.slider(
        "Redshift z",
        min_value=0.0,
        max_value=1.0,
        value=0.206,
        step=0.01,
        help="Abell 209: 0.206, Bullet Cluster: 0.296"
    )
    
    st.markdown("---")
    
    # Display derived parameters
    const = PhysicalConstants()
    dark_photon_mass = 10 ** dark_mass_log
    mass_kg = const.eV_to_kg(dark_photon_mass)
    de_broglie = const.h / (mass_kg * velocity)
    
    st.markdown("### 📐 Derived Parameters")
    st.info(f"""
    **de Broglie λ**: {de_broglie:.2e} m  
    **Mass**: {dark_photon_mass:.2e} eV  
    **Mixing ε**: {10**mixing_log:.2e}  
    **Velocity**: {velocity/1000:.0f} km/s
    """)

# ============================================================================
# Mode 1: Single Analysis
# ============================================================================

if mode == "🔬 Single Analysis":
    st.markdown("## 🔬 Entanglement Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio(
            "Data Source",
            ["🌌 Sample: Bullet Cluster", "🌠 Sample: Abell 209", "📁 Upload FITS", "🎨 Generate Synthetic"],
            horizontal=True
        )
    
    with col2:
        st.markdown("### Quick Actions")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Load data based on selection
    if data_source == "🌌 Sample: Bullet Cluster":
        from scipy.ndimage import gaussian_filter
        size = 512
        data = np.zeros((size, size))
        cx, cy = size//2, size//2
        
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i-cx)**2 + (j-cy)**2)
                data[j, i] = np.exp(-r**2 / (2 * 100**2))
                bx, by = cx-100, cy-80
                rb = np.sqrt((i-bx)**2 + (j-by)**2)
                data[j, i] += 0.7 * np.exp(-rb**2 / (2 * 45**2))
                if 55 < rb < 85:
                    data[j, i] += 0.25
        
        data = np.clip(data, 0, 1)
        st.session_state.data = data
        st.success("✅ Bullet Cluster data loaded")
        
    elif data_source == "🌠 Sample: Abell 209":
        size = 512
        data = np.zeros((size, size))
        cx, cy = size//2, size//2
        
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i-cx)**2 + (j-cy)**2)
                data[j, i] = np.exp(-r**2 / (2 * 90**2))
        
        subs = [(-90, -70, 0.65), (-45, -35, 0.5), (80, 45, 0.45), (25, -85, 0.4)]
        for sx, sy, amp in subs:
            for i in range(size):
                for j in range(size):
                    r = np.sqrt((i-(cx+sx))**2 + (j-(cy+sy))**2)
                    data[j, i] += amp * np.exp(-r**2 / (2 * 40**2))
        
        data = np.clip(data, 0, 1)
        st.session_state.data = data
        st.success("✅ Abell 209 data loaded")
        
    elif data_source == "📁 Upload FITS":
        uploaded_file = st.file_uploader("Choose FITS file", type=['fits', 'fit'])
        if uploaded_file is not None:
            try:
                from astropy.io import fits
                with fits.open(uploaded_file) as hdul:
                    data = hdul[0].data
                    if len(data.shape) == 3:
                        data = data[0]
                    data = data.astype(np.float32)
                    data = (data - data.min()) / (data.max() - data.min() + 1e-10)
                    st.session_state.data = data
                    st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading FITS: {e}")
    
    else:  # Generate Synthetic
        st.markdown("**Synthetic Galaxy Cluster**")
        size = st.slider("Image Size", 256, 1024, 512, 128)
        num_subclusters = st.slider("Number of Subclusters", 1, 8, 4)
        
        if st.button("Generate"):
            data = np.zeros((size, size))
            cx, cy = size//2, size//2
            
            # Central galaxy
            for i in range(size):
                for j in range(size):
                    r = np.sqrt((i-cx)**2 + (j-cy)**2)
                    data[j, i] = np.exp(-r**2 / (2 * 90**2))
            
            # Random subclusters
            np.random.seed(42)
            for _ in range(num_subclusters):
                sx = np.random.randint(-150, 150)
                sy = np.random.randint(-150, 150)
                amp = np.random.uniform(0.3, 0.8)
                sigma = np.random.randint(30, 60)
                
                for i in range(size):
                    for j in range(size):
                        r = np.sqrt((i-(cx+sx))**2 + (j-(cy+sy))**2)
                        data[j, i] += amp * np.exp(-r**2 / (2 * sigma**2))
            
            data = np.clip(data, 0, 1)
            st.session_state.data = data
            st.success("✅ Synthetic cluster generated")
    
    # Display data preview
    if 'data' in st.session_state and st.session_state.data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Image**")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(st.session_state.data, cmap='gray', origin='lower')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
    
    # Run analysis
    if run_analysis and 'data' in st.session_state and st.session_state.data is not None:
        with st.spinner("🔄 Computing photon-dark-photon entanglement..."):
            # Calculate fringe spacing
            distance_m = 430 * 3.086e22
            pixel_scale_rad = pixel_scale * (np.pi / (180 * 3600))
            pixel_scale_m = pixel_scale_rad * distance_m
            mass_kg = const.eV_to_kg(10**dark_mass_log)
            de_broglie_calc = const.h / (mass_kg * velocity)
            fringe_spacing_px = de_broglie_calc / pixel_scale_m
            
            # Run physics engine
            metadata = st.session_state.engine.initialize_from_image(
                image_data=st.session_state.data,
                pixel_scale_arcsec=pixel_scale,
                dark_photon_mass_eV=10**dark_mass_log,
                mixing_epsilon=10**mixing_log,
                relative_velocity=velocity,
                redshift=redshift,
                distance_mpc=430
            )
            
            entangled = st.session_state.engine.get_entanglement_map()
            diff = entangled - st.session_state.data
            
            # Store results
            st.session_state.current_results = {
                'entangled': entangled,
                'diff': diff,
                'metadata': metadata,
                'fringe_spacing': fringe_spacing_px
            }
            
            # Add to history
            st.session_state.results_history.append({
                'timestamp': datetime.now(),
                'entropy': metadata.get('entropy', 0),
                'concurrence': metadata.get('concurrence', 0),
                'fringe': fringe_spacing_px,
                'mass': 10**dark_mass_log,
                'mixing': 10**mixing_log
            })
    
    # Display results
    if 'current_results' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 Results")
        
        meta = st.session_state.current_results['metadata']
        fringe = st.session_state.current_results['fringe_spacing']
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.5f}</div><div class="metric-label">Entropy (bits)</div></div>'.format(meta.get('entropy', 0)), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.5e}</div><div class="metric-label">Concurrence</div></div>'.format(meta.get('concurrence', 0)), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.3f}</div><div class="metric-label">Purity</div></div>'.format(meta.get('purity', 0)), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.1f}</div><div class="metric-label">Fringe (pixels)</div></div>'.format(fringe), unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="metric-card"><div class="metric-value">{:.2e}</div><div class="metric-label">Dark Mass (eV)</div></div>'.format(10**dark_mass_log), unsafe_allow_html=True)
        
        # Visualization
        st.markdown("### 🖼️ Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Photon-Dark-Photon Entanglement Analysis\n'
                     f'Entropy = {meta.get("entropy", 0):.5f} bits | Concurrence = {meta.get("concurrence", 0):.5e}\n'
                     f'Fringe Spacing = {fringe:.1f} pixels | Dark Photon Mass = {10**dark_mass_log:.2e} eV',
                     fontsize=14, fontweight='bold')
        
        # Original
        ax = axes[0, 0]
        im = ax.imshow(st.session_state.data, cmap='gray', origin='lower')
        ax.set_title('Original Image')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Entangled
        ax = axes[0, 1]
        im = ax.imshow(st.session_state.current_results['entangled'], cmap='plasma', origin='lower')
        ax.set_title('Entangled Map')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Difference
        ax = axes[1, 0]
        im = ax.imshow(st.session_state.current_results['diff'], cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
        ax.set_title('Difference Map (Entangled - Original)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Fringe Zoom
        ax = axes[1, 1]
        zoom = slice(st.session_state.data.shape[0]//2-100, st.session_state.data.shape[0]//2+100)
        im = ax.imshow(st.session_state.current_results['entangled'][zoom, zoom], cmap='plasma', origin='lower')
        ax.set_title(f'Zoom: Fringe Pattern (spacing ~{fringe:.1f} px)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 Download Visualization (PNG)", buf, file_name="entanglement_analysis.png", mime="image/png")
        
        with col2:
            output = io.BytesIO()
            np.savez_compressed(output, original=st.session_state.data, entangled=st.session_state.current_results['entangled'], difference=st.session_state.current_results['diff'])
            output.seek(0)
            st.download_button("📊 Download Data (NPZ)", output, file_name="entanglement_results.npz")

# ============================================================================
# Mode 2: Parameter Explorer
# ============================================================================

elif mode == "📈 Parameter Explorer":
    st.markdown("## 📈 Parameter Space Explorer")
    st.markdown("Explore how entanglement entropy varies with dark photon parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mass_range = st.slider(
            "Dark Photon Mass Range (log eV)",
            min_value=-24.0,
            max_value=-20.0,
            value=[-23.0, -21.0],
            step=0.5
        )
    
    with col2:
        mixing_range = st.slider(
            "Mixing ε Range (log)",
            min_value=-12.0,
            max_value=-5.0,
            value=[-10.0, -7.0],
            step=0.5
        )
    
    # Generate parameter grid
    mass_grid = np.linspace(mass_range[0], mass_range[1], 20)
    mixing_grid = np.linspace(mixing_range[0], mixing_range[1], 20)
    
    if st.button("🔬 Explore Parameter Space", type="primary"):
        with st.spinner("Exploring parameter space..."):
            # Create sample data
            size = 256
            data = np.zeros((size, size))
            cx, cy = size//2, size//2
            
            for i in range(size):
                for j in range(size):
                    r = np.sqrt((i-cx)**2 + (j-cy)**2)
                    data[j, i] = np.exp(-r**2 / (2 * 90**2))
            
            subs = [(-90, -70, 0.65), (-45, -35, 0.5), (80, 45, 0.45), (25, -85, 0.4)]
            for sx, sy, amp in subs:
                for i in range(size):
                    for j in range(size):
                        r = np.sqrt((i-(cx+sx))**2 + (j-(cy+sy))**2)
                        data[j, i] += amp * np.exp(-r**2 / (2 * 40**2))
            
            data = np.clip(data, 0, 1)
            
            # Grid search
            entropy_grid = np.zeros((len(mass_grid), len(mixing_grid)))
            
            for i, mass_log in enumerate(mass_grid):
                for j, mix_log in enumerate(mixing_grid):
                    engine = PhotonDarkPhotonEngine()
                    metadata = engine.initialize_from_image(
                        image_data=data,
                        pixel_scale_arcsec=0.05,
                        dark_photon_mass_eV=10**mass_log,
                        mixing_epsilon=10**mix_log,
                        relative_velocity=2000000,
                        redshift=0.206
                    )
                    entropy_grid[i, j] = metadata.get('entropy', 0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=entropy_grid.T,
                x=mass_grid,
                y=mixing_grid,
                colorscale='Plasma',
                colorbar=dict(title="Entropy (bits)")
            ))
            
            fig.update_layout(
                title="Parameter Space: Entropy vs Dark Photon Mass and Mixing",
                xaxis_title="Dark Photon Mass (log eV)",
                yaxis_title="Mixing ε (log)",
                width=800,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find optimum
            max_idx = np.unravel_index(np.argmax(entropy_grid), entropy_grid.shape)
            st.success(f"Maximum Entropy: {entropy_grid[max_idx]:.5f} bits at mass = 10^{mass_grid[max_idx[0]]:.2f} eV, ε = 10^{mixing_grid[max_idx[1]]:.2f}")

# ============================================================================
# Mode 3: Batch Processing
# ============================================================================

elif mode == "📊 Batch Processing":
    st.markdown("## 📊 Batch Processing")
    st.markdown("Process multiple FITS files or generate multiple synthetic clusters")
    
    batch_type = st.radio("Batch Type", ["Multiple FITS Files", "Parameter Variations", "Noise Realizations"])
    
    if batch_type == "Parameter Variations":
        num_variations = st.slider("Number of Parameter Variations", 5, 50, 10)
        
        if st.button("🚀 Run Batch", type="primary"):
            with st.progress(0):
                results = []
                for idx in range(num_variations):
                    # Vary parameters
                    mass_log = np.random.uniform(-24, -20)
                    mix_log = np.random.uniform(-12, -5)
                    velocity = np.random.uniform(500, 5000) * 1000
                    
                    # Generate data
                    size = 256
                    data = np.zeros((size, size))
                    cx, cy = size//2, size//2
                    
                    for i in range(size):
                        for j in range(size):
                            r = np.sqrt((i-cx)**2 + (j-cy)**2)
                            data[j, i] = np.exp(-r**2 / (2 * 90**2))
                    
                    # Run physics
                    engine = PhotonDarkPhotonEngine()
                    metadata = engine.initialize_from_image(
                        image_data=data,
                        pixel_scale_arcsec=0.05,
                        dark_photon_mass_eV=10**mass_log,
                        mixing_epsilon=10**mix_log,
                        relative_velocity=velocity,
                        redshift=0.206
                    )
                    
                    results.append({
                        'mass_eV': 10**mass_log,
                        'mixing': 10**mix_log,
                        'velocity_km_s': velocity/1000,
                        'entropy': metadata.get('entropy', 0),
                        'concurrence': metadata.get('concurrence', 0)
                    })
                    
                    st.progress((idx + 1) / num_variations)
            
            # Display results
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            # Visualize
            fig = px.scatter(df, x='mass_eV', y='entropy', color='velocity_km_s',
                             title='Entropy vs Dark Photon Mass',
                             log_x=True, labels={'mass_eV': 'Dark Photon Mass (eV)', 'entropy': 'Entropy (bits)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results (CSV)", csv, file_name="batch_results.csv")

# ============================================================================
# Mode 4: Tutorial
# ============================================================================

elif mode == "🎓 Tutorial":
    st.markdown("## 🎓 Tutorial: Photon-Dark-Photon Entanglement")
    
    tabs = st.tabs(["📖 Theory", "🔬 Physics", "📊 Examples", "🎯 Interpretation"])
    
    with tabs[0]:
        st.markdown("""
        ### The Photon-Dark-Photon Entanglement Hypothesis
        
        Dark matter may consist of ultralight **dark photons** that mix with visible photons via kinetic mixing.
        
        #### Key Idea
        In merging galaxy clusters (like the Bullet Cluster), the relative motion between visible and dark matter creates **quantum interference patterns** detectable in astronomical images.
        
        #### Core Equations
        
        <div class="formula">
        Two-field coupled system:
