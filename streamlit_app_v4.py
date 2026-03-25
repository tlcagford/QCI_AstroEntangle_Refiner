"""
QCI AstroEntangle Refiner v4.0
Photon-Dark-Photon Entanglement Analysis for Astronomical Images
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Import physics engine
from pdp_physics_working import PhotonDarkPhotonEngine, PhysicalConstants
st.set_page_config(
    page_title="QCI AstroEntangle Refiner v4.0",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 QCI AstroEntangle Refiner v4.0</div>', unsafe_allow_html=True)
st.markdown("### Photon-Dark-Photon Entanglement Analysis")

# Sidebar controls
with st.sidebar:
    st.header("⚛️ Dark Photon Parameters")
    
    dark_mass_log = st.slider(
        "Dark Photon Mass (log eV)",
        min_value=-24.0, max_value=-20.0, value=-22.0, step=0.5,
        help="FDM mass scale: 10⁻²² eV for galaxy-scale effects"
    )
    
    mixing_log = st.slider(
        "Mixing Parameter ε (log)",
        min_value=-12.0, max_value=-5.0, value=-8.0, step=0.5,
        help="Kinetic mixing strength between photon and dark photon"
    )
    
    velocity = st.slider(
        "Merger Velocity (km/s)",
        min_value=500, max_value=5000, value=2000, step=100
    ) * 1000
    
    pixel_scale = st.slider(
        "Pixel Scale (arcsec/pixel)",
        min_value=0.02, max_value=0.2, value=0.05, step=0.01
    )
    
    redshift = st.slider(
        "Redshift z",
        min_value=0.0, max_value=1.0, value=0.206, step=0.01
    )
    
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📁 Data Input")
    
    data_source = st.radio(
        "Select data source:",
        ["🌌 Sample: Bullet Cluster", "🌠 Sample: Abell 209", "🎨 Generate Synthetic"]
    )

# Create sample data function
def create_sample_data(cluster_type, size=512):
    data = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    if cluster_type == "bullet":
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i-cx)**2 + (j-cy)**2)
                data[j, i] = np.exp(-r**2 / (2 * 100**2))
                bx, by = cx-100, cy-80
                rb = np.sqrt((i-bx)**2 + (j-by)**2)
                data[j, i] += 0.7 * np.exp(-rb**2 / (2 * 45**2))
                if 55 < rb < 85:
                    data[j, i] += 0.25
    else:
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
    
    return np.clip(data, 0, 1)

# Generate data based on selection
size = 512
if data_source == "🌌 Sample: Bullet Cluster":
    data = create_sample_data("bullet", size)
    st.success("✅ Bullet Cluster data loaded")
elif data_source == "🌠 Sample: Abell 209":
    data = create_sample_data("abell", size)
    st.success("✅ Abell 209 data loaded")
else:
    size = st.slider("Image Size", 256, 1024, 512)
    data = create_sample_data("bullet", size)
    st.success(f"✅ Synthetic data created: {size}x{size}")

# Display preview
st.subheader("📷 Image Preview")
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(data, cmap='gray', origin='lower')
ax.axis('off')
st.pyplot(fig)
plt.close(fig)

# Run analysis
if run_button:
    with st.spinner("🔄 Computing photon-dark-photon entanglement..."):
        try:
            const = PhysicalConstants()
            engine = PhotonDarkPhotonEngine()
            
            # Calculate derived parameters
            distance_m = 430 * 3.086e22
            pixel_scale_rad = pixel_scale * (np.pi / (180 * 3600))
            pixel_scale_m = pixel_scale_rad * distance_m
            mass_kg = const.eV_to_kg(10**dark_mass_log)
            de_broglie = const.h / (mass_kg * velocity)
            fringe_spacing_px = de_broglie / pixel_scale_m
            
            # Run physics
            metadata = engine.initialize_from_image(
                image_data=data,
                pixel_scale_arcsec=pixel_scale,
                dark_photon_mass_eV=10**dark_mass_log,
                mixing_epsilon=10**mixing_log,
                relative_velocity=velocity,
                redshift=redshift,
                distance_mpc=430
            )
            
            entangled = engine.get_entanglement_map()
            diff = entangled - data
            
            st.success("✅ Analysis complete!")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("von Neumann Entropy", f"{metadata.get('entropy', 0):.5f} bits")
            with col2:
                st.metric("Concurrence", f"{metadata.get('concurrence', 0):.5e}")
            with col3:
                st.metric("Fringe Spacing", f"{fringe_spacing_px:.1f} pixels")
            with col4:
                st.metric("Dark Photon Mass", f"{10**dark_mass_log:.2e} eV")
            
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Photon-Dark-Photon Entanglement\n'
                        f'Entropy = {metadata.get("entropy", 0):.5f} bits | '
                        f'Concurrence = {metadata.get("concurrence", 0):.5e}',
                        fontsize=14, fontweight='bold')
            
            axes[0,0].imshow(data, cmap='gray', origin='lower')
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(entangled, cmap='plasma', origin='lower')
            axes[0,1].set_title('Entangled Map')
            axes[0,1].axis('off')
            
            im = axes[1,0].imshow(diff, cmap='RdBu', origin='lower', vmin=-0.1, vmax=0.1)
            axes[1,0].set_title('Difference Map')
            axes[1,0].axis('off')
            plt.colorbar(im, ax=axes[1,0], fraction=0.046)
            
            zoom = slice(data.shape[0]//2-100, data.shape[0]//2+100)
            axes[1,1].imshow(entangled[zoom, zoom], cmap='plasma', origin='lower')
            axes[1,1].set_title(f'Fringe Pattern (~{fringe_spacing_px:.0f} px)')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button("📥 Download Results (PNG)", buf, 
                              file_name="entanglement_analysis.png", mime="image/png")
                              
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            st.info("Make sure pdp_physics_working.py is present and has all required functions.")

st.markdown("---")
st.markdown("© 2026 Tony E. Ford | [GitHub](https://github.com/tlcagford/QCI_AstroEntangle_Refiner)")
st.markdown(r"""
### 🔬 Physics Behind the Analysis

The interference pattern is governed by:
\[
P_{\gamma \to A'} = 4\epsilon^2 \sin^2\left(\frac{\Delta m^2 L}{4E}\right)
\]
...
""")
