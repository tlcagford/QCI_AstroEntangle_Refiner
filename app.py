"""
QCAUS - Spectral & Color Heat Pattern Analyzer
Analyzes the spectral characteristics of FDM Soliton and PDP Entanglement patterns
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_power_spectrum(image):
    """Compute 2D power spectrum (Fourier magnitude squared)"""
    fft_img = fft2(image)
    fft_shift = fftshift(fft_img)
    power_spectrum = np.abs(fft_shift)**2
    return power_spectrum

def compute_radial_profile(power_spectrum):
    """Compute radial average of power spectrum"""
    rows, cols = power_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create radial grid
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Bin radii
    max_r = min(crow, ccol)
    bins = np.arange(0, max_r, 1)
    radial_profile = np.zeros(len(bins) - 1)
    
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    return bins[:-1], radial_profile

def compute_angular_variance(power_spectrum):
    """Compute angular variance around each radius"""
    rows, cols = power_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol)**2 + (y - crow)**2)
    theta = np.arctan2(y - crow, x - ccol)
    
    # Bin radii and angles
    max_r = min(crow, ccol)
    r_bins = np.linspace(0, max_r, 50)
    theta_bins = np.linspace(-np.pi, np.pi, 36)
    
    radial_std = []
    r_centers = []
    
    for i in range(len(r_bins) - 1):
        r_mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(r_mask):
            angular_vals = []
            for j in range(len(theta_bins) - 1):
                theta_mask = (theta >= theta_bins[j]) & (theta < theta_bins[j+1])
                combined = r_mask & theta_mask
                if np.any(combined):
                    angular_vals.append(np.mean(power_spectrum[combined]))
            if angular_vals:
                radial_std.append(np.std(angular_vals))
                r_centers.append((r_bins[i] + r_bins[i+1]) / 2)
    
    return np.array(r_centers), np.array(radial_std)

def compute_heatmap_metrics(image):
    """Compute statistical metrics for heatmap analysis"""
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'skew': float(np.mean(((image - np.mean(image)) / np.std(image))**3)),
        'kurtosis': float(np.mean(((image - np.mean(image)) / np.std(image))**4) - 3),
        'entropy': float(-np.sum(image * np.log(image + 1e-10)))
    }

def create_custom_colormap(name, colors):
    """Create custom colormap for heat visualization"""
    return LinearSegmentedColormap.from_list(name, colors)

# ============================================================================
# SPECTRAL PATTERN GENERATOR
# ============================================================================

def generate_fdm_soliton_pattern(size=256, k=1.0):
    """Generate FDM Soliton pattern with spectral characteristics"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    # FDM Soliton: ρ(r) ∝ [sin(kr)/(kr)]²
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(r > 0, (np.sin(k * r) / (k * r))**2, 1.0)
    
    return pattern / pattern.max()

def generate_pdp_entanglement_pattern(size=256, omega=0.5, fringe=1.0):
    """Generate PDP Entanglement pattern with spectral characteristics"""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # PDP Entanglement: oscillatory pattern from dark photon mixing
    pattern = np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe)) * np.sin(10 * R)
    pattern = pattern - pattern.min()
    pattern = pattern / pattern.max()
    
    return pattern

def generate_interference_pattern(size=256, k1=1.0, k2=1.5, theta=30):
    """Generate quantum interference pattern between soliton and entanglement"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r1 = np.sqrt(X**2 + Y**2)
    
    # Rotated coordinates for second source
    theta_rad = np.radians(theta)
    X2 = X * np.cos(theta_rad) - Y * np.sin(theta_rad)
    Y2 = X * np.sin(theta_rad) + Y * np.cos(theta_rad)
    r2 = np.sqrt(X2**2 + Y2**2)
    
    pattern1 = np.where(r1 > 0, (np.sin(k1 * r1) / (k1 * r1))**2, 1.0)
    pattern2 = np.where(r2 > 0, (np.sin(k2 * r2) / (k2 * r2))**2, 1.0)
    
    interference = (pattern1 + pattern2) / 2
    return interference / interference.max()

# ============================================================================
# SPECTRAL & COLOR ANALYSIS TAB
# ============================================================================

def spectral_analysis_tab():
    """Create the Spectral & Color Heat Pattern Analysis tab"""
    
    st.header("🌈 Spectral & Color Heat Pattern Analyzer")
    st.markdown("Analyzing quantum interference patterns, power spectra, and thermal color mappings")
    
    # Pattern selection
    pattern_type = st.radio(
        "Select Pattern Type",
        ["FDM Soliton", "PDP Entanglement", "Quantum Interference"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Parameters")
        
        if pattern_type == "FDM Soliton":
            k = st.slider("k (Soliton Scale)", 0.5, 3.0, 1.0, 0.05,
                         help="Controls soliton size: larger k = smaller, more compact core")
            pattern = generate_fdm_soliton_pattern(size=512, k=k)
            title = f"FDM Soliton (k={k:.2f})"
            
            st.markdown("""
            **FDM Soliton Formula:**
            $$\\rho(r) = \\rho_0 \\left[\\frac{\\sin(kr)}{kr}\\right]^2$$
            
            **Interpretation:** Ground state of ultra-light dark matter
            - Central core with quantum interference ripples
            - **k** controls soliton compactness
            - Bright central region = dark matter density peak
            """)
            
        elif pattern_type == "PDP Entanglement":
            omega = st.slider("Ω (Entanglement Strength)", 0.2, 1.5, 0.5, 0.05,
                             help="Photon-dark photon coupling strength")
            fringe = st.slider("Fringe Scale", 0.5, 3.0, 1.0, 0.1,
                              help="Quantum interference pattern scale")
            pattern = generate_pdp_entanglement_pattern(size=512, omega=omega, fringe=fringe)
            title = f"PDP Entanglement (Ω={omega:.2f}, fringe={fringe:.2f})"
            
            st.markdown("""
            **PDP Entanglement Formula:**
            $$\\mathcal{L}_{\\text{mix}} = \\frac{\\varepsilon}{2} F_{\\mu\\nu} F'^{\\mu\\nu}$$
            
            **Interpretation:** Dark photon mixing signatures
            - **Ω** controls entanglement strength
            - **Fringe** sets oscillation wavelength
            - Darker regions = stronger dark photon presence
            """)
            
        else:  # Quantum Interference
            k1 = st.slider("k₁ (First Source)", 0.5, 2.5, 1.0, 0.05)
            k2 = st.slider("k₂ (Second Source)", 0.5, 2.5, 1.5, 0.05)
            theta = st.slider("Separation Angle (°)", 0, 90, 30, 5)
            pattern = generate_interference_pattern(size=512, k1=k1, k2=k2, theta=theta)
            title = f"Quantum Interference (k₁={k1:.2f}, k₂={k2:.2f}, θ={theta}°)"
            
            st.markdown("""
            **Quantum Interference Formula:**
            $$I(r) = \\left[\\frac{\\sin(k_1 r_1)}{k_1 r_1}\\right]^2 + \\left[\\frac{\\sin(k_2 r_2)}{k_2 r_2}\\right]^2$$
            
            **Interpretation:** Interference between two quantum sources
            - **k₁, k₂** = soliton scales of each source
            - **θ** = angular separation
            - Moiré patterns = quantum coherence
            """)
    
    with col2:
        st.subheader("Spatial Pattern")
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pattern, cmap='plasma', origin='lower')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label="Intensity")
        st.pyplot(fig)
        
        # Download pattern
        from io import BytesIO
        import base64
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qcaus_{pattern_type.lower().replace(' ', '_')}_{timestamp}.png"
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download Pattern</a>'
        st.markdown(href, unsafe_allow_html=True)
        plt.close(fig)
    
    # ========================================================================
    # SPECTRAL ANALYSIS
    # ========================================================================
    
    st.subheader("📊 Spectral Analysis")
    
    # Compute power spectrum
    power_spec = compute_power_spectrum(pattern)
    r_vals, radial_profile = compute_radial_profile(power_spec)
    r_centers, angular_variance = compute_angular_variance(power_spec)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Power Spectrum (2D Fourier)**")
        fig_ps, ax_ps = plt.subplots(figsize=(6, 6))
        ax_ps.imshow(np.log(power_spec + 1), cmap='hot', origin='lower')
        ax_ps.set_title("Power Spectrum (log scale)")
        ax_ps.axis('off')
        st.pyplot(fig_ps)
        plt.close(fig_ps)
    
    with col_b:
        st.write("**Radial Power Profile**")
        fig_rad, ax_rad = plt.subplots(figsize=(6, 4))
        ax_rad.plot(r_vals, radial_profile, 'b-', linewidth=2)
        ax_rad.set_xlabel("Radial Frequency")
        ax_rad.set_ylabel("Power")
        ax_rad.set_title("Radial Power Spectrum")
        ax_rad.set_xscale('log')
        ax_rad.set_yscale('log')
        ax_rad.grid(True, alpha=0.3)
        st.pyplot(fig_rad)
        plt.close(fig_rad)
    
    # ========================================================================
    # HEATMAP METRICS
    # ========================================================================
    
    st.subheader("🔥 Heatmap Metrics")
    
    metrics = compute_heatmap_metrics(pattern)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Mean Intensity", f"{metrics['mean']:.4f}")
    col_m2.metric("Std Dev", f"{metrics['std']:.4f}")
    col_m3.metric("Max Intensity", f"{metrics['max']:.4f}")
    col_m4.metric("Min Intensity", f"{metrics['min']:.4f}")
    
    col_m5, col_m6, col_m7 = st.columns(3)
    col_m5.metric("Skewness", f"{metrics['skew']:.3f}", help="Positive = right-skewed, negative = left-skewed")
    col_m6.metric("Kurtosis", f"{metrics['kurtosis']:.3f}", help=">3 = peaked, <3 = flat")
    col_m7.metric("Entropy", f"{metrics['entropy']:.3f}", help="Information content")
    
    # ========================================================================
    # COLOR MAP COMPARISON
    # ========================================================================
    
    st.subheader("🎨 Color Map Comparison")
    
    # Define color maps to compare
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm', 'seismic', 'RdYlBu_r']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, cmap_name in enumerate(cmaps):
        ax = axes[i]
        im = ax.imshow(pattern, cmap=cmap_name, origin='lower')
        ax.set_title(cmap_name)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # ========================================================================
    # ANGULAR VARIANCE
    # ========================================================================
    
    st.subheader("📐 Angular Variance Analysis")
    
    fig_ang, ax_ang = plt.subplots(figsize=(8, 5))
    ax_ang.plot(r_centers, angular_variance, 'r-', linewidth=2)
    ax_ang.set_xlabel("Radial Distance (pixels)")
    ax_ang.set_ylabel("Angular Variance")
    ax_ang.set_title("Angular Variance vs Radius")
    ax_ang.grid(True, alpha=0.3)
    st.pyplot(fig_ang)
    plt.close(fig_ang)
    
    # ========================================================================
    # 3D SURFACE PLOT
    # ========================================================================
    
    st.subheader("🗺️ 3D Surface Visualization")
    
    # Downsample for faster rendering
    step = 4
    X = np.arange(0, pattern.shape[0], step)
    Y = np.arange(0, pattern.shape[1], step)
    X, Y = np.meshgrid(X, Y)
    Z = pattern[::step, ::step]
    
    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')])
    fig_3d.update_layout(
        title="3D Heatmap Surface",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Intensity",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # ========================================================================
    # PHYSICS EXPLANATION
    # ========================================================================
    
    with st.expander("🔬 Understanding Spectral & Color Patterns", expanded=False):
        st.markdown(r"""
        ### 🎨 Color Heat Patterns
        
        | Color Map | Best For | Interpretation |
        |-----------|----------|----------------|
        | **Plasma** | Soliton cores | Bright central region = dark matter density peak |
        | **Viridis** | PDP patterns | Smooth transitions in entanglement strength |
        | **Inferno** | High contrast | Sharp boundaries between quantum states |
        | **Hot** | Thermal analogy | Intensity as "temperature" of quantum field |
        
        ### 📊 Spectral Signatures
        
        | Feature | What It Reveals |
        |---------|-----------------|
        | **Central Peak** | Soliton core density |
        | **Ring Patterns** | Quantum interference fringes |
        | **Radial Profile** | 1/r² falloff characteristic of solitons |
        | **Angular Variance** | Isotropy vs anisotropy in pattern |
        
        ### 🔥 Heatmap Metrics
        
        - **Skewness > 0**: More high-intensity pixels (bright core)
        - **Skewness < 0**: More low-intensity pixels (diffuse)
        - **Kurtosis > 3**: Sharp, peaked distribution
        - **Kurtosis < 3**: Broad, flat distribution
        - **Higher Entropy**: More complex, information-rich pattern
        
        ### 📡 Connection to Your Radar Data
        
        The spectral patterns from FDM solitons and PDP entanglement can be used to:
        1. **Classify** different types of quantum signatures
        2. **Calibrate** the PDP filter parameters (Ω, fringe scale)
        3. **Validate** theoretical predictions against observed patterns
        """)

# ============================================================================
# MAIN APP - ADD THIS TAB
# ============================================================================

# Add this to your main app's tab creation:
# tab1, tab2, tab3, tab4, tab5 = st.tabs([
#     "🔭 QCI AstroEntangle",
#     "⚡ Magnetar QED", 
#     "🌀 Primordial Entanglement",
#     "📊 QCIS Power Spectra",
#     "🌈 Spectral & Color Analysis"  # NEW TAB
# ])
#
# with tab5:
#     spectral_analysis_tab()
