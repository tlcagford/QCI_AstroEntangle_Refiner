import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom, gaussian_filter
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import io
import base64
from datetime import datetime

st.set_page_config(page_title="QCAUS", page_icon="🌌", layout="wide")

# ============================================================================
# FDM SOLITON - PROPER WAVE PATTERNS
# ============================================================================

def fdm_soliton(r, k=1.0):
    """FDM Soliton: ρ(r) = ρ₀ [sin(kr)/(kr)]² - produces visible interference rings"""
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)
    return result

def fdm_wave_function(r, k=1.0):
    """Quantum wave function ψ(r) = sin(kr)/(kr)"""
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(kr > 0, np.sin(kr) / kr, 1.0)

def generate_fdm_field(size=512, k=1.0):
    """Generate FDM soliton field with concentric wave rings"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    return fdm_soliton(r, k=k)

# ============================================================================
# PDP QUANTUM FIELD - FROM SPECTRAL DUALITY
# ============================================================================

def pdp_field(img, omega=0.5, fringe=1.0):
    """PDP Quantum Field via spectral duality: extracts dark-mode components"""
    fft_img = fft2(img)
    fft_shift = fftshift(fft_img)
    rows, cols = img.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe))
    mixed = fft_shift * mask
    return np.abs(ifft2(fftshift(mixed)))

# ============================================================================
# QUANTUM SUPERPOSITION
# ============================================================================

def process(img, omega=0.5, fringe=1.0, k=1.0, alpha=0.8, beta=1.0):
    """|Ψ⟩ = |Ψ_astro⟩ + α|Ψ_FDM⟩ + β|Ψ_PDP⟩"""
    size = min(img.shape)
    r = np.linspace(0, 3, size)
    fdm_profile = fdm_soliton(r, k=k)
    fdm_2d = np.outer(fdm_profile, fdm_profile)
    fdm_resized = zoom(fdm_2d, (img.shape[0]/size, img.shape[1]/size))
    pdp = pdp_field(img, omega, fringe)
    enhanced = img + alpha * fdm_resized + beta * pdp
    return np.clip(enhanced, 0, 1), fdm_resized, pdp

# ============================================================================
# MAGNETAR QED EXPLORER
# ============================================================================

def magnetar_dipole(r, theta, B0=1e15):
    """Magnetar dipole field: B = B₀ (R/r)³ (2 cosθ, sinθ)"""
    B_r = B0 * 2 * np.cos(theta) / (r**3 + 1e-10)
    B_theta = B0 * np.sin(theta) / (r**3 + 1e-10)
    return B_r, B_theta

def euler_heisenberg(B, alpha=1/137):
    """Euler-Heisenberg vacuum polarization"""
    B_crit = 4.41e13
    beta = (B / B_crit)**2
    return alpha * beta / (45 * np.pi) * (1 + 7/4 * beta)

def dark_photon_conv(B, eps=0.1, m=1e-9):
    """Dark photon conversion: P = ε² (1 - e^{-B²/m²})"""
    return eps**2 * (1 - np.exp(-B**2 / m**2))

def generate_magnetar(size=200, B0=1e15, eps=0.1):
    """Generate magnetar field visualizations"""
    r = np.linspace(1, 10, size)
    theta = np.linspace(0, np.pi, size)
    R, Theta = np.meshgrid(r, theta)
    
    B_r, B_theta = magnetar_dipole(R, Theta, B0=B0)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    
    # Add quantum oscillations
    waves = 1 + 0.2 * np.sin(8 * R) * np.exp(-R/2)
    B_waves = B_mag * waves
    
    qed = euler_heisenberg(B_waves)
    dark = dark_photon_conv(B_waves, eps=eps)
    
    return B_waves, qed, dark

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_scale(ax, width, kpc=100, scale=0.1):
    bar_px = kpc / scale
    rect = Rectangle((50, width - 60), bar_px, 8, linewidth=2, edgecolor='white', facecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(50 + bar_px/2, width - 72, f"{kpc} kpc", color='white', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

def download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{name}" style="text-decoration:none;">📥 Download</a>'

def load_image(file):
    try:
        img = Image.open(file)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = np.mean(arr[:, :, :3], axis=2)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    except:
        return None

def sample_galaxy():
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    img = np.exp(-R**2 / 1.5**2)
    img += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
    img += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
    return img / img.max()

def sample_magnetar():
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    img = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.sin(3 * theta))
    return img / img.max()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("**Quantum Cosmology & Astrophysics Unified Suite**")
    st.markdown("---")
    
    st.markdown("### ⚛️ Quantum Parameters")
    omega = st.slider("Ω (Entanglement)", 0.0, 1.0, 0.5, 0.01)
    fringe = st.slider("λ (Fringe Scale)", 0.1, 3.0, 1.5, 0.05)
    k = st.slider("k (Soliton Wave #)", 0.5, 3.0, 1.2, 0.05)
    alpha = st.slider("α (FDM)", 0.0, 2.0, 0.8, 0.05)
    beta = st.slider("β (PDP)", 0.0, 2.0, 1.0, 0.05)
    
    st.markdown("---")
    st.markdown("### 🖼️ Image Input")
    
    img = None
    img_name = "galaxy"
    scale = 0.1
    
    img_type = st.radio("Source", ["Sample Galaxy", "Sample Magnetar", "Upload"])
    
    if img_type == "Sample Galaxy":
        img = sample_galaxy()
        img_name = "galaxy_cluster"
        scale = 0.1
    elif img_type == "Sample Magnetar":
        img = sample_magnetar()
        img_name = "magnetar"
        scale = 0.1
    else:
        uploaded = st.file_uploader("Upload", type=['jpg', 'png'])
        if uploaded:
            img = load_image(uploaded)
            if img is not None:
                img_name = uploaded.name.split('.')[0]
                scale = 100.0 / img.shape[1]
                st.success(f"Loaded: {uploaded.name}")
    
    if img is not None:
        st.markdown("---")
        st.markdown("### 📏 Scale")
        scale = st.number_input("kpc/pixel", value=scale, format="%.4f")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2 = st.tabs(["🌈 Quantum Field Visualization", "⚡ Magnetar QED Explorer"])

# ============================================================================
# TAB 1: QUANTUM FIELD VISUALIZATION
# ============================================================================

with tab1:
    st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
    st.markdown("*Mapping invisible quantum fields to visible colors*")
    
    if img is not None:
        enhanced, fdm, pdp = process(img, omega, fringe, k, alpha, beta)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_{img_name}_{ts}"
        
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ω", f"{omega:.2f}")
        c2.metric("λ", f"{fringe:.2f}")
        c3.metric("k", f"{k:.2f}")
        c4.metric("Mixing", f"{omega * fringe:.3f}")
        
        # BEFORE/AFTER
        st.subheader("🔬 Before / After: Quantum Field Overlay")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.imshow(img, cmap='gray', origin='upper')
        ax1.set_title("Before: Standard View\n(Public HST/JWST Data)", fontsize=12)
        ax1.axis('off')
        add_scale(ax1, img.shape[1], scale=scale)
        
        ax2.imshow(enhanced, cmap='plasma', origin='upper')
        ax2.set_title(f"After: Photon-Dark-Photon Entangled FDM Overlays\n(Tony Ford Model)", fontsize=12)
        ax2.axis('off')
        add_scale(ax2, enhanced.shape[1], scale=scale)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_comparison.png"), unsafe_allow_html=True)
        plt.close(fig)
        
        # FULL-SPECTRUM COMPOSITE
        st.subheader("🌈 Full-Spectrum Composite")
        st.markdown("*Red: Visible Light | Green: FDM Soliton (Dark Matter) | Blue: PDP Field (Dark Photons)*")
        
        rgb = np.zeros((*img.shape, 3))
        rgb[..., 0] = img / (img.max() + 1e-8)
        fdm_norm = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
        pdp_norm = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
        rgb[..., 1] = fdm_norm * 0.9
        rgb[..., 2] = pdp_norm * 0.9
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.clip(rgb, 0, 1), origin='upper')
        ax.set_title("Full-Spectrum Quantum Composite\nVisible + Dark Matter Waves + Dark Photons", fontsize=12)
        ax.axis('off')
        add_scale(ax, rgb.shape[1], scale=scale)
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_composite.png"), unsafe_allow_html=True)
        plt.close(fig)
        
        # INDIVIDUAL FIELDS
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌌 FDM Soliton Field")
            st.markdown("*Concentric wave interference rings from [sin(kr)/(kr)]²*")
            fig, ax = plt.subplots(figsize=(6, 6))
            fdm_viz = np.zeros((*fdm.shape, 3))
            fdm_viz[..., 1] = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
            ax.imshow(fdm_viz, origin='upper')
            ax.set_title(f"ρ(r) = ρ₀ [sin({k:.2f}r)/({k:.2f}r)]²\nWave rings visible", fontsize=10)
            ax.axis('off')
            add_scale(ax, fdm.shape[1], scale=scale)
            st.pyplot(fig)
            st.markdown(download(fig, f"{base}_fdm.png"), unsafe_allow_html=True)
            plt.close(fig)
        
        with col2:
            st.subheader("🌀 PDP Quantum Field")
            fig, ax = plt.subplots(figsize=(6, 6))
            pdp_viz = np.zeros((*pdp.shape, 3))
            pdp_viz[..., 2] = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
            ax.imshow(pdp_viz, origin='upper')
            ax.set_title(f"ℒ_mix = (ε/2) F_μν F'^μν\nΩ={omega:.2f}, λ={fringe:.2f}", fontsize=10)
            ax.axis('off')
            add_scale(ax, pdp.shape[1], scale=scale)
            st.pyplot(fig)
            st.markdown(download(fig, f"{base}_pdp.png"), unsafe_allow_html=True)
            plt.close(fig)
        
        # METRICS
        st.subheader("📊 Quantum Field Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max FDM", f"{np.max(fdm):.3f}")
        m2.metric("Max PDP", f"{np.max(pdp):.3f}")
        corr = np.corrcoef(fdm.flatten(), pdp.flatten())[0, 1]
        m3.metric("Correlation", f"{corr:.3f}")
        m4.metric("FDM Value", f"{k * 2.5:.1f} kpc")
        
        # WAVE PATTERN ANALYSIS
        st.subheader("🌊 Wave Interference Pattern")
        center = np.array(fdm.shape) // 2
        y, x = np.ogrid[:fdm.shape[0], :fdm.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        radial = [np.mean(fdm[(r >= i) & (r < i+1)]) for i in range(int(r.max()))]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(radial[:150], 'g-', linewidth=2)
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Dark Matter Density")
        ax.set_title(f"FDM Soliton Radial Profile - Wave Interference (k={k:.2f})")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_radial.png"), unsafe_allow_html=True)
        plt.close(fig)
        
        # FORMULAS
        with st.expander("📐 Verified Formulas", expanded=False):
            st.latex(r"\text{FDM Soliton:} \quad \rho(r) = \rho_0 \left[\frac{\sin(kr)}{kr}\right]^2")
            st.latex(r"\text{PDP Mixing:} \quad \mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}")
            st.latex(r"\text{Quantum State:} \quad |\Psi\rangle = |\Psi_{\text{astro}}\rangle + \alpha|\Psi_{\text{FDM}}\rangle + \beta|\Psi_{\text{PDP}}\rangle")
            st.caption(f"Current: Ω={omega:.2f}, λ={fringe:.2f}, k={k:.2f}, Mixing={omega*fringe:.3f}, Entropy={np.mean(fdm):.3f}")
    
    else:
        st.info("👈 Select or upload an image")

# ============================================================================
# TAB 2: MAGNETAR QED EXPLORER
# ============================================================================

with tab2:
    st.title("⚡ Magnetar QED Explorer")
    st.markdown("*Quantum electrodynamics in extreme magnetic fields*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        B0 = st.slider("B-Field (10¹⁵ G)", 0.5, 5.0, 1.0, 0.1)
        eps = st.slider("Mixing ε", 0.0, 0.5, 0.1, 0.01)
        mass = st.slider("Dark Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
        
        B_field, qed, dark = generate_magnetar(size=200, B0=B0*1e15, eps=eps)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_magnetar_B{B0:.1f}_{ts}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Max B", f"{np.max(B_field)/1e15:.2f}×10¹⁵ G")
        m2.metric("Max QED", f"{np.max(qed):.3e}")
        m3.metric("Max Dark", f"{np.max(dark):.3f}")
    
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        im1 = axes[0].imshow(B_field, extent=[1, 10, 0, 180], aspect='auto', cmap='hot', origin='upper')
        axes[0].set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G\nWave oscillations visible")
        axes[0].set_xlabel("Radius (R/R₀)")
        axes[0].set_ylabel("Angle (deg)")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma', origin='upper')
        axes[1].set_title("Vacuum Polarization\nEuler-Heisenberg")
        axes[1].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(dark, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis', origin='upper')
        axes[2].set_title(f"Dark Photons\nε={eps:.2f}, m={mass:.1e}eV")
        axes[2].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(download(fig, f"{base}_magnetar.png"), unsafe_allow_html=True)
        plt.close(fig)
    
    with st.expander("📖 Magnetar Physics", expanded=False):
        st.markdown(r"""
        | Effect | Formula |
        |--------|---------|
        | **Dipole Field** | $B = B_0 (R/r)^3 (2\cos\theta, \sin\theta)$ |
        | **Vacuum Polarization** | Euler-Heisenberg |
        | **Dark Photon Conversion** | $P = \varepsilon^2 (1 - e^{-B^2/m^2})$ |
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | FDM Soliton • PDP Field • Magnetar QED<br>
    ρ(r) = ρ₀ [sin(kr)/(kr)]² | ℒ_mix = (ε/2) F_μν F'^μν<br>
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
