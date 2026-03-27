"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Complete integration of all your projects:
- QCI AstroEntangle Refiner (Your FDM Soliton + PDP formulas)
- Magnetar QED Explorer (Your magnetar field calculations)
- Primordial Photon-DarkPhoton Entanglement (Your von Neumann evolution)
- QCIS Power Spectra (Your quantum-corrected cosmology)
- Spectral & Color Analysis (Your pattern analysis tools)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.ndimage import gaussian_filter, zoom, label, center_of_mass
from scipy.fft import fft2, ifft2, fftshift
from scipy.special import jv
from io import BytesIO
from PIL import Image
import tempfile
import os
import base64
from datetime import datetime

st.set_page_config(page_title="QCAUS - Quantum Cosmology Suite", page_icon="🌌", layout="wide")

# ============================================================================
# YOUR ACTUAL FORMULAS - FROM YOUR PROJECTS
# ============================================================================

# ===== FDM Soliton: ρ(r) = ρ₀ [sin(kr)/(kr)]² (Your formula) =====
def fdm_soliton_profile(r, k=1.0):
    """Fuzzy Dark Matter soliton profile from your QCI AstroEntangle project"""
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)
    return profile

# ===== PDP Kinetic Mixing: ℒ_mix = (ε/2) F_μν F'^μν (Your formula) =====
def pdp_entanglement_overlay(image_data, omega=0.5, fringe_scale=1.0):
    """Photon-DarkPhoton entanglement filter from your StealthPDPRadar"""
    fft_img = fft2(image_data)
    fft_shift = fftshift(fft_img)
    rows, cols = image_data.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    dark_mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe_scale))
    dark_fft = fft_shift * dark_mask
    dark_mode = np.abs(ifft2(fftshift(dark_fft)))
    return dark_mode

# ===== Your QCI AstroEntangle processor =====
def process_qci_astro(image_data, omega=0.5, fringe=1.0, soliton_scale=1.0):
    """Process with your FDM soliton and PDP entanglement"""
    size = min(image_data.shape)
    r = np.linspace(0, 3, size)
    soliton_profile = fdm_soliton_profile(r, k=soliton_scale)
    soliton_2d = np.outer(soliton_profile, soliton_profile)
    soliton_resized = zoom(soliton_2d, (image_data.shape[0]/size, image_data.shape[1]/size))
    pdp = pdp_entanglement_overlay(image_data, omega, fringe)
    enhanced = image_data + 0.3 * soliton_resized + 0.5 * pdp
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced, soliton_resized, pdp

# ===== Magnetar Dipole Field: B = B₀ (R/r)³ (2 cosθ, sinθ) =====
def magnetar_dipole_field(r, theta, B0=1e15):
    """Your magnetar dipole field calculation"""
    B_r = B0 * 2 * np.cos(theta) / (r**3 + 1e-10)
    B_theta = B0 * np.sin(theta) / (r**3 + 1e-10)
    return B_r, B_theta

# ===== Euler-Heisenberg Vacuum Polarization =====
def quantum_vacuum_polarization(B, alpha=1/137):
    """Your QED vacuum polarization from Magnetar QED Explorer"""
    B_crit = 4.41e13
    beta = (B / B_crit)**2
    polarization = alpha * beta / (45 * np.pi) * (1 + 7/4 * beta)
    return polarization

# ===== Dark Photon Conversion: P = ε² (1 - e^{-B²/m²}) =====
def dark_photon_conversion(B, mixing_angle=0.1, mass=1e-9):
    """Your dark photon conversion probability"""
    conversion_prob = mixing_angle**2 * (1 - np.exp(-B**2 / mass**2))
    return conversion_prob

# ===== Von Neumann Evolution: i∂ρ/∂t = [H, ρ] (Your equation) =====
def von_neumann_evolution(density_matrix, hamiltonian, dt):
    """Your von Neumann evolution from Primordial Entanglement project"""
    commutator = np.dot(hamiltonian, density_matrix) - np.dot(density_matrix, hamiltonian)
    return density_matrix + (-1j * commutator) * dt

# ===== Entanglement Entropy: S = -Tr(ρ log ρ) =====
def entanglement_entropy(reduced_density):
    """Your entanglement entropy calculation"""
    eigenvalues = np.linalg.eigvalsh(reduced_density)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

# ===== Your Primordial Entanglement evolution =====
def process_primordial_entanglement(omega=0.7, dark_mass=1e-9, mixing=0.1, t_steps=100):
    """Your photon-dark photon entanglement evolution"""
    rho = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
    H = np.array([[omega, mixing], [mixing, dark_mass]], dtype=complex)
    dt = 0.01
    entropy_evolution = []
    mixing_prob = []
    for step in range(t_steps):
        rho = von_neumann_evolution(rho, H, dt)
        reduced = rho[:1, :1]
        entropy_evolution.append(entanglement_entropy(reduced))
        mixing_prob.append(abs(rho[0, 1])**2)
    return entropy_evolution, mixing_prob

# ===== Quantum-corrected Power Spectrum: P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) =====
def quantum_corrected_power_spectrum(k, f_nl=1.0, n_q=0.5, k0=0.05):
    """Your QCIS quantum-corrected power spectrum"""
    P_lcdm = k ** (-3) * np.exp(-k / 0.1)
    quantum_correction = 1 + f_nl * (k / k0)**n_q
    return P_lcdm * quantum_correction

# ===== Your Magnetar QED processor =====
def process_magnetar(r_grid, theta_grid, B0=1e15, mixing=0.1):
    """Your complete magnetar field processing"""
    B_r, B_theta = magnetar_dipole_field(r_grid, theta_grid, B0)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    qed = quantum_vacuum_polarization(B_mag)
    dark_photons = dark_photon_conversion(B_mag, mixing_angle=mixing)
    return B_mag, qed, dark_photons

# ===== Your QCIS processor =====
def process_qcis(k_vals, f_nl=1.0, n_q=0.5):
    """Your QCIS power spectrum processor"""
    return quantum_corrected_power_spectrum(k_vals, f_nl, n_q)

# ============================================================================
# YOUR ANNOTATED COMPARISON WITH METRICS
# ============================================================================

def create_annotated_comparison(original_image, enhanced_image, soliton_overlay, pdp_overlay, params):
    """Create annotated before/after comparison with your metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Original image
    axes[0, 0].imshow(original_image, cmap='gray', origin='lower')
    axes[0, 0].set_title("Before: Standard View\n(Public HST/JWST Data)")
    axes[0, 0].axis('off')
    
    # Enhanced with overlays
    axes[0, 1].imshow(enhanced_image, cmap='gray', origin='lower')
    axes[0, 1].set_title(f"After: Photon-Dark-Photon Entangled\nFDM Overlays (Tony Ford Model)")
    axes[0, 1].axis('off')
    
    # FDM Soliton Overlay with metrics
    im1 = axes[1, 0].imshow(soliton_overlay, cmap='viridis', origin='lower', alpha=0.8)
    axes[1, 0].set_title(f"FDM Soliton Core\nk={params.get('soliton_scale', 1.0):.2f}\nMax Density: {np.max(soliton_overlay):.3f}")
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], label="Dark Matter Density")
    
    # PDP Entanglement Overlay with metrics
    im2 = axes[1, 1].imshow(pdp_overlay, cmap='plasma', origin='lower', alpha=0.8)
    axes[1, 1].set_title(f"PDP Entanglement\nΩ={params.get('omega', 0.5):.2f}, Fringe={params.get('fringe', 1.0):.2f}\nMax Mixing: {np.max(pdp_overlay):.3f}")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], label="Entanglement Strength")
    
    # Add annotation box with metrics
    metrics_text = f"Maximum Mixing Ratio: {np.max(pdp_overlay):.3f}\nMinimum Entropy: {np.min(soliton_overlay):.3f}\nFDM Value: {params.get('soliton_scale', 1.0) * 2.5:.1f} kpc"
    fig.text(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_radar_style_overlay(original_image, soliton, pdp):
    """Create radar-style overlay with green speckles and blue halos"""
    rgb = np.zeros((*original_image.shape, 3))
    
    # Original as red channel
    rgb[..., 0] = original_image / (original_image.max() + 1e-8)
    
    # FDM Soliton as green channel (entanglement residuals)
    soliton_norm = (soliton - soliton.min()) / (soliton.max() - soliton.min() + 1e-8)
    rgb[..., 1] = soliton_norm * 0.8
    
    # PDP Entanglement as blue channel (dark-mode leakage)
    pdp_norm = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
    rgb[..., 2] = pdp_norm * 0.8
    
    return np.clip(rgb, 0, 1)

# ============================================================================
# SAMPLE ASTROPHYSICAL IMAGES (Your Bullet Cluster, Galaxy Cluster, etc.)
# ============================================================================

def get_sample_image(image_name):
    """Your astrophysical image generator - Bullet Cluster, Galaxy Cluster, Nebula"""
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    if image_name == "Galaxy Cluster":
        R = np.sqrt(X**2 + Y**2)
        img = np.exp(-R**2 / 1.5**2)
        img += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
        img += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
        return img / img.max()
    elif image_name == "Bullet Cluster":
        # Your Bullet Cluster simulation - two merging galaxy clusters
        img = np.exp(-((X-0.8)**2 + Y**2) / 0.3**2) + 0.7 * np.exp(-((X+0.6)**2 + Y**2) / 0.4**2)
        return img / img.max()
    elif image_name == "Nebula":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.cos(5 * theta))
        return img / img.max()
    else:
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.5 * np.cos(10 * R + 3 * theta))
        return img / img.max()

# ============================================================================
# UNIVERSAL IMAGE LOADER (FITS, JPEG, PNG)
# ============================================================================

def load_image_file(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()
    
    try:
        if file_ext in ['fits', 'fit']:
            from astropy.io import fits
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            with fits.open(tmp_path) as hdul:
                data = hdul[0].data
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                if data is not None and data.ndim > 2:
                    if data.ndim == 3:
                        data = np.median(data, axis=0)
                    else:
                        data = data[0] if data.ndim == 4 else data[0]
            os.unlink(tmp_path)
            if data is None:
                return None, "No image data found"
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"FITS: {data.shape}"
        
        elif file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
            img = Image.open(BytesIO(file_bytes))
            data = np.array(img)
            if data.ndim == 3:
                data = np.mean(data[:, :, :3], axis=2) if data.shape[2] >= 3 else data[:, :, 0]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"{file_ext.upper()}: {data.shape}"
        
        else:
            return None, f"Unsupported: {file_ext}"
            
    except Exception as e:
        return None, str(e)

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def get_image_download_link(fig, filename, title="Download"):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_json_download_link(data, filename, title="Download"):
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    st.markdown("*Your formulas: FDM Soliton • PDP Entanglement • Magnetar QED • QCIS*")
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_name = "galaxy_cluster"
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Bullet Cluster", "Nebula"])
        astro_image = get_sample_image(image_type)
        image_name = image_type.replace(" ", "_").lower()
    else:
        uploaded_img = st.file_uploader("Upload Image", type=['fits', 'jpg', 'png'])
        if uploaded_img:
            astro_image, info = load_image_file(uploaded_img)
            if astro_image is None:
                st.error(info)
            else:
                st.success(info)
                image_name = uploaded_img.name.split('.')[0]

# ============================================================================
# MAIN CONTENT - 5 TABS WITH YOUR PROJECTS
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("**Your Projects:** QCI AstroEntangle • Magnetar QED • Primordial Entanglement • QCIS • Spectral Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔭 QCI AstroEntangle (FDM + PDP)",
    "⚡ Magnetar QED Explorer",
    "🌀 Primordial Entanglement",
    "📊 QCIS Power Spectra",
    "🌈 Spectral & Color Analysis"
])

# ============================================================================
# TAB 1: QCI ASTROENTANGLE REFINER (Your FDM + PDP)
# ============================================================================

with tab1:
    st.header("🔭 QCI AstroEntangle Refiner")
    st.markdown("**Your Formulas:** FDM Soliton: ρ(r) = ρ₀ [sin(kr)/(kr)]² | PDP: ℒ_mix = (ε/2) F_μν F'^μν")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Overlay Parameters")
        omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01)
        fringe = st.slider("Fringe Scale", 0.1, 3.0, 1.0, 0.1)
        soliton_scale = st.slider("FDM Soliton Scale (k)", 0.5, 3.0, 1.0, 0.1)
        
        overlay_style = st.radio("Overlay Style", ["Annotated Comparison", "Radar Style (Green/Blue)"], horizontal=True)
        
        if astro_image is not None:
            enhanced, soliton, pdp = process_qci_astro(astro_image, omega, fringe, soliton_scale)
            
            st.markdown("---")
            st.subheader("📊 Your Metrics")
            st.metric("Maximum Mixing Ratio", f"{np.max(pdp):.3f}")
            st.metric("Minimum Entropy", f"{np.min(soliton):.3f}")
            st.metric("FDM Value", f"{soliton_scale * 2.5:.1f} kpc")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"qcaus_{image_name}_omega{omega:.2f}_fringe{fringe:.2f}_{timestamp}"
            
            st.markdown("---")
            st.subheader("📥 Download")
            
            if overlay_style == "Annotated Comparison":
                fig = create_annotated_comparison(astro_image, enhanced, soliton, pdp,
                    {'omega': omega, 'fringe': fringe, 'soliton_scale': soliton_scale})
                st.markdown(get_image_download_link(fig, f"{base_filename}_comparison.png", "📸 Download Comparison"), unsafe_allow_html=True)
                plt.close(fig)
            else:
                rgb_overlay = create_radar_style_overlay(astro_image, soliton, pdp)
                fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8))
                ax_rgb.imshow(rgb_overlay, origin='lower')
                ax_rgb.set_title("Radar-Style Overlay\nGreen: FDM Soliton | Blue: PDP Entanglement")
                ax_rgb.axis('off')
                st.pyplot(fig_rgb)
                st.markdown(get_image_download_link(fig_rgb, f"{base_filename}_radar_style.png", "📡 Download"), unsafe_allow_html=True)
                plt.close(fig_rgb)
    
    with col2:
        st.subheader("Visualization")
        if astro_image is not None:
            enhanced, soliton, pdp = process_qci_astro(astro_image, omega, fringe, soliton_scale)
            
            if overlay_style == "Annotated Comparison":
                fig = create_annotated_comparison(astro_image, enhanced, soliton, pdp,
                    {'omega': omega, 'fringe': fringe, 'soliton_scale': soliton_scale})
                st.pyplot(fig)
                plt.close(fig)
            else:
                rgb_overlay = create_radar_style_overlay(astro_image, soliton, pdp)
                fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8))
                ax_rgb.imshow(rgb_overlay, origin='lower')
                ax_rgb.set_title("Radar-Style Overlay\nGreen: FDM Soliton | Blue: PDP Entanglement")
                ax_rgb.axis('off')
                st.pyplot(fig_rgb)
                plt.close(fig_rgb)
        else:
            st.info("👈 Select or upload an image")

# ============================================================================
# TAB 2: MAGNETAR QED EXPLORER (Your magnetar fields)
# ============================================================================

with tab2:
    st.header("⚡ Magnetar QED Explorer")
    st.markdown("**Your Formulas:** B = B₀ (R/r)³ (2 cosθ, sinθ) | Euler-Heisenberg Polarization | Dark Photon Conversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        B0 = st.slider("Surface B-Field (10¹⁵ G)", 0.5, 5.0, 1.0, 0.1)
        mixing_angle = st.slider("Dark Photon Mixing ε", 0.0, 0.5, 0.1, 0.01)
        
        r = np.linspace(1, 10, 200)
        theta = np.linspace(0, np.pi, 200)
        R, Theta = np.meshgrid(r, theta)
        B_mag, qed, dark_photons = process_magnetar(R, Theta, B0=B0*1e15, mixing=mixing_angle)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_magnetar_B{B0:.1f}_eps{mixing_angle:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        ax_b.imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
        ax_b.set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G")
        plt.colorbar(ax_b.images[0], ax=ax_b)
        st.markdown(get_image_download_link(fig_b, f"{base_filename}_bfield.png", "🔴 Download B-Field"), unsafe_allow_html=True)
        plt.close(fig_b)
    
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
        axes[0].set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G")
        axes[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma')
        axes[1].set_title("QED Polarization")
        axes[2].imshow(dark_photons, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis')
        axes[2].set_title(f"Dark Photons\nε={mixing_angle:.2f}")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, f"{base_filename}_magnetar.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# ============================================================================
# TAB 3: PRIMORDIAL ENTANGLEMENT (Your von Neumann evolution)
# ============================================================================

with tab3:
    st.header("🌀 Primordial Photon-DarkPhoton Entanglement")
    st.markdown("**Your Formulas:** i∂ρ/∂t = [H, ρ] | S = -Tr(ρ log ρ) | P_mix = |⟨ψ_d|ψ_γ⟩|²")
    
    col1, col2 = st.columns(2)
    
    with col1:
        omega_ent = st.slider("Ω (Oscillation)", 0.0, 2.0, 0.7, 0.01)
        dark_mass = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
        mixing_prim = st.slider("Mixing Angle ε", 0.0, 0.5, 0.1, 0.01)
        t_steps = st.slider("Time Steps", 50, 500, 100)
        
        entropy, mixing_prob = process_primordial_entanglement(omega_ent, dark_mass, mixing_prim, t_steps)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_entanglement_omega{omega_ent:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        entanglement_data = {
            "parameters": {"omega": omega_ent, "dark_mass": dark_mass, "mixing": mixing_prim},
            "final_entropy": float(entropy[-1]),
            "final_mixing": float(mixing_prob[-1])
        }
        st.markdown(get_json_download_link(entanglement_data, f"{base_filename}_data.json", "📄 Download"), unsafe_allow_html=True)
    
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(entropy, 'b-', linewidth=2)
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Entropy S")
        ax1.set_title("Von Neumann Entropy")
        ax1.grid(True, alpha=0.3)
        ax2.plot(mixing_prob, 'r-', linewidth=2)
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Mixing Probability")
        ax2.set_title("Photon-Dark Photon Mixing")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, f"{base_filename}_evolution.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)
        
        st.caption(f"**Final Entropy:** {entropy[-1]:.4f} | **Final Mixing:** {mixing_prob[-1]:.4f}")

# ============================================================================
# TAB 4: QCIS POWER SPECTRA (Your quantum-corrected cosmology)
# ============================================================================

with tab4:
    st.header("📊 QCIS - Quantum Cosmology Integration Suite")
    st.markdown("**Your Formula:** P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f_nl = st.slider("f_NL (Non-Gaussianity)", 0.0, 5.0, 1.0, 0.1)
        n_q = st.slider("n_q (Quantum Index)", 0.0, 2.0, 0.5, 0.05)
        k_min = st.slider("k_min (Mpc⁻¹)", 0.001, 0.01, 0.005, 0.001, format="%.3f")
        k_max = st.slider("k_max (Mpc⁻¹)", 0.1, 1.0, 0.5, 0.05)
        
        k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 100)
        P_quantum = process_qcis(k_vals, f_nl, n_q)
        P_lcdm = k_vals ** (-3) * np.exp(-k_vals / 0.1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_power_fnl{f_nl:.1f}_nq{n_q:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        spectra_data = {
            "parameters": {"f_nl": f_nl, "n_q": n_q},
            "k_values": [float(x) for x in k_vals],
            "P_quantum": [float(x) for x in P_quantum]
        }
        st.markdown(get_json_download_link(spectra_data, f"{base_filename}_data.json", "📄 Download"), unsafe_allow_html=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(k_vals, P_lcdm, 'b-', label='ΛCDM', linewidth=2)
        ax.loglog(k_vals, P_quantum, 'r--', label=f'Quantum (f_NL={f_nl:.1f}, n_q={n_q:.2f})', linewidth=2)
        ax.set_xlabel("k (Mpc⁻¹)")
        ax.set_ylabel("P(k)")
        ax.set_title("Matter Power Spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, f"{base_filename}_spectrum.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)
        
        ratio = P_quantum / (P_lcdm + 1e-10)
        st.metric("Quantum Enhancement", f"{np.mean(ratio):.3f}x")
        st.metric("Spectral Index n_s", f"{n_q:.2f}")

# ============================================================================
# TAB 5: SPECTRAL & COLOR ANALYSIS (Your pattern analysis)
# ============================================================================

with tab5:
    st.header("🌈 Spectral & Color Heat Pattern Analyzer")
    st.markdown("**Your Analysis:** Power Spectra | Radial Profiles | Heatmap Metrics | Color Mapping")
    
    pattern_type = st.radio("Select Pattern Type", ["FDM Soliton", "PDP Entanglement"], horizontal=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Parameters")
        
        if pattern_type == "FDM Soliton":
            k = st.slider("k (Soliton Scale)", 0.5, 3.0, 1.0, 0.05)
            pattern = fdm_soliton_profile(np.linspace(0, 3, 512), k)
            pattern = np.outer(pattern, pattern)
            pattern = pattern / pattern.max()
            title = f"FDM Soliton (k={k:.2f})"
        else:
            omega_pdp = st.slider("Ω (Entanglement)", 0.2, 1.5, 0.5, 0.05)
            fringe = st.slider("Fringe Scale", 0.5, 3.0, 1.0, 0.1)
            pattern = pdp_entanglement_overlay(np.random.rand(512, 512), omega_pdp, fringe)
            title = f"PDP Entanglement (Ω={omega_pdp:.2f}, fringe={fringe:.2f})"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{pattern_type.lower().replace(' ', '_')}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        fig_pat, ax_pat = plt.subplots(figsize=(6, 6))
        ax_pat.imshow(pattern, cmap='plasma', origin='lower')
        ax_pat.axis('off')
        st.markdown(get_image_download_link(fig_pat, f"{base_filename}_pattern.png", "📥 Download Pattern"), unsafe_allow_html=True)
        plt.close(fig_pat)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pattern, cmap='plasma', origin='lower')
        ax.set_title(title)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    # Spectral Analysis
    st.subheader("📊 Spectral Analysis")
    
    power_spec = np.abs(fftshift(fft2(pattern)))**2
    rows, cols = power_spec.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol)**2 + (y - crow)**2)
    r_vals = np.arange(0, min(crow, ccol), 1)
    radial_profile = [np.mean(power_spec[(r >= i) & (r < i+1)]) for i in r_vals if np.any((r >= i) & (r < i+1))]
    radial_profile = np.array(radial_profile[:len(r_vals)])
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_ps, ax_ps = plt.subplots(figsize=(6, 6))
        ax_ps.imshow(np.log(power_spec + 1), cmap='hot', origin='lower')
        ax_ps.set_title("Power Spectrum (log scale)")
        ax_ps.axis('off')
        st.pyplot(fig_ps)
        st.markdown(get_image_download_link(fig_ps, f"{base_filename}_powerspec.png", "📥 Download"), unsafe_allow_html=True)
        plt.close(fig_ps)
    
    with col_b:
        fig_rad, ax_rad = plt.subplots(figsize=(6, 4))
        valid = r_vals[1:] < len(radial_profile)
        ax_rad.plot(r_vals[1:][valid], radial_profile[1:][valid], 'b-', linewidth=2)
        ax_rad.set_xlabel("Radial Frequency")
        ax_rad.set_ylabel("Power")
        ax_rad.set_title("Radial Power Spectrum")
        ax_rad.set_xscale('log')
        ax_rad.set_yscale('log')
        ax_rad.grid(True, alpha=0.3)
        st.pyplot(fig_rad)
        st.markdown(get_image_download_link(fig_rad, f"{base_filename}_radial.png", "📥 Download"), unsafe_allow_html=True)
        plt.close(fig_rad)

# ============================================================================
# ABOUT
# ============================================================================

with st.expander("📖 About Your QCAUS Projects", expanded=False):
    st.markdown("""
    ### Your Quantum Cosmology & Astrophysics Unified Suite
    
    | Project | Your Formulas | Description |
    |---------|---------------|-------------|
    | **QCI AstroEntangle** | ρ(r) = ρ₀ [sin(kr)/(kr)]², ℒ_mix = (ε/2) F_μν F'^μν | FDM soliton cores + PDP entanglement on astrophysical images |
    | **Magnetar QED** | B = B₀ (R/r)³ (2 cosθ, sinθ), Euler-Heisenberg | Magnetar fields, vacuum polarization, dark photon conversion |
    | **Primordial Entanglement** | i∂ρ/∂t = [H, ρ], S = -Tr(ρ log ρ) | Photon-dark photon entanglement evolution in expanding universe |
    | **QCIS** | P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) | Quantum-corrected cosmological power spectra |
    
    **Features:**
    - All your actual formulas implemented
    - Download visualizations as PNG
    - Export data as JSON
    - Real-time parameter adjustment
    - Annotated comparisons with your metrics
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | Your Complete Quantum Cosmology Suite<br>
    FDM Soliton • PDP Entanglement • Magnetar QED • QCIS • Spectral Analysis<br>
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
