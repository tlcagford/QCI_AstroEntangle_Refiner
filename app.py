"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Complete version with unique element IDs for all widgets
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
import json
from scipy.ndimage import gaussian_filter, zoom
from scipy.fft import fft2, ifft2, fftshift
from io import BytesIO
from PIL import Image
import tempfile
import os
import base64
from datetime import datetime

st.set_page_config(page_title="QCAUS - Quantum Cosmology Suite", page_icon="🌌", layout="wide")

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
# PROJECT 1: FDM SOLITON + PDP ENTANGLEMENT
# ============================================================================

def fdm_soliton_profile(r, k=1.0):
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)
    return profile

def pdp_entanglement_overlay(image_data, omega=0.5, fringe_scale=1.0):
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

def process_qci_astro(image_data, omega=0.5, fringe=1.0, soliton_scale=1.0):
    size = min(image_data.shape)
    r = np.linspace(0, 3, size)
    soliton_profile = fdm_soliton_profile(r, k=soliton_scale)
    soliton_2d = np.outer(soliton_profile, soliton_profile)
    soliton_resized = zoom(soliton_2d, (image_data.shape[0]/size, image_data.shape[1]/size))
    pdp = pdp_entanglement_overlay(image_data, omega, fringe)
    enhanced = image_data + 0.3 * soliton_resized + 0.5 * pdp
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced, soliton_resized, pdp

def add_scale_bar(ax, image_width_pixels, physical_width_kpc=100, pixel_scale_kpc=0.1):
    bar_length_pixels = physical_width_kpc / pixel_scale_kpc
    x_start = image_width_pixels - bar_length_pixels - 50
    y_start = 50
    rect = Rectangle((x_start, y_start), bar_length_pixels, 8,
                     linewidth=2, edgecolor='white', facecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x_start + bar_length_pixels/2, y_start + 25, f"{physical_width_kpc} kpc",
            color='white', fontsize=10, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

# ============================================================================
# PROJECT 2: MAGNETAR QED EXPLORER
# ============================================================================

def magnetar_dipole_field(r, theta, B0=1e15):
    B_r = B0 * 2 * np.cos(theta) / (r**3 + 1e-10)
    B_theta = B0 * np.sin(theta) / (r**3 + 1e-10)
    return B_r, B_theta

def quantum_vacuum_polarization(B, alpha=1/137):
    B_crit = 4.41e13
    beta = (B / B_crit)**2
    polarization = alpha * beta / (45 * np.pi) * (1 + 7/4 * beta)
    return polarization

def dark_photon_conversion(B, mixing_angle=0.1, mass=1e-9):
    conversion_prob = mixing_angle**2 * (1 - np.exp(-B**2 / mass**2))
    return conversion_prob

def process_magnetar(r_grid, theta_grid, B0=1e15, mixing=0.1, mass=1e-9):
    B_r, B_theta = magnetar_dipole_field(r_grid, theta_grid, B0)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    qed = quantum_vacuum_polarization(B_mag)
    dark_photons = dark_photon_conversion(B_mag, mixing_angle=mixing, mass=mass)
    return B_mag, qed, dark_photons

# ============================================================================
# PROJECT 3: PRIMORDIAL ENTANGLEMENT
# ============================================================================

def von_neumann_evolution(density_matrix, hamiltonian, dt):
    commutator = np.dot(hamiltonian, density_matrix) - np.dot(density_matrix, hamiltonian)
    return density_matrix + (-1j * commutator) * dt

def entanglement_entropy(reduced_density):
    eigenvalues = np.linalg.eigvalsh(reduced_density)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def process_primordial_entanglement(omega_val=0.7, dark_mass_val=1e-9, mixing_val=0.1, t_steps=100):
    rho = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
    H = np.array([[omega_val, mixing_val], [mixing_val, dark_mass_val]], dtype=complex)
    dt = 0.01
    entropy_evolution = []
    mixing_prob = []
    for step in range(t_steps):
        rho = von_neumann_evolution(rho, H, dt)
        reduced = rho[:1, :1]
        entropy_evolution.append(entanglement_entropy(reduced))
        mixing_prob.append(abs(rho[0, 1])**2)
    return entropy_evolution, mixing_prob

# ============================================================================
# PROJECT 4: QCIS POWER SPECTRA
# ============================================================================

def quantum_corrected_power_spectrum(k, f_nl_val=1.0, n_q_val=0.5, k0=0.05):
    P_lcdm = k ** (-3) * np.exp(-k / 0.1)
    quantum_correction = 1 + f_nl_val * (k / k0)**n_q_val
    return P_lcdm * quantum_correction

def process_qcis(k_vals, f_nl_val=1.0, n_q_val=0.5):
    return quantum_corrected_power_spectrum(k_vals, f_nl_val, n_q_val)

# ============================================================================
# PROJECT 5: SPECTRAL ANALYSIS
# ============================================================================

def compute_power_spectrum(image):
    fft_img = fft2(image)
    fft_shift = fftshift(fft_img)
    return np.abs(fft_shift)**2

def compute_radial_profile(power_spectrum):
    rows, cols = power_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol)**2 + (y - crow)**2)
    max_r = min(crow, ccol)
    bins = np.arange(0, max_r, 1)
    radial_profile = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    return bins[:-1], radial_profile

def generate_fdm_pattern(size=256, k_val=1.0):
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(r > 0, (np.sin(k_val * r) / (k_val * r))**2, 1.0)
    return pattern / pattern.max()

def generate_pdp_pattern(size=256, omega_val=0.5, fringe_val=1.0):
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    pattern = np.exp(-omega_val * R**2) * (1 - np.exp(-R**2 / fringe_val)) * np.sin(10 * R)
    pattern = pattern - pattern.min()
    pattern = pattern / pattern.max()
    return pattern

# ============================================================================
# SAMPLE ASTROPHYSICAL IMAGES
# ============================================================================

def get_sample_image(image_name):
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
        img = np.exp(-((X-0.8)**2 + Y**2) / 0.3**2) + 0.7 * np.exp(-((X+0.6)**2 + Y**2) / 0.4**2)
        return img / img.max()
    elif image_name == "Magnetar":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.sin(3 * theta))
        return img / img.max()
    else:
        return np.random.randn(size, size)

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
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    st.markdown("*FDM Soliton • PDP Entanglement • Magnetar QED • QCIS*")
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"], key="img_source")
    
    astro_image = None
    image_name = "galaxy_cluster"
    pixel_scale_kpc = 0.1
    
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Bullet Cluster", "Magnetar"], key="sample_img_type")
        astro_image = get_sample_image(image_type)
        image_name = image_type.replace(" ", "_").lower()
    else:
        uploaded_img = st.file_uploader("Upload Image", type=['fits', 'jpg', 'png'], key="img_upload")
        if uploaded_img:
            astro_image, info = load_image_file(uploaded_img)
            if astro_image is None:
                st.error(info)
            else:
                st.success(info)
                image_name = uploaded_img.name.split('.')[0]
                pixel_scale_kpc = 100.0 / astro_image.shape[1]
    
    st.header("📏 Scale")
    pixel_scale_kpc = st.number_input("kpc/pixel", value=pixel_scale_kpc, format="%.4f", key="scale_input")

# ============================================================================
# MAIN CONTENT - 5 TABS
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("**FDM Soliton • PDP Entanglement • Magnetar QED • Primordial Entanglement • QCIS**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔭 QCI AstroEntangle",
    "⚡ Magnetar QED Explorer",
    "🌀 Primordial Entanglement",
    "📊 QCIS Power Spectra",
    "🌈 Spectral Analysis"
])

# ============================================================================
# TAB 1: QCI ASTROENTANGLE
# ============================================================================

with tab1:
    st.header("🔭 QCI AstroEntangle Refiner")
    st.markdown("**FDM Soliton:** ρ(r) = ρ₀ [sin(kr)/(kr)]² | **PDP:** ℒ_mix = (ε/2) F_μν F'^μν")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        omega1 = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01, key="omega_tab1")
        fringe1 = st.slider("Fringe Scale", 0.1, 3.0, 1.0, 0.1, key="fringe_tab1")
        soliton_scale1 = st.slider("FDM Soliton Scale (k)", 0.5, 3.0, 1.0, 0.1, key="soliton_tab1")
        
        if astro_image is not None:
            enhanced, soliton, pdp = process_qci_astro(astro_image, omega1, fringe1, soliton_scale1)
            
            st.markdown("---")
            st.subheader("📊 Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Max Mixing", f"{np.max(pdp):.3f}")
            col_m2.metric("Min Entropy", f"{np.min(soliton):.3f}")
            col_m3.metric("FDM Value", f"{soliton_scale1 * 2.5:.1f} kpc")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"qcaus_{image_name}_omega{omega1:.2f}_{timestamp}"
            
            st.markdown("---")
            st.subheader("📥 Download")
            
            fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(astro_image, cmap='gray', origin='lower')
            ax1.set_title("Original")
            ax1.axis('off')
            add_scale_bar(ax1, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            
            ax2.imshow(enhanced, cmap='gray', origin='lower')
            ax2.set_title("FDM + PDP Enhanced")
            ax2.axis('off')
            add_scale_bar(ax2, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            
            plt.tight_layout()
            st.pyplot(fig_comp)
            st.markdown(get_image_download_link(fig_comp, f"{base_filename}_comparison.png", "📸 Download"), unsafe_allow_html=True)
            plt.close(fig_comp)
    
    with col2:
        if astro_image is not None:
            enhanced, soliton, pdp = process_qci_astro(astro_image, omega1, fringe1, soliton_scale1)
            
            fig_rgb, ax_rgb = plt.subplots(figsize=(8, 8))
            rgb = np.zeros((*astro_image.shape, 3))
            rgb[..., 0] = astro_image / (astro_image.max() + 1e-8)
            rgb[..., 1] = (soliton - soliton.min()) / (soliton.max() - soliton.min() + 1e-8) * 0.8
            rgb[..., 2] = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8) * 0.8
            ax_rgb.imshow(np.clip(rgb, 0, 1), origin='lower')
            ax_rgb.set_title("Radar-Style Overlay\nGreen: FDM | Blue: PDP")
            ax_rgb.axis('off')
            add_scale_bar(ax_rgb, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.pyplot(fig_rgb)
            plt.close(fig_rgb)
        else:
            st.info("👈 Select or upload an image")

# ============================================================================
# TAB 2: MAGNETAR QED EXPLORER
# ============================================================================

with tab2:
    st.header("⚡ Magnetar QED Explorer")
    st.markdown("**Magnetar Field:** B = B₀ (R/r)³ (2 cosθ, sinθ) | **Vacuum Polarization:** Euler-Heisenberg | **Dark Photons:** P = ε² (1 - e^{-B²/m²})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        B0 = st.slider("Surface B-Field (10¹⁵ G)", 0.5, 5.0, 1.0, 0.1, key="B0_tab2")
        mixing_angle2 = st.slider("Dark Photon Mixing ε", 0.0, 0.5, 0.1, 0.01, key="mixing_tab2")
        dark_mass2 = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e", key="darkmass_tab2")
        
        r = np.linspace(1, 10, 200)
        theta = np.linspace(0, np.pi, 200)
        R, Theta = np.meshgrid(r, theta)
        B_mag, qed, dark_photons = process_magnetar(R, Theta, B0=B0*1e15, mixing=mixing_angle2, mass=dark_mass2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_magnetar_B{B0:.1f}_eps{mixing_angle2:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Physics Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Max B-Field", f"{np.max(B_mag)/1e15:.2f}×10¹⁵ G")
        col_m2.metric("Max Polarization", f"{np.max(qed):.3e}")
        col_m3.metric("Max Dark Photons", f"{np.max(dark_photons):.3f}")
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        ax_b.imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
        ax_b.set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G")
        plt.colorbar(ax_b.images[0], ax=ax_b)
        st.markdown(get_image_download_link(fig_b, f"{base_filename}_bfield.png", "🔴 Download"), unsafe_allow_html=True)
        plt.close(fig_b)
    
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        im1 = axes[0].imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
        axes[0].set_title(f"B-Field\n{B0:.1f}×10¹⁵ G")
        axes[0].set_xlabel("Radius (R/R₀)")
        axes[0].set_ylabel("Angle (deg)")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma')
        axes[1].set_title("Vacuum Polarization\nEuler-Heisenberg")
        axes[1].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(dark_photons, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis')
        axes[2].set_title(f"Dark Photons\nε={mixing_angle2:.2f}, m={dark_mass2:.1e}eV")
        axes[2].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, f"{base_filename}_magnetar.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# ============================================================================
# TAB 3: PRIMORDIAL ENTANGLEMENT (FIXED DUPLICATE KEYS)
# ============================================================================

with tab3:
    st.header("🌀 Primordial Photon-DarkPhoton Entanglement")
    st.markdown("**Von Neumann:** i∂ρ/∂t = [H, ρ] | **Entropy:** S = -Tr(ρ log ρ)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        omega_ent = st.slider("Ω (Oscillation)", 0.0, 2.0, 0.7, 0.01, key="omega_tab3")
        dark_mass3 = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e", key="darkmass_tab3")
        mixing_prim = st.slider("Mixing Angle ε", 0.0, 0.5, 0.1, 0.01, key="mixing_tab3")
        t_steps = st.slider("Time Steps", 50, 500, 100, key="tsteps_tab3")
        
        entropy, mixing_prob = process_primordial_entanglement(omega_ent, dark_mass3, mixing_prim, t_steps)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_entanglement_omega{omega_ent:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Final Entropy", f"{entropy[-1]:.4f}")
        col_m2.metric("Final Mixing", f"{mixing_prob[-1]:.4f}")
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        entanglement_data = {
            "parameters": {"omega": omega_ent, "dark_mass": dark_mass3, "mixing": mixing_prim},
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

# ============================================================================
# TAB 4: QCIS POWER SPECTRA
# ============================================================================

with tab4:
    st.header("📊 QCIS - Quantum Cosmology Integration Suite")
    st.markdown("**Power Spectrum:** P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f_nl = st.slider("f_NL (Non-Gaussianity)", 0.0, 5.0, 1.0, 0.1, key="fnl_tab4")
        n_q = st.slider("n_q (Quantum Index)", 0.0, 2.0, 0.5, 0.05, key="nq_tab4")
        k_min = st.slider("k_min (Mpc⁻¹)", 0.001, 0.01, 0.005, 0.001, format="%.3f", key="kmin_tab4")
        k_max = st.slider("k_max (Mpc⁻¹)", 0.1, 1.0, 0.5, 0.05, key="kmax_tab4")
        
        k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 100)
        P_quantum = process_qcis(k_vals, f_nl, n_q)
        P_lcdm = k_vals ** (-3) * np.exp(-k_vals / 0.1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_power_fnl{f_nl:.1f}_nq{n_q:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        ratio = P_quantum / (P_lcdm + 1e-10)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Quantum Enhancement", f"{np.mean(ratio):.3f}x")
        col_m2.metric("Spectral Index", f"{n_q:.2f}")
        
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

# ============================================================================
# TAB 5: SPECTRAL ANALYSIS
# ============================================================================

with tab5:
    st.header("🌈 Spectral & Color Heat Pattern Analyzer")
    
    pattern_type = st.radio("Pattern Type", ["FDM Soliton", "PDP Entanglement"], horizontal=True, key="pattern_type")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pattern_type == "FDM Soliton":
            k_val = st.slider("k (Soliton Scale)", 0.5, 3.0, 1.0, 0.05, key="k_tab5")
            pattern = generate_fdm_pattern(size=512, k_val=k_val)
            title = f"FDM Soliton (k={k_val:.2f})"
        else:
            omega_pdp = st.slider("Ω (Entanglement)", 0.2, 1.5, 0.5, 0.05, key="omega_tab5")
            fringe_val = st.slider("Fringe Scale", 0.5, 3.0, 1.0, 0.1, key="fringe_tab5")
            pattern = generate_pdp_pattern(size=512, omega_val=omega_pdp, fringe_val=fringe_val)
            title = f"PDP Entanglement (Ω={omega_pdp:.2f})"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{pattern_type.lower().replace(' ', '_')}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Mean", f"{np.mean(pattern):.3f}")
        col_m2.metric("Std Dev", f"{np.std(pattern):.3f}")
        col_m3.metric("Max", f"{np.max(pattern):.3f}")
        
        st.markdown("---")
        st.subheader("📥 Download")
        
        fig_pat, ax_pat = plt.subplots(figsize=(6, 6))
        ax_pat.imshow(pattern, cmap='plasma', origin='lower')
        ax_pat.axis('off')
        st.markdown(get_image_download_link(fig_pat, f"{base_filename}_pattern.png", "📥 Download"), unsafe_allow_html=True)
        plt.close(fig_pat)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pattern, cmap='plasma', origin='lower')
        ax.set_title(title)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    # Spectral Analysis
    st.subheader("📊 Power Spectrum Analysis")
    
    power_spec = compute_power_spectrum(pattern)
    r_vals, radial_profile = compute_radial_profile(power_spec)
    
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

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite
    
    | Project | Formula | Description |
    |---------|---------|-------------|
    | **QCI AstroEntangle** | ρ(r) = ρ₀ [sin(kr)/(kr)]², ℒ_mix = (ε/2) F_μν F'^μν | FDM soliton + PDP entanglement on images |
    | **Magnetar QED** | B = B₀ (R/r)³ (2 cosθ, sinθ), Euler-Heisenberg | Magnetar fields, vacuum polarization, dark photons |
    | **Primordial Entanglement** | i∂ρ/∂t = [H, ρ], S = -Tr(ρ log ρ) | Photon-dark photon entanglement evolution |
    | **QCIS** | P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) | Quantum-corrected power spectra |
    
    **Supported Formats:** FITS, JPEG, PNG
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | FDM Soliton • PDP Entanglement • Magnetar QED • QCIS<br>
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
