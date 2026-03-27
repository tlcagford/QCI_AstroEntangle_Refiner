"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Complete version with all 5 projects + Spectral Analysis
- QCI AstroEntangle Refiner (FDM Soliton + PDP Entanglement)
- Magnetar QED Explorer
- Primordial Photon-DarkPhoton Entanglement
- QCIS Power Spectra
- Spectral & Color Analysis (NEW)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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
    """Generate download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_json_download_link(data, filename, title="Download"):
    """Generate download link for JSON data"""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_array_download_link(data, filename, title="Download"):
    """Generate download link for numpy array"""
    buf = BytesIO()
    np.savez_compressed(buf, data=data)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

# ============================================================================
# UNIVERSAL IMAGE LOADER
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
# PROJECT 1: QCI AstroEntangle Refiner
# ============================================================================

def fdm_soliton_profile(r, k=1.0):
    """Fuzzy Dark Matter soliton profile: ρ(r) ∝ [sin(kr)/(kr)]²"""
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)
    return profile

def pdp_entanglement_overlay(image_data, omega=0.5, fringe_scale=1.0):
    """Photon-DarkPhoton entanglement filter for image enhancement"""
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
    """Process QCI AstroEntangle Refiner with FDM soliton and PDP"""
    size = min(image_data.shape)
    r = np.linspace(0, 3, size)
    soliton_profile = fdm_soliton_profile(r, k=soliton_scale)
    soliton_2d = np.outer(soliton_profile, soliton_profile)
    soliton_resized = zoom(soliton_2d, (image_data.shape[0]/size, image_data.shape[1]/size))
    pdp = pdp_entanglement_overlay(image_data, omega, fringe)
    enhanced = image_data + 0.3 * soliton_resized + 0.5 * pdp
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced, soliton_resized, pdp

# ============================================================================
# PROJECT 2: Magnetar QED Explorer
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

def process_magnetar(r_grid, theta_grid, B0=1e15, mixing=0.1):
    B_r, B_theta = magnetar_dipole_field(r_grid, theta_grid, B0)
    B_mag = np.sqrt(B_r**2 + B_theta**2)
    qed = quantum_vacuum_polarization(B_mag)
    dark_photons = dark_photon_conversion(B_mag, mixing_angle=mixing)
    return B_mag, qed, dark_photons

# ============================================================================
# PROJECT 3: Primordial Photon-DarkPhoton Entanglement
# ============================================================================

def von_neumann_evolution(density_matrix, hamiltonian, dt):
    commutator = np.dot(hamiltonian, density_matrix) - np.dot(density_matrix, hamiltonian)
    return density_matrix + (-1j * commutator) * dt

def entanglement_entropy(reduced_density):
    eigenvalues = np.linalg.eigvalsh(reduced_density)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def process_primordial_entanglement(omega=0.7, dark_mass=1e-9, mixing=0.1, t_steps=100):
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

# ============================================================================
# PROJECT 4: QCIS - Quantum Cosmology Integration Suite
# ============================================================================

def quantum_corrected_power_spectrum(k, f_nl=1.0, n_q=0.5, k0=0.05):
    P_lcdm = k ** (-3) * np.exp(-k / 0.1)
    quantum_correction = 1 + f_nl * (k / k0)**n_q
    return P_lcdm * quantum_correction

def process_qcis(k_vals, f_nl=1.0, n_q=0.5):
    return quantum_corrected_power_spectrum(k_vals, f_nl, n_q)

# ============================================================================
# PROJECT 5: SPECTRAL & COLOR ANALYSIS
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

def compute_angular_variance(power_spectrum):
    """Compute angular variance around each radius"""
    rows, cols = power_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol)**2 + (y - crow)**2)
    theta = np.arctan2(y - crow, x - ccol)
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
        'skew': float(np.mean(((image - np.mean(image)) / (np.std(image) + 1e-10))**3)),
        'kurtosis': float(np.mean(((image - np.mean(image)) / (np.std(image) + 1e-10))**4) - 3),
        'entropy': float(-np.sum(image * np.log(image + 1e-10)))
    }

def generate_fdm_soliton_pattern(size=256, k=1.0):
    """Generate FDM Soliton pattern"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        pattern = np.where(r > 0, (np.sin(k * r) / (k * r))**2, 1.0)
    return pattern / pattern.max()

def generate_pdp_entanglement_pattern(size=256, omega=0.5, fringe=1.0):
    """Generate PDP Entanglement pattern"""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    pattern = np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe)) * np.sin(10 * R)
    pattern = pattern - pattern.min()
    pattern = pattern / pattern.max()
    return pattern

def generate_interference_pattern(size=256, k1=1.0, k2=1.5, theta=30):
    """Generate quantum interference pattern"""
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    r1 = np.sqrt(X**2 + Y**2)
    theta_rad = np.radians(theta)
    X2 = X * np.cos(theta_rad) - Y * np.sin(theta_rad)
    Y2 = X * np.sin(theta_rad) + Y * np.cos(theta_rad)
    r2 = np.sqrt(X2**2 + Y2**2)
    pattern1 = np.where(r1 > 0, (np.sin(k1 * r1) / (k1 * r1))**2, 1.0)
    pattern2 = np.where(r2 > 0, (np.sin(k2 * r2) / (k2 * r2))**2, 1.0)
    interference = (pattern1 + pattern2) / 2
    return interference / interference.max()

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
    elif image_name == "Nebula":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.cos(5 * theta))
        return img / img.max()
    elif image_name == "Bullet Cluster":
        img = np.exp(-((X-0.8)**2 + Y**2) / 0.3**2) + 0.7 * np.exp(-((X+0.6)**2 + Y**2) / 0.4**2)
        return img / img.max()
    else:
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.5 * np.cos(10 * R + 3 * theta))
        return img / img.max()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_name = "galaxy_cluster"
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Nebula", "Bullet Cluster"])
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
# MAIN CONTENT - 5 TABS
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("FDM Soliton • PDP Entanglement • Magnetar QED • QCIS • Spectral Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔭 QCI AstroEntangle",
    "⚡ Magnetar QED",
    "🌀 Primordial Entanglement",
    "📊 QCIS Power Spectra",
    "🌈 Spectral & Color Analysis"
])

# ============================================================================
# TAB 1: QCI AstroEntangle Refiner
# ============================================================================

with tab1:
    st.header("🔭 QCI AstroEntangle Refiner")
    st.markdown("FDM Soliton Physics + Photon-DarkPhoton Entanglement Overlay")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Controls")
        omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01, key="qci_omega")
        fringe = st.slider("Fringe Scale", 0.1, 3.0, 1.0, 0.1, key="qci_fringe")
        soliton_scale = st.slider("FDM Soliton Scale", 0.5, 3.0, 1.0, 0.1)
        
        if astro_image is not None:
            enhanced, soliton, pdp = process_qci_astro(astro_image, omega, fringe, soliton_scale)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"qcaus_{image_name}_omega{omega:.2f}_fringe{fringe:.2f}_{timestamp}"
            
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            # Individual downloads
            fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
            ax_orig.imshow(astro_image, cmap='gray', origin='lower')
            ax_orig.axis('off')
            st.markdown(get_image_download_link(fig_orig, f"{base_filename}_original.png", "📷 Download Original"), unsafe_allow_html=True)
            plt.close(fig_orig)
            
            fig_enh, ax_enh = plt.subplots(figsize=(6, 6))
            ax_enh.imshow(enhanced, cmap='gray', origin='lower')
            ax_enh.axis('off')
            st.markdown(get_image_download_link(fig_enh, f"{base_filename}_enhanced.png", "✨ Download Enhanced"), unsafe_allow_html=True)
            plt.close(fig_enh)
    
    with col2:
        st.subheader("Before / After Comparison")
        if astro_image is not None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes[0, 0].imshow(astro_image, cmap='gray', origin='lower')
            axes[0, 0].set_title("Original")
            axes[0, 0].axis('off')
            axes[0, 1].imshow(soliton, cmap='viridis', origin='lower')
            axes[0, 1].set_title(f"FDM Soliton\nk={soliton_scale:.2f}")
            axes[0, 1].axis('off')
            axes[1, 0].imshow(pdp, cmap='plasma', origin='lower')
            axes[1, 0].set_title(f"PDP Entanglement\nΩ={omega:.2f}")
            axes[1, 0].axis('off')
            axes[1, 1].imshow(enhanced, cmap='gray', origin='lower')
            axes[1, 1].set_title("Enhanced")
            axes[1, 1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, f"{base_filename}_comparison.png", "📸 Download Comparison"), unsafe_allow_html=True)
            plt.close(fig)
        else:
            st.info("👈 Select or upload an image")

# ============================================================================
# TAB 2: Magnetar QED Explorer
# ============================================================================

with tab2:
    st.header("⚡ Magnetar QED Explorer")
    st.markdown("Magnetar Fields • Quantum Vacuum Polarization • Dark Photon Conversion")
    
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
        st.subheader("📥 Download Results")
        
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        im_b = ax_b.imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
        plt.colorbar(im_b, ax=ax_b)
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
# TAB 3: Primordial Photon-DarkPhoton Entanglement
# ============================================================================

with tab3:
    st.header("🌀 Primordial Photon-DarkPhoton Entanglement")
    st.markdown("Von Neumann Evolution • Entanglement Entropy • Mixing Probability")
    
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
        st.subheader("📥 Download Results")
        
        entanglement_data = {
            "parameters": {"omega": omega_ent, "dark_mass": dark_mass, "mixing": mixing_prim},
            "final_entropy": float(entropy[-1]),
            "final_mixing": float(mixing_prob[-1])
        }
        st.markdown(get_json_download_link(entanglement_data, f"{base_filename}_data.json", "📄 Download Data"), unsafe_allow_html=True)
    
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
# TAB 4: QCIS Power Spectra
# ============================================================================

with tab4:
    st.header("📊 QCIS - Quantum Cosmology Integration Suite")
    st.markdown("Quantum-Corrected Power Spectra • ΛCDM vs Quantum")
    
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
        st.subheader("📥 Download Results")
        
        spectra_data = {
            "parameters": {"f_nl": f_nl, "n_q": n_q},
            "k_values": [float(x) for x in k_vals],
            "P_quantum": [float(x) for x in P_quantum]
        }
        st.markdown(get_json_download_link(spectra_data, f"{base_filename}_data.json", "📄 Download Data"), unsafe_allow_html=True)
    
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
        st.metric("Spectral Index", f"{n_q:.2f}")

# ============================================================================
# TAB 5: Spectral & Color Analysis
# ============================================================================

with tab5:
    st.header("🌈 Spectral & Color Heat Pattern Analyzer")
    st.markdown("Analyzing quantum interference patterns, power spectra, and thermal color mappings")
    
    pattern_type = st.radio(
        "Select Pattern Type",
        ["FDM Soliton", "PDP Entanglement", "Quantum Interference"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Pattern Parameters")
        
        if pattern_type == "FDM Soliton":
            k = st.slider("k (Soliton Scale)", 0.5, 3.0, 1.0, 0.05)
            pattern = generate_fdm_soliton_pattern(size=512, k=k)
            title = f"FDM Soliton (k={k:.2f})"
            
            st.markdown("""
            **FDM Soliton Formula:**
            $$\\rho(r) = \\rho_0 \\left[\\frac{\\sin(kr)}{kr}\\right]^2$$
            
            **Interpretation:** Ground state of ultra-light dark matter
            - Central core with quantum interference ripples
            - **k** controls soliton compactness
            """)
            
        elif pattern_type == "PDP Entanglement":
            omega_pdp = st.slider("Ω (Entanglement Strength)", 0.2, 1.5, 0.5, 0.05)
            fringe = st.slider("Fringe Scale", 0.5, 3.0, 1.0, 0.1)
            pattern = generate_pdp_entanglement_pattern(size=512, omega=omega_pdp, fringe=fringe)
            title = f"PDP Entanglement (Ω={omega_pdp:.2f}, fringe={fringe:.2f})"
            
            st.markdown("""
            **PDP Entanglement Formula:**
            $$\\mathcal{L}_{\\text{mix}} = \\frac{\\varepsilon}{2} F_{\\mu\\nu} F'^{\\mu\\nu}$$
            
            **Interpretation:** Dark photon mixing signatures
            - **Ω** controls entanglement strength
            - **Fringe** sets oscillation wavelength
            """)
            
        else:
            k1 = st.slider("k₁ (First Source)", 0.5, 2.5, 1.0, 0.05)
            k2 = st.slider("k₂ (Second Source)", 0.5, 2.5, 1.5, 0.05)
            theta = st.slider("Separation Angle (°)", 0, 90, 30, 5)
            pattern = generate_interference_pattern(size=512, k1=k1, k2=k2, theta=theta)
            title = f"Quantum Interference (k₁={k1:.2f}, k₂={k2:.2f}, θ={theta}°)"
            
            st.markdown("""
            **Quantum Interference Formula:**
            $$I(r) = \\left[\\frac{\\sin(k_1 r_1)}{k_1 r_1}\\right]^2 + \\left[\\frac{\\sin(k_2 r_2)}{k_2 r_2}\\right]^2$$
            
            **Interpretation:** Interference between two quantum sources
            """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pattern, cmap='plasma', origin='lower')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label="Intensity")
        st.pyplot(fig)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qcaus_{pattern_type.lower().replace(' ', '_')}_{timestamp}.png"
        st.markdown(get_image_download_link(fig, filename, "📥 Download Pattern"), unsafe_allow_html=True)
        plt.close(fig)
    
    # Spectral Analysis
    st.subheader("📊 Spectral Analysis")
    
    power_spec = compute_power_spectrum(pattern)
    r_vals, radial_profile = compute_radial_profile(power_spec)
    r_centers, angular_variance = compute_angular_variance(power_spec)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_ps, ax_ps = plt.subplots(figsize=(6, 6))
        ax_ps.imshow(np.log(power_spec + 1), cmap='hot', origin='lower')
        ax_ps.set_title("Power Spectrum (log scale)")
        ax_ps.axis('off')
        st.pyplot(fig_ps)
        st.markdown(get_image_download_link(fig_ps, f"qcaus_{pattern_type.lower().replace(' ', '_')}_powerspec_{timestamp}.png", "📥 Download"), unsafe_allow_html=True)
        plt.close(fig_ps)
    
    with col_b:
        fig_rad, ax_rad = plt.subplots(figsize=(6, 4))
        ax_rad.plot(r_vals[1:], radial_profile[1:], 'b-', linewidth=2)
        ax_rad.set_xlabel("Radial Frequency")
        ax_rad.set_ylabel("Power")
        ax_rad.set_title("Radial Power Spectrum")
        ax_rad.set_xscale('log')
        ax_rad.set_yscale('log')
        ax_rad.grid(True, alpha=0.3)
        st.pyplot(fig_rad)
        st.markdown(get_image_download_link(fig_rad, f"qcaus_{pattern_type.lower().replace(' ', '_')}_radial_{timestamp}.png", "📥 Download"), unsafe_allow_html=True)
        plt.close(fig_rad)
    
    # Heatmap Metrics
    st.subheader("🔥 Heatmap Metrics")
    
    metrics = compute_heatmap_metrics(pattern)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Mean", f"{metrics['mean']:.4f}")
    col_m2.metric("Std Dev", f"{metrics['std']:.4f}")
    col_m3.metric("Max", f"{metrics['max']:.4f}")
    col_m4.metric("Min", f"{metrics['min']:.4f}")
    
    col_m5, col_m6, col_m7 = st.columns(3)
    col_m5.metric("Skewness", f"{metrics['skew']:.3f}")
    col_m6.metric("Kurtosis", f"{metrics['kurtosis']:.3f}")
    col_m7.metric("Entropy", f"{metrics['entropy']:.3f}")
    
    # Color Map Comparison
    st.subheader("🎨 Color Map Comparison")
    
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'coolwarm', 'seismic', 'RdYlBu_r']
    fig_cmap, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, cmap_name in enumerate(cmaps):
        ax = axes[i]
        im = ax.imshow(pattern, cmap=cmap_name, origin='lower')
        ax.set_title(cmap_name, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    st.pyplot(fig_cmap)
    st.markdown(get_image_download_link(fig_cmap, f"qcaus_{pattern_type.lower().replace(' ', '_')}_colormaps_{timestamp}.png", "📥 Download"), unsafe_allow_html=True)
    plt.close(fig_cmap)
    
    # Intensity Histogram
    st.subheader("📊 Intensity Distribution")
    
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(pattern.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel("Intensity")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Pixel Intensity Distribution")
    ax_hist.grid(True, alpha=0.3)
    st.pyplot(fig_hist)
    st.markdown(get_image_download_link(fig_hist, f"qcaus_{pattern_type.lower().replace(' ', '_')}_histogram_{timestamp}.png", "📥 Download"), unsafe_allow_html=True)
    plt.close(fig_hist)

# ============================================================================
# ABOUT
# ============================================================================

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite
    
    **Five Integrated Projects:**
    
    1. **QCI AstroEntangle Refiner** - FDM soliton physics + PDP entanglement on astronomical images
    2. **Magnetar QED Explorer** - Strong-field QED, dark photon conversion
    3. **Primordial Entanglement** - Von Neumann evolution of photon-dark photon system
    4. **QCIS** - Quantum-corrected cosmological power spectra
    5. **Spectral & Color Analysis** - Analyze patterns, power spectra, and color mappings
    
    **Supported Formats:** FITS, JPEG, PNG
    
    **Features:**
    - Download all visualizations as PNG
    - Export data as JSON
    - Real-time parameter adjustment
    - Color map comparison
    - Spectral analysis with power spectra
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | FDM Soliton • PDP Entanglement • Magnetar QED • QCIS • Spectral Analysis<br>
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
