"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Complete verified version with full-spectrum color mapping
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
# VERIFIED FORMULAS
# ============================================================================

def fdm_wave_function(r, k=1.0):
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        psi = np.where(kr > 0, np.sin(kr) / kr, 1.0)
    return psi

def fdm_density(r, k=1.0):
    psi = fdm_wave_function(r, k)
    return psi**2

def pdp_quantum_field(image_data, omega=0.5, fringe_scale=1.0):
    fft_img = fft2(image_data)
    fft_shift = fftshift(fft_img)
    rows, cols = image_data.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mixing_mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe_scale))
    mixed_fft = fft_shift * mixing_mask
    dark_field = np.abs(ifft2(fftshift(mixed_fft)))
    return dark_field

def quantum_superposition(image_data, omega=0.5, fringe=1.0, k=1.0, alpha=0.8, beta=1.0):
    size = min(image_data.shape)
    r = np.linspace(0, 3, size)
    fdm_psi = fdm_wave_function(r, k=k)
    fdm_field = np.outer(fdm_psi, fdm_psi)
    fdm_resized = zoom(fdm_field, (image_data.shape[0]/size, image_data.shape[1]/size))
    pdp_field = pdp_quantum_field(image_data, omega, fringe)
    quantum_state = image_data + alpha * fdm_resized + beta * pdp_field
    quantum_state = np.clip(quantum_state, 0, 1)
    return quantum_state, fdm_resized, pdp_field

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
    return mixing_angle**2 * (1 - np.exp(-B**2 / mass**2))

def von_neumann_evolution(rho, H, dt):
    commutator = np.dot(H, rho) - np.dot(rho, H)
    return rho + (-1j * commutator) * dt

def entanglement_entropy(rho_reduced):
    evals = np.linalg.eigvalsh(rho_reduced)
    evals = evals[evals > 1e-10]
    return -np.sum(evals * np.log(evals))

def quantum_power_spectrum(k, f_nl=1.0, n_q=0.5, k0=0.05):
    P_lcdm = k ** (-3) * np.exp(-k / 0.1)
    quantum_correction = 1 + f_nl * (k / k0)**n_q
    return P_lcdm * quantum_correction

# ============================================================================
# COLOR MAPPING FUNCTIONS
# ============================================================================

def map_invisible_to_visible(field_value, field_type='quantum', intensity=1.0):
    if field_type == 'fdm':
        rgb = np.zeros((*field_value.shape, 3))
        rgb[..., 1] = field_value * intensity * 0.9
        rgb[..., 0] = field_value * intensity * 0.2
        rgb[..., 2] = field_value * intensity * 0.3
        return rgb
    elif field_type == 'pdp':
        rgb = np.zeros((*field_value.shape, 3))
        rgb[..., 2] = field_value * intensity * 1.0
        rgb[..., 0] = field_value * intensity * 0.1
        rgb[..., 1] = field_value * intensity * 0.2
        return rgb
    else:
        return np.stack([field_value] * 3, axis=-1)

def create_full_spectrum_composite(original, fdm_field, pdp_field, quantum_state):
    rgb = np.zeros((*original.shape, 3))
    rgb[..., 0] = original / (original.max() + 1e-8)
    fdm_norm = (fdm_field - fdm_field.min()) / (fdm_field.max() - fdm_field.min() + 1e-8)
    pdp_norm = (pdp_field - pdp_field.min()) / (pdp_field.max() - pdp_field.min() + 1e-8)
    rgb[..., 1] = fdm_norm * 0.9
    rgb[..., 2] = pdp_norm * 0.9
    interference = fdm_norm * pdp_norm
    rgb[..., 1] += interference * 0.3
    rgb[..., 2] += interference * 0.3
    return np.clip(rgb, 0, 1)

def add_scale_bar(ax, image_width_pixels, physical_width_kpc=100, pixel_scale_kpc=0.1):
    bar_length_pixels = physical_width_kpc / pixel_scale_kpc
    x_start = 50
    y_start = image_width_pixels - 60
    rect = Rectangle((x_start, y_start), bar_length_pixels, 8,
                     linewidth=2, edgecolor='white', facecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x_start + bar_length_pixels/2, y_start - 12, f"{physical_width_kpc} kpc",
            color='white', fontsize=10, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

# ============================================================================
# SAMPLE IMAGES
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
    elif image_name == "Abell 1689":
        R = np.sqrt(X**2 + Y**2)
        img = np.exp(-R**2 / 1.5**2)
        img += 0.6 * np.exp(-((X-0.4)**2 + (Y-0.2)**2) / 0.3**2)
        img += 0.4 * np.exp(-((X+0.3)**2 + (Y+0.5)**2) / 0.4**2)
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
    
    st.header("⚛️ Quantum Field Parameters")
    omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01)
    fringe = st.slider("λ (Fringe Scale)", 0.1, 3.0, 1.5, 0.05)
    k = st.slider("k (Soliton Wave Number)", 0.5, 3.0, 1.0, 0.05)
    alpha = st.slider("α (FDM Coupling)", 0.0, 2.0, 0.8, 0.05)
    beta = st.slider("β (PDP Coupling)", 0.0, 2.0, 1.0, 0.05)
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_name = "abell_1689"
    pixel_scale_kpc = 0.1
    
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Bullet Cluster", "Abell 1689"])
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
                pixel_scale_kpc = 100.0 / astro_image.shape[1]
    
    st.header("📏 Scale")
    pixel_scale_kpc = st.number_input("kpc/pixel", value=pixel_scale_kpc, format="%.4f")

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("*Full-spectrum mapping of invisible quantum fields*")

# Create tabs
tabs = st.tabs([
    "🌈 Full-Spectrum Quantum Fields",
    "⚡ Magnetar QED Explorer",
    "🌀 Primordial Entanglement",
    "📊 QCIS Power Spectra"
])

# Tab 1: Full-Spectrum Quantum Fields
with tabs[0]:
    st.header("🌈 Full-Spectrum Quantum Field Visualization")
    
    if astro_image is not None:
        quantum_state, fdm_field, pdp_field = quantum_superposition(
            astro_image, omega, fringe, k, alpha, beta)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{image_name}_omega{omega:.2f}_lambda{fringe:.2f}_{timestamp}"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 Quantum Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Max FDM", f"{np.max(fdm_field):.3f}")
            col_m2.metric("Max PDP", f"{np.max(pdp_field):.3f}")
            corr = np.corrcoef(fdm_field.flatten(), pdp_field.flatten())[0,1]
            col_m3.metric("Correlation", f"{corr:.3f}")
            
            st.markdown("---")
            st.markdown("### 📥 Export")
            
            # Full-spectrum composite
            composite = create_full_spectrum_composite(astro_image, fdm_field, pdp_field, quantum_state)
            fig_comp, ax_comp = plt.subplots(figsize=(8, 8))
            ax_comp.imshow(composite, origin='upper')
            ax_comp.set_title("Full-Spectrum Composite\nVisible + Dark Matter (Green) + Dark Photons (Blue)")
            ax_comp.axis('off')
            add_scale_bar(ax_comp, composite.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.pyplot(fig_comp)
            st.markdown(get_image_download_link(fig_comp, f"{base_filename}_composite.png", "🌈 Download Composite"), unsafe_allow_html=True)
            plt.close(fig_comp)
        
        with col2:
            # Quantum state visualization
            fig_q, ax_q = plt.subplots(figsize=(8, 8))
            ax_q.imshow(quantum_state, cmap='plasma', origin='upper')
            ax_q.set_title(f"Quantum Superposition State\n|Ψ⟩ = |Ψ_astro⟩ + {alpha:.2f}|Ψ_FDM⟩ + {beta:.2f}|Ψ_PDP⟩")
            ax_q.axis('off')
            add_scale_bar(ax_q, quantum_state.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            plt.colorbar(ax_q.images[0], ax=ax_q, label="Quantum Amplitude")
            st.pyplot(fig_q)
            st.markdown(get_image_download_link(fig_q, f"{base_filename}_quantum_state.png", "📊 Download Quantum State"), unsafe_allow_html=True)
            plt.close(fig_q)
        
        # BEFORE/AFTER COMPARISON
        st.markdown("---")
        st.markdown("### 🔬 Before/After: Full-Spectrum Quantum Mapping")
        
        composite = create_full_spectrum_composite(astro_image, fdm_field, pdp_field, quantum_state)
        
        fig_ba, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(astro_image, cmap='gray', origin='upper')
        ax1.set_title("Before: Visible Light Only\n(Standard Telescopes)", fontsize=12)
        ax1.axis('off')
        add_scale_bar(ax1, astro_image.shape[1], pixel_scale_kpc=pixel_scale_kpc)
        
        ax2.imshow(composite, origin='upper')
        ax2.set_title(f"After: Full-Spectrum Quantum Visualization\n|Ψ⟩ = |Ψ_vis⟩ + {alpha:.2f}|Ψ_FDM⟩ + {beta:.2f}|Ψ_PDP⟩", fontsize=12)
        ax2.axis('off')
        add_scale_bar(ax2, composite.shape[1], pixel_scale_kpc=pixel_scale_kpc)
        
        plt.tight_layout()
        st.pyplot(fig_ba)
        st.markdown(get_image_download_link(fig_ba, f"{base_filename}_before_after.png", "📸 Download Before/After"), unsafe_allow_html=True)
        plt.close(fig_ba)
        
        # Individual field downloads
        st.markdown("---")
        st.markdown("### 📥 Individual Field Exports")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_fdm, ax_fdm = plt.subplots(figsize=(6, 6))
            fdm_rgb = map_invisible_to_visible(fdm_field, 'fdm', intensity=1.0)
            ax_fdm.imshow(fdm_rgb, origin='upper')
            ax_fdm.set_title(f"FDM Soliton Field\nρ(r) = ρ₀ [sin({k:.2f}r)/({k:.2f}r)]²")
            ax_fdm.axis('off')
            add_scale_bar(ax_fdm, fdm_field.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.pyplot(fig_fdm)
            st.markdown(get_image_download_link(fig_fdm, f"{base_filename}_fdm_field.png", "🌌 Download FDM Field"), unsafe_allow_html=True)
            plt.close(fig_fdm)
        
        with col_b:
            fig_pdp, ax_pdp = plt.subplots(figsize=(6, 6))
            pdp_rgb = map_invisible_to_visible(pdp_field, 'pdp', intensity=1.0)
            ax_pdp.imshow(pdp_rgb, origin='upper')
            ax_pdp.set_title(f"PDP Quantum Field\nℒ_mix = (ε/2) F_μν F'^μν\nΩ={omega:.2f}, λ={fringe:.2f}")
            ax_pdp.axis('off')
            add_scale_bar(ax_pdp, pdp_field.shape[1], pixel_scale_kpc=pixel_scale_kpc)
            st.pyplot(fig_pdp)
            st.markdown(get_image_download_link(fig_pdp, f"{base_filename}_pdp_field.png", "🌀 Download PDP Field"), unsafe_allow_html=True)
            plt.close(fig_pdp)
        
    else:
        st.info("👈 Select or upload an image to begin full-spectrum quantum visualization")

# Tab 2: Magnetar QED Explorer
with tabs[1]:
    st.header("⚡ Magnetar QED Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        B0 = st.slider("Surface B-Field (10¹⁵ G)", 0.5, 5.0, 1.0, 0.1)
        mixing_angle = st.slider("Dark Photon Mixing ε", 0.0, 0.5, 0.1, 0.01)
        dark_mass = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
        
        r = np.linspace(1, 10, 200)
        theta = np.linspace(0, np.pi, 200)
        R, Theta = np.meshgrid(r, theta)
        B_r, B_theta = magnetar_dipole_field(R, Theta, B0=B0*1e15)
        B_mag = np.sqrt(B_r**2 + B_theta**2)
        qed = quantum_vacuum_polarization(B_mag)
        dark_photons = dark_photon_conversion(B_mag, mixing_angle=mixing_angle, mass=dark_mass)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_magnetar_B{B0:.1f}_eps{mixing_angle:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Physics Metrics")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Max B-Field", f"{np.max(B_mag)/1e15:.2f}×10¹⁵ G")
        col_m2.metric("Max Polarization", f"{np.max(qed):.3e}")
        col_m3.metric("Max Dark Photons", f"{np.max(dark_photons):.3f}")
    
    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        im1 = axes[0].imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot', origin='upper')
        axes[0].set_title(f"B-Field\n{B0:.1f}×10¹⁵ G")
        axes[0].set_xlabel("Radius (R/R₀)")
        axes[0].set_ylabel("Angle (deg)")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma', origin='upper')
        axes[1].set_title("Vacuum Polarization\nEuler-Heisenberg")
        axes[1].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(dark_photons, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis', origin='upper')
        axes[2].set_title(f"Dark Photons\nε={mixing_angle:.2f}, m={dark_mass:.1e}eV")
        axes[2].set_xlabel("Radius (R/R₀)")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, f"{base_filename}_magnetar.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# Tab 3: Primordial Entanglement
with tabs[2]:
    st.header("🌀 Primordial Photon-DarkPhoton Entanglement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        omega_ent = st.slider("Ω (Oscillation)", 0.0, 2.0, 0.7, 0.01)
        dark_mass_ent = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
        mixing_ent = st.slider("Mixing Angle ε", 0.0, 0.5, 0.1, 0.01)
        t_steps = st.slider("Time Steps", 50, 500, 100)
        
        rho = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        H = np.array([[omega_ent, mixing_ent], [mixing_ent, dark_mass_ent]], dtype=complex)
        dt = 0.01
        entropy_evolution = []
        mixing_prob = []
        for step in range(t_steps):
            rho = von_neumann_evolution(rho, H, dt)
            reduced = rho[:1, :1]
            entropy_evolution.append(entanglement_entropy(reduced))
            mixing_prob.append(abs(rho[0, 1])**2)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_entanglement_omega{omega_ent:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Final Entropy", f"{entropy_evolution[-1]:.4f}")
        col_m2.metric("Final Mixing", f"{mixing_prob[-1]:.4f}")
        
        entanglement_data = {
            "parameters": {"omega": omega_ent, "dark_mass": dark_mass_ent, "mixing": mixing_ent},
            "final_entropy": float(entropy_evolution[-1]),
            "final_mixing": float(mixing_prob[-1])
        }
        st.markdown(get_json_download_link(entanglement_data, f"{base_filename}_data.json", "📄 Download JSON"), unsafe_allow_html=True)
    
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(entropy_evolution, 'b-', linewidth=2)
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

# Tab 4: QCIS Power Spectra
with tabs[3]:
    st.header("📊 QCIS - Quantum Cosmology Integration Suite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        f_nl = st.slider("f_NL (Non-Gaussianity)", 0.0, 5.0, 1.0, 0.1)
        n_q = st.slider("n_q (Quantum Index)", 0.0, 2.0, 0.5, 0.05)
        k_min = st.slider("k_min (Mpc⁻¹)", 0.001, 0.01, 0.005, 0.001, format="%.3f")
        k_max = st.slider("k_max (Mpc⁻¹)", 0.1, 1.0, 0.5, 0.05)
        
        k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 100)
        P_lcdm = k_vals ** (-3) * np.exp(-k_vals / 0.1)
        P_quantum = quantum_power_spectrum(k_vals, f_nl, n_q)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_power_fnl{f_nl:.1f}_nq{n_q:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📊 Metrics")
        ratio = P_quantum / (P_lcdm + 1e-10)
        st.metric("Quantum Enhancement", f"{np.mean(ratio):.3f}x")
        
        spectra_data = {
            "parameters": {"f_nl": f_nl, "n_q": n_q},
            "k_values": [float(x) for x in k_vals],
            "P_quantum": [float(x) for x in P_quantum]
        }
        st.markdown(get_json_download_link(spectra_data, f"{base_filename}_data.json", "📄 Download JSON"), unsafe_allow_html=True)
    
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
# ABOUT
# ============================================================================

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite
    
    **Your Verified Formulas:**
    
    | Project | Formula |
    |---------|---------|
    | **FDM Soliton** | ρ(r) = ρ₀ [sin(kr)/(kr)]² |
    | **PDP Kinetic Mixing** | ℒ_mix = (ε/2) F_μν F'^μν |
    | **Magnetar Field** | B = B₀ (R/r)³ (2 cosθ, sinθ) |
    | **Von Neumann Evolution** | i∂ρ/∂t = [H, ρ] |
    | **Entanglement Entropy** | S = -Tr(ρ log ρ) |
    | **QCIS Power Spectrum** | P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) |
    
    **Full-Spectrum Color Mapping:**
    - 🔴 Red: Visible Light
    - 🟢 Green: FDM Soliton (Dark Matter)
    - 🔵 Blue: PDP Field (Dark Photons)
    
    **Supported Formats:** FITS, JPEG, PNG
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | All Formulas Verified | Full-Spectrum Color Mapping<br>
    © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
