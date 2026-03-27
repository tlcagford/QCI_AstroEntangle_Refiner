"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
All 4 Projects + Radar Data Integration + Download Features
FIXED: Comparison grid with radar overlay
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
        
        elif file_ext in ['dcm', 'dicom']:
            import pydicom
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            ds = pydicom.dcmread(tmp_path)
            data = ds.pixel_array
            os.unlink(tmp_path)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"DICOM: {data.shape}"
        
        elif file_ext == 'csv':
            df = pd.read_csv(BytesIO(file_bytes), header=None)
            data = df.values.astype(float)
            data = np.nan_to_num(data)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"CSV: {data.shape}"
        
        elif file_ext in ['npy', 'npz']:
            if file_ext == 'npy':
                data = np.load(BytesIO(file_bytes))
            else:
                npz = np.load(BytesIO(file_bytes))
                data = npz[list(npz.keys())[0]]
            if data.ndim > 2:
                data = np.mean(data, axis=0) if data.ndim == 3 else data[0]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"{file_ext.upper()}: {data.shape}"
        
        elif file_ext in ['tiff', 'tif', 'jpg', 'jpeg', 'png', 'bmp']:
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
    """Generate download link for matplotlib figure"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_array_download_link(data, filename, title="Download"):
    """Generate download link for numpy array as NPZ"""
    buf = BytesIO()
    np.savez_compressed(buf, data=data)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_json_download_link(data, filename, title="Download"):
    """Generate download link for JSON data"""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

# ============================================================================
# PROJECT 1: QCI AstroEntangle Refiner (FDM Soliton + PDP Overlay)
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

def radar_to_overlay(radar_data, image_shape):
    """Convert radar detections to RGBA overlay"""
    H, W = image_shape
    overlay = np.zeros((H, W, 4))
    
    for det in radar_data.get('pdp_detections', []):
        x = int(det.get('azimuth_deg', 180) / 360 * W)
        y = int(det.get('range_km', 150) / 300 * H)
        conf = det.get('confidence', 0.5)
        size = int(15 + conf * 25)
        
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < size:
                        intensity = (0.5 + conf*0.5) * np.exp(-dist**2/(size/3)**2)
                        if conf > 0.6:
                            overlay[yy, xx, 0] += intensity * 1.0
                            overlay[yy, xx, 1] += intensity * 0.2
                            overlay[yy, xx, 2] += intensity * 0.1
                        elif conf > 0.4:
                            overlay[yy, xx, 0] += intensity * 0.2
                            overlay[yy, xx, 1] += intensity * 0.8
                            overlay[yy, xx, 2] += intensity * 0.8
                        else:
                            overlay[yy, xx, 0] += intensity * 0.4
                            overlay[yy, xx, 1] += intensity * 0.2
                            overlay[yy, xx, 2] += intensity * 0.9
                        overlay[yy, xx, 3] = intensity * 0.7
    
    for c in range(3):
        max_val = np.max(overlay[..., c])
        if max_val > 0:
            overlay[..., c] /= max_val
    overlay[..., 3] = np.clip(overlay[..., 3], 0, 1)
    
    return overlay

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
    elif image_name == "CMB":
        img = np.random.randn(size, size)
        img = gaussian_filter(img, sigma=5)
        return (img - img.min()) / (img.max() - img.min())
    else:
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.5 * np.cos(10 * R + 3 * theta))
        return img / img.max()

# ============================================================================
# RADAR DATA FUNCTIONS
# ============================================================================

def create_sample_radar_data():
    return {
        "timestamp": datetime.now().isoformat(),
        "parameters": {"omega": 0.72, "fringe": 1.75, "entanglement": 0.44, "mixing": 0.17},
        "pdp_detections": [{"range_km": 148.8, "azimuth_deg": 179.0, "confidence": 0.641}],
        "aircraft_data": [{"callsign": "STEALTH", "stealth_level": "Low", "range_km": 150, "azimuth_deg": 180}]
    }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("📡 Radar Data Import")
    use_sample = st.checkbox("Use Sample Radar Data", value=True)
    radar_data = None
    if use_sample:
        radar_data = create_sample_radar_data()
        st.success("✅ Using sample radar data")
    else:
        uploaded_radar = st.file_uploader("Upload Radar JSON", type=['json'])
        if uploaded_radar:
            try:
                radar_data = json.load(uploaded_radar)
                st.success(f"✅ Loaded {len(radar_data.get('pdp_detections', []))} detections")
            except:
                st.error("Invalid JSON")
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_name = "galaxy_cluster"
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Nebula", "Bullet Cluster", "CMB", "Spiral Galaxy"])
        astro_image = get_sample_image(image_type)
        image_name = image_type.replace(" ", "_").lower()
    else:
        uploaded_img = st.file_uploader("Upload Image", type=['fits', 'jpg', 'png', 'csv', 'npy', 'npz'])
        if uploaded_img:
            astro_image, info = load_image_file(uploaded_img)
            if astro_image is None:
                st.error(info)
            else:
                st.success(info)
                image_name = uploaded_img.name.split('.')[0]
    
    st.header("🎨 Overlay Settings")
    overlay_opacity = st.slider("Radar Overlay Opacity", 0.0, 1.0, 0.5)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("FDM Soliton • PDP Entanglement • Magnetar QED • Quantum Cosmology")

# ============================================================================
# TAB 1: QCI AstroEntangle Refiner
# ============================================================================

st.header("🔭 QCI AstroEntangle Refiner")
st.markdown("FDM Soliton Physics + Photon-DarkPhoton Entanglement Overlay")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Controls")
    omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01, key="qci_omega")
    fringe = st.slider("Fringe Scale", 0.1, 3.0, 1.0, 0.1, key="qci_fringe")
    soliton_scale = st.slider("FDM Soliton Scale", 0.5, 3.0, 1.0, 0.1, key="soliton_scale")
    
    if astro_image is not None:
        enhanced, soliton, pdp = process_qci_astro(astro_image, omega, fringe, soliton_scale)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{image_name}_omega{omega:.2f}_fringe{fringe:.2f}_soliton{soliton_scale:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Create individual figures for download
        fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
        ax_orig.imshow(astro_image, cmap='gray', origin='lower')
        ax_orig.set_title("Original Image")
        ax_orig.axis('off')
        st.markdown(get_image_download_link(fig_orig, f"{base_filename}_original.png", "📷 Download Original"), unsafe_allow_html=True)
        plt.close(fig_orig)
        
        fig_sol, ax_sol = plt.subplots(figsize=(6, 6))
        ax_sol.imshow(soliton, cmap='viridis', origin='lower')
        ax_sol.set_title(f"FDM Soliton Core (k={soliton_scale:.2f})")
        ax_sol.axis('off')
        st.markdown(get_image_download_link(fig_sol, f"{base_filename}_soliton.png", "🌌 Download FDM Soliton"), unsafe_allow_html=True)
        plt.close(fig_sol)
        
        fig_pdp, ax_pdp = plt.subplots(figsize=(6, 6))
        ax_pdp.imshow(pdp, cmap='plasma', origin='lower')
        ax_pdp.set_title(f"PDP Entanglement (Ω={omega:.2f})")
        ax_pdp.axis('off')
        st.markdown(get_image_download_link(fig_pdp, f"{base_filename}_pdp.png", "🌀 Download PDP Entanglement"), unsafe_allow_html=True)
        plt.close(fig_pdp)
        
        fig_enh, ax_enh = plt.subplots(figsize=(6, 6))
        ax_enh.imshow(enhanced, cmap='gray', origin='lower')
        ax_enh.set_title("Enhanced Image")
        ax_enh.axis('off')
        st.markdown(get_image_download_link(fig_enh, f"{base_filename}_enhanced.png", "✨ Download Enhanced Image"), unsafe_allow_html=True)
        plt.close(fig_enh)
        
        metadata = {
            "project": "QCI AstroEntangle Refiner",
            "image": image_name,
            "parameters": {"omega": omega, "fringe": fringe, "soliton_scale": soliton_scale},
            "timestamp": timestamp,
            "soliton_stats": {"mean": float(np.mean(soliton)), "max": float(np.max(soliton))},
            "pdp_stats": {"mean": float(np.mean(pdp)), "max": float(np.max(pdp))}
        }
        st.markdown(get_json_download_link(metadata, f"{base_filename}_metadata.json", "📄 Download Metadata"), unsafe_allow_html=True)

with col2:
    st.subheader("Before / After Comparison")
    if astro_image is not None:
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(astro_image, cmap='gray', origin='lower')
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(soliton, cmap='viridis', origin='lower')
        axes[0, 1].set_title(f"FDM Soliton Core\nk={soliton_scale:.2f}")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pdp, cmap='plasma', origin='lower')
        axes[1, 0].set_title(f"PDP Entanglement\nΩ={omega:.2f}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(enhanced, cmap='gray', origin='lower')
        axes[1, 1].set_title("Enhanced + Overlay")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download comparison grid
        comp_filename = f"{base_filename}_comparison_grid.png"
        st.markdown(get_image_download_link(fig, comp_filename, "📸 Download Comparison Grid"), unsafe_allow_html=True)
        plt.close(fig)
        
        # Radar Overlay on Enhanced Image
        if radar_data:
            st.subheader("📡 Radar Detection Overlay")
            overlay = radar_to_overlay(radar_data, astro_image.shape)
            astro_rgb = np.stack([enhanced] * 3, axis=-1)
            alpha = overlay[..., 3:4] * overlay_opacity
            combined = astro_rgb * (1 - alpha) + overlay[..., :3] * alpha
            
            fig_radar, ax_radar = plt.subplots(figsize=(8, 8))
            ax_radar.imshow(np.clip(combined, 0, 1), origin='lower')
            ax_radar.set_title("Enhanced Image + Radar Overlay")
            ax_radar.axis('off')
            st.pyplot(fig_radar)
            
            radar_filename = f"{base_filename}_radar_overlay.png"
            st.markdown(get_image_download_link(fig_radar, radar_filename, "📡 Download Radar Overlay"), unsafe_allow_html=True)
            plt.close(fig_radar)
    else:
        st.info("👈 Select or upload an image")

# ============================================================================
# TAB 2: Magnetar QED Explorer
# ============================================================================

st.header("⚡ Magnetar QED Explorer")
st.markdown("Magnetar Fields • Quantum Vacuum Polarization • Dark Photon Conversion")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Magnetar Parameters")
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
    
    # Download B-field
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    im_b = ax_b.imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
    ax_b.set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G")
    plt.colorbar(im_b, ax=ax_b)
    st.markdown(get_image_download_link(fig_b, f"{base_filename}_bfield.png", "🔴 Download B-Field"), unsafe_allow_html=True)
    plt.close(fig_b)
    
    # Download QED
    fig_q, ax_q = plt.subplots(figsize=(6, 4))
    im_q = ax_q.imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma')
    ax_q.set_title("Quantum Vacuum Polarization")
    plt.colorbar(im_q, ax=ax_q)
    st.markdown(get_image_download_link(fig_q, f"{base_filename}_qed.png", "⚛️ Download QED Polarization"), unsafe_allow_html=True)
    plt.close(fig_q)
    
    # Download Dark Photons
    fig_d, ax_d = plt.subplots(figsize=(6, 4))
    im_d = ax_d.imshow(dark_photons, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis')
    ax_d.set_title(f"Dark Photons\nε={mixing_angle:.2f}")
    plt.colorbar(im_d, ax=ax_d)
    st.markdown(get_image_download_link(fig_d, f"{base_filename}_dark_photons.png", "🌑 Download Dark Photons"), unsafe_allow_html=True)
    plt.close(fig_d)

with col2:
    st.subheader("Visualization")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im1 = axes[0].imshow(B_mag, extent=[1, 10, 0, 180], aspect='auto', cmap='hot')
    axes[0].set_title(f"Magnetic Field\n{B0:.1f}×10¹⁵ G")
    axes[0].set_xlabel("Radius (R/R₀)")
    axes[0].set_ylabel("Angle (deg)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma')
    axes[1].set_title("Quantum Vacuum\nPolarization")
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(dark_photons, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis')
    axes[2].set_title(f"Dark Photons\nε={mixing_angle:.2f}")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    comp_filename = f"{base_filename}_magnetar_comparison.png"
    st.markdown(get_image_download_link(fig, comp_filename, "📸 Download Complete Visualization"), unsafe_allow_html=True)
    plt.close(fig)

# ============================================================================
# TAB 3: Primordial Photon-DarkPhoton Entanglement
# ============================================================================

st.header("🌀 Primordial Photon-DarkPhoton Entanglement")
st.markdown("Von Neumann Evolution • Entanglement Entropy • Mixing Probability")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameters")
    omega_ent = st.slider("Ω (Oscillation)", 0.0, 2.0, 0.7, 0.01, key="prim_omega")
    dark_mass = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    mixing_prim = st.slider("Mixing Angle ε", 0.0, 0.5, 0.1, 0.01, key="prim_mixing")
    t_steps = st.slider("Time Steps", 50, 500, 100)
    
    entropy, mixing_prob = process_primordial_entanglement(omega_ent, dark_mass, mixing_prim, t_steps)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"qcaus_entanglement_omega{omega_ent:.2f}_mass{dark_mass:.1e}_eps{mixing_prim:.2f}_{timestamp}"
    
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    entanglement_data = {
        "parameters": {"omega": omega_ent, "dark_photon_mass": dark_mass, "mixing_angle": mixing_prim, "time_steps": t_steps},
        "entropy_evolution": [float(x) for x in entropy],
        "mixing_probability": [float(x) for x in mixing_prob],
        "final_entropy": float(entropy[-1]),
        "final_mixing": float(mixing_prob[-1])
    }
    st.markdown(get_json_download_link(entanglement_data, f"{base_filename}_data.json", "📄 Download Entanglement Data"), unsafe_allow_html=True)
    st.markdown(get_array_download_link(np.array(entropy), f"{base_filename}_entropy.npy", "📊 Download Entropy Array"), unsafe_allow_html=True)

with col2:
    st.subheader("Entanglement Evolution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(entropy, 'b-', linewidth=2)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Entanglement Entropy S")
    ax1.set_title("Von Neumann Entropy")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mixing_prob, 'r-', linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Mixing Probability |⟨ψ_d|ψ_γ⟩|²")
    ax2.set_title("Photon-Dark Photon Mixing")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown(get_image_download_link(fig, f"{base_filename}_evolution.png", "📸 Download Evolution Plot"), unsafe_allow_html=True)
    plt.close(fig)
    
    st.caption(f"**Final Entropy:** {entropy[-1]:.4f} | **Final Mixing:** {mixing_prob[-1]:.4f}")

# ============================================================================
# TAB 4: QCIS - Quantum Cosmology Integration Suite
# ============================================================================

st.header("📊 QCIS - Quantum Cosmology Integration Suite")
st.markdown("Quantum-Corrected Power Spectra • ΛCDM vs Quantum • Bayesian Evidence")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cosmological Parameters")
    f_nl = st.slider("f_NL (Non-Gaussianity)", 0.0, 5.0, 1.0, 0.1)
    n_q = st.slider("n_q (Quantum Index)", 0.0, 2.0, 0.5, 0.05)
    k_min = st.slider("k_min (Mpc⁻¹)", 0.001, 0.01, 0.005, 0.001, format="%.3f")
    k_max = st.slider("k_max (Mpc⁻¹)", 0.1, 1.0, 0.5, 0.05)
    
    k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 100)
    P_quantum = process_qcis(k_vals, f_nl, n_q)
    P_lcdm = k_vals ** (-3) * np.exp(-k_vals / 0.1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"qcaus_power_spectrum_fnl{f_nl:.1f}_nq{n_q:.2f}_{timestamp}"
    
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    spectra_data = {
        "parameters": {"f_nl": f_nl, "n_q": n_q, "k_min": k_min, "k_max": k_max},
        "k_values": [float(x) for x in k_vals],
        "P_lcdm": [float(x) for x in P_lcdm],
        "P_quantum": [float(x) for x in P_quantum],
        "ratio_mean": float(np.mean(P_quantum / (P_lcdm + 1e-10)))
    }
    st.markdown(get_json_download_link(spectra_data, f"{base_filename}_data.json", "📄 Download Power Spectra Data"), unsafe_allow_html=True)
    st.markdown(get_array_download_link(k_vals, f"{base_filename}_k_values.npy", "📊 Download k-values"), unsafe_allow_html=True)
    st.markdown(get_array_download_link(P_quantum, f"{base_filename}_P_quantum.npy", "📊 Download Quantum Power Spectrum"), unsafe_allow_html=True)

with col2:
    st.subheader("Power Spectra")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k_vals, P_lcdm, 'b-', label='ΛCDM', linewidth=2)
    ax.loglog(k_vals, P_quantum, 'r--', label=f'Quantum (f_NL={f_nl:.1f}, n_q={n_q:.2f})', linewidth=2)
    ax.set_xlabel("k (Mpc⁻¹)")
    ax.set_ylabel("P(k)")
    ax.set_title("Matter Power Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown(get_image_download_link(fig, f"{base_filename}_power_spectrum.png", "📸 Download Power Spectrum"), unsafe_allow_html=True)
    plt.close(fig)
    
    ratio = P_quantum / (P_lcdm + 1e-10)
    st.metric("Quantum Enhancement", f"{np.mean(ratio):.3f}x")
    st.metric("Spectral Index n_s", f"{n_q:.2f}")

# ============================================================================
# ABOUT
# ============================================================================

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite
    
    **Four Integrated Projects:**
    
    1. **QCI AstroEntangle Refiner** - FDM soliton physics + PDP entanglement on astronomical images
    2. **Magnetar QED Explorer** - Strong-field QED, dark photon conversion
    3. **Primordial Entanglement** - Von Neumann evolution of photon-dark photon system
    4. **QCIS** - Quantum-corrected cosmological power spectra
    
    **Radar Data Integration:**
    - Upload JSON exports from StealthPDPRadar
    - Overlay detection signatures on any image
    - Download all results with smart naming
    
    **Smart File Naming:** `project_target_parameters_timestamp.png`
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | FDM Soliton • PDP Entanglement • Magnetar QED • QCIS<br>
    Integrated with StealthPDPRadar | © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
