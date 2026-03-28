"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS) v6.0
Complete Integration:
1. QCI AstroEntangle Refiner - FDM Solitons & Image Processing
2. Stellaris QED Explorer - Magnetar Physics & Dark Photons
3. Primordial Entanglement - Quantum Mixing & Conversion
4. QCIS - Quantum-Corrected Cosmology & Power Spectra

Author: Tony E. Ford
License: Dual License (Academic/Commercial)
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, sobel
from scipy.integrate import odeint
from scipy.fft import fft2, fftshift
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
import json
from dataclasses import dataclass
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="QCAUS v6.0 - Unified Quantum Cosmology Suite",
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

# Dark professional theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a1a; }
    [data-testid="stSidebar"] { background: #0f0f1f; border-right: 2px solid #00aaff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #ffffff !important; }
    .stTitle, h1, h2, h3 { color: #00aaff !important; }
    [data-testid="stMetricValue"] { color: #00aaff !important; }
    .stDownloadButton button { background-color: #00aaff; color: white; border-radius: 8px; }
    .stImage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── PRELOADED DATASETS ─────────────────────────────────────────────
PRELOADED = {
    "🌌 Bullet Cluster": {"fringe": 70, "omega": 0.75, "scale_kpc": 200, "pattern": "bullet", 
                         "desc": "Merging cluster – dark matter separation"},
    "🔭 Abell 1689": {"fringe": 55, "omega": 0.65, "scale_kpc": 150, "pattern": "abell", 
                      "desc": "Strong lensing – prominent soliton core"},
    "✨ Abell 209": {"fringe": 60, "omega": 0.70, "scale_kpc": 100, "pattern": "abell", 
                     "desc": "Balanced fringe visibility"},
    "🦀 Crab Nebula": {"fringe": 50, "omega": 0.68, "scale_kpc": 2, "pattern": "crab", 
                       "desc": "Supernova remnant – filamentary structure"},
    "📡 Centaurus A": {"fringe": 45, "omega": 0.62, "scale_kpc": 50, "pattern": "centaurus", 
                       "desc": "Radio galaxy – jet structure"}
}


# ── GENERATE SAMPLE IMAGE ─────────────────────────────────────────────
def generate_sample(size=400, pattern="abell"):
    """Generate realistic synthetic image"""
    img = np.zeros((size, size))
    cx, cy = size//2, size//2
    
    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if pattern == "bullet":
                r2 = np.sqrt((i - cx - 50)**2 + (j - cy + 30)**2)
                img[i, j] = np.exp(-r/50) + 0.7 * np.exp(-r2/40)
            elif pattern == "abell":
                img[i, j] = np.exp(-r/60) + 0.2 * np.sin(r/25) * np.exp(-r/80)
            elif pattern == "crab":
                img[i, j] = np.exp(-r/80) + 0.15 * np.sin(i/15) * np.cos(j/15) * np.exp(-r/100)
            elif pattern == "centaurus":
                theta = np.arctan2(j - cy, i - cx)
                img[i, j] = np.exp(-r/70) * (1 + 0.4 * np.cos(2*theta))
            else:
                img[i, j] = np.exp(-r/60)
    
    img = img + np.random.randn(size, size) * 0.02
    img = np.clip(img, 0, None)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# ── QCI ASTROENTANGLE REFINER FUNCTIONS ─────────────────────────────────────────────
def create_soliton(size, fringe):
    """FDM Soliton Core - [sin(kr)/kr]²"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    r_s = 0.2 * (50.0 / max(fringe, 1))
    k = np.pi / max(r_s, 0.01)
    kr = k * r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    soliton = (soliton - soliton.min()) / (soliton.max() - soliton.min() + 1e-9)
    soliton = gaussian_filter(soliton, sigma=2)
    return soliton


def create_wave(size, fringe):
    """Dark Photon Wave Pattern"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 20.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    pattern = radial * 0.5 + spiral * 0.5
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    return pattern


def create_dark_matter_density(image, soliton):
    """Dark Matter Density from gradients"""
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    return np.clip(soliton * 0.6 + gradient * 0.4, 0, 1)


def create_rgb_overlay(image, dark_photon, dm_density, soliton):
    """RGB Composite: R=Image, G=Dark Photon, B=Dark Matter"""
    img_norm = np.clip(image, 0, 1)
    dp_norm = np.clip(dark_photon, 0, 1)
    dm_norm = np.clip(dm_density, 0, 1)
    sol_norm = np.clip(soliton, 0, 1)
    
    red = img_norm
    green = img_norm * 0.3 + dp_norm * 0.5 + sol_norm * 0.2
    blue = img_norm * 0.2 + dm_norm * 0.6 + sol_norm * 0.2
    
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 1)


def process_qci(image, omega, fringe, brightness=1.2):
    """Full QCI processing"""
    h, w = image.shape
    
    soliton = create_soliton((h, w), fringe)
    wave = create_wave((h, w), fringe)
    dm_density = create_dark_matter_density(image, soliton)
    
    mixing = omega * 0.6
    
    result = image * (1 - mixing * 0.4)
    result = result + wave * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = create_rgb_overlay(result, wave, dm_density, soliton)
    entropy = -mixing * np.log(mixing + 1e-12)
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'wave': wave,
        'dark_matter': dm_density,
        'rgb': rgb,
        'mixing': mixing,
        'entropy': entropy,
        'metadata': {
            'omega': omega,
            'fringe': fringe,
            'mixing': mixing,
            'entropy': entropy,
            'brightness': brightness,
        }
    }


# ── STELLARIS QED FUNCTIONS ─────────────────────────────────────────────
def stellaris_magnetar_field(B_surface, r, theta, inclination=0):
    """Magnetar dipole field"""
    theta_rad = np.radians(theta + inclination)
    B0 = B_surface / (r**3)
    B_r = 2 * B0 * np.cos(theta_rad)
    B_theta = B0 * np.sin(theta_rad)
    return B_r, B_theta, np.sqrt(B_r**2 + B_theta**2)


def stellaris_dark_photon_conversion(B, L, epsilon, m_dark, omega=1e18):
    """Dark photon conversion probability"""
    if m_dark <= 0:
        return (epsilon * B / 1e15)**2 * np.ones_like(L)
    hbar_ev_s = 6.582e-16
    c_km_s = 3e5
    conv_len = 4 * omega * hbar_ev_s * c_km_s / (m_dark**2)
    P = (epsilon * B / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    return np.clip(P, 1e-30, 1)


def stellaris_schwinger_pair(B_field):
    """Schwinger pair production rate"""
    B_norm = B_field / B_crit
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.exp(-np.pi / (B_norm + 1e-9))


# ── PRIMORDIAL ENTANGLEMENT FUNCTIONS ─────────────────────────────────────────────
def primordial_von_neumann_evolution(omega, m_dark, H=70, t_max=1.0):
    """Solve von Neumann equation"""
    epsilon = omega * 0.1
    t = np.linspace(0, t_max, 200)
    
    # Analytic approximation for demonstration
    mixing_evolution = epsilon * (1 - np.exp(-H * t))
    entropy_evolution = -mixing_evolution * np.log(mixing_evolution + 1e-12)
    
    return mixing_evolution, entropy_evolution, t


# ── QCIS FRAMEWORK FUNCTIONS ─────────────────────────────────────────────
def qcis_power_spectrum(k, A_s=2.1e-9, n_s=0.965, f_nl=1.0, r=0.01):
    """Quantum-corrected power spectrum"""
    k0 = 0.05
    P_lcdm = A_s * (k / k0)**(n_s - 1)
    quantum_correction = 1 + f_nl * (k / k0)**0.8
    P_quantum = P_lcdm * quantum_correction
    tensor_spectrum = r * P_lcdm
    return P_lcdm, P_quantum, tensor_spectrum, quantum_correction


def qcis_transfer_function(k, omega_m=0.3, omega_b=0.05, h=0.7):
    """Matter transfer function"""
    q = k / (omega_m * h**2)
    T_EH = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
    return T_EH


# ── UTILITY FUNCTIONS ─────────────────────────────────────────────
def array_to_pil(arr):
    """Convert numpy array to PIL Image"""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black', dpi=100)
    buf.seek(0)
    return Image.open(buf)


def add_annotations(image_array, metadata, scale_kpc=100, title_prefix="Before"):
    """Add QCI-style annotations"""
    img_pil = array_to_pil(image_array).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    bar_y = h - 45
    draw.rectangle([20, bar_y, 20 + scale_bar_px, bar_y + 6], fill='white')
    draw.text((20 + 35, bar_y - 20), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_tiny)
    
    # North indicator
    draw.line([w - 35, 30, w - 35, 65], fill='white', width=3)
    draw.text((w - 45, 12), "N", fill='white', font=font_small)
    
    # Physics info box
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ_FDM = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    ]
    
    box_bg = (0, 0, 0, 200)
    draw.rectangle([12, 12, 280, 12 + len(info_lines) * 24 + 8], fill=box_bg, outline='white')
    for i, line in enumerate(info_lines):
        draw.text((18, 18 + i * 24), line, fill='cyan', font=font_tiny)
    
    # Title
    if title_prefix == "Before":
        title_text = "Before: Standard View\n(Public HST/JWST Data)"
    else:
        title_text = "After: Photon-Dark-Photon Entangled\nFDM Overlays (Tony Ford Model)"
    
    bbox = draw.textbbox((0, 0), title_text, font=font_small)
    title_width = bbox[2] - bbox[0]
    draw.rectangle([w//2 - title_width//2 - 15, 10, w//2 + title_width//2 + 15, 58], 
                   fill=(0, 0, 0, 200), outline='white')
    draw.text((w//2 - title_width//2, 15), title_text, fill='white', font=font_small, align='center')
    
    return np.array(img_pil) / 255.0


def create_annotated_side_by_side(original_array, processed_array, metadata, scale_kpc=100):
    """Create side-by-side comparison using PIL"""
    original_pil = add_annotations(original_array, metadata, scale_kpc, "Before")
    processed_pil = add_annotations(processed_array, metadata, scale_kpc, "After")
    
    w_total = original_pil.shape[1] + processed_pil.shape[1]
    h_total = max(original_pil.shape[0], processed_pil.shape[0])
    
    combined = np.zeros((h_total, w_total, 3))
    combined[:, :original_pil.shape[1]] = original_pil
    combined[:, original_pil.shape[1]:] = processed_pil
    
    return combined


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCAUS v6.0")
    st.markdown("*Unified Quantum Cosmology Suite*")
    st.markdown("---")
    
    # Data source
    data_source = st.radio("📁 Data Source", ["📤 Upload Image", "🌌 Preloaded Example"])
    
    if data_source == "📤 Upload Image":
        uploaded = st.file_uploader("Drag & drop file here", type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff'])
        use_upload = True
        current_preset = PRELOADED["🌌 Bullet Cluster"]
    else:
        selected = st.selectbox("Select Object", list(PRELOADED.keys()))
        current_preset = PRELOADED[selected]
        use_upload = False
        st.info(current_preset['desc'])
    
    st.markdown("---")
    st.markdown("### ⚛️ Unified Parameters")
    
    if use_upload:
        omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
        fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    else:
        omega = st.slider("Ω Entanglement", 0.1, 1.0, current_preset["omega"], 0.05)
        fringe = st.slider("Fringe Scale", 20, 120, current_preset["fringe"], 5)
    
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], index=1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("ε Mixing", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford | QCAUS v6.0 | Unified Suite")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("*Complete Integration: QCI + Stellaris + Primordial + QCIS*")
st.markdown("---")

# Metrics
B_ratio = B_surface / B_crit
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("B / B_crit", f"{B_ratio:.2e}")
with col2:
    st.metric("Max γ→A'", f"{(epsilon * B_surface / 1e15)**2:.2e}")
with col3:
    st.metric("Dark Photon Mass", f"{m_dark:.1e} eV")
with col4:
    st.metric("Ω Entanglement", f"{omega:.2f}")

if B_ratio > 1:
    st.warning(f"⚠️ Super-critical field! B/B_crit = {B_ratio:.2e} | QED dominates.")


# ── LOAD AND PROCESS IMAGE ─────────────────────────────────────────────
if use_upload and uploaded is not None:
    with st.spinner(f"Processing {uploaded.name}..."):
        ext = uploaded.name.split(".")[-1].lower()
        if ext == 'fits':
            try:
                from astropy.io import fits
                with fits.open(io.BytesIO(uploaded.read())) as hdul:
                    img_data = hdul[0].data.astype(np.float32)
                    if len(img_data.shape) > 2:
                        img_data = img_data[0]
            except ImportError:
                st.error("Astropy not installed. Install with: pip install astropy")
                st.stop()
        else:
            img = Image.open(uploaded).convert('L')
            img_data = np.array(img, dtype=np.float32) / 255.0
        
        if img_data.shape[0] > 500:
            from skimage.transform import resize
            img_data = resize(img_data, (500, 500), preserve_range=True)
        
        qci_results = process_qci(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {uploaded.name}")

elif not use_upload:
    with st.spinner(f"Loading {selected}..."):
        pattern = PRELOADED[selected]["pattern"]
        img_data = generate_sample(400, pattern)
        qci_results = process_qci(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {selected}")

else:
    st.info("📁 **Upload an image or select a preloaded example**")
    st.stop()


# ── UPDATE METADATA ─────────────────────────────────────────────
qci_results['metadata'].update({
    'omega': omega,
    'fringe': fringe,
    'mixing': qci_results['mixing'],
    'entropy': qci_results['entropy'],
    'brightness': brightness,
})


# ── ANNOTATED SIDE-BY-SIDE ─────────────────────────────────────────────
st.markdown("### 📊 Annotated Comparison")
combined_img = create_annotated_side_by_side(
    qci_results['original'],
    qci_results['rgb'],
    qci_results['metadata'],
    scale_kpc
)
st.image(combined_img, width='stretch')


# ── QUANTUM COMPONENTS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚛️ Quantum Components")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.image(array_to_pil(qci_results['soliton']), caption="FDM Soliton Core", width='stretch')
    st.caption(f"Peak: {qci_results['soliton'].max():.3f} | ρ(r) ∝ [sin(kr)/kr]²")

with col_b:
    st.image(array_to_pil(qci_results['wave']), caption="Dark Photon Field", width='stretch')
    st.caption(f"Contrast: {qci_results['wave'].std():.3f} | λ = h/(m v)")

with col_c:
    st.image(array_to_pil(qci_results['entangled']), caption="PDP Entangled", width='stretch')
    st.caption(f"Mixing: {qci_results['mixing']:.3f} | Entropy: {qci_results['entropy']:.3f}")


# ── MAGNETAR PHYSICS TABS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Magnetar Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetic Field", "🕳️ Dark Photons", "🌀 Kerr Spacetime"])

# Helper to display plots
def show_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black', dpi=100)
    buf.seek(0)
    st.image(Image.open(buf), width='stretch')
    plt.close(fig)

# Tab 1: Magnetar Field
with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 7), facecolor='#0a0a1a')
    ax1.set_facecolor('#0a0a1a')
    
    r = np.linspace(1.2, 5, 40)
    theta = np.linspace(0, 2*np.pi, 40)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    B_val = B_surface / (R**3)
    B_norm = np.log10(B_val + 1e-9)
    B_norm = (B_norm - B_norm.min()) / (B_norm.max() - B_norm.min() + 1e-9)
    
    sc = ax1.scatter(X, Y, c=B_norm, cmap='plasma', s=3, alpha=0.7)
    ax1.add_patch(Circle((0, 0), 1, color='#ff4444', alpha=0.9))
    ax1.text(0, 0, 'NS', color='white', ha='center', va='center', fontsize=12)
    ax1.set_aspect('equal')
    ax1.set_xlim(-5.5, 5.5)
    ax1.set_ylim(-5.5, 5.5)
    ax1.set_title(f'Magnetar Field | B = {B_surface:.1e} G', color='#00aaff')
    ax1.axis('off')
    plt.colorbar(sc, ax=ax1, fraction=0.046, label='log₁₀|B|')
    
    show_plot(fig1)

# Tab 2: Dark Photons
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0a0a1a')
    ax2.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 4, 500)
    P = stellaris_dark_photon_conversion(B_surface, L, epsilon, m_dark)
    
    ax2.semilogx(L, P, '#00aaff', linewidth=2.5)
    ax2.axhline(y=(epsilon * B_surface / 1e15)**2, color='#ff8888', linestyle='--', 
               label=f'Max P = {(epsilon * B_surface / 1e15)**2:.2e}')
    ax2.set_xlabel('Length (km)', color='white')
    ax2.set_ylabel('P(γ→A\')', color='white')
    ax2.set_title('Dark Photon Conversion', color='#00aaff')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    ax2.tick_params(colors='white')
    ax2.set_yscale('log')
    
    show_plot(fig2)
    st.caption(f"ε = {epsilon:.1e} | m' = {m_dark:.1e} eV | B = {B_surface:.1e} G")

# Tab 3: Kerr Geodesics
with tab3:
    fig3, ax3 = plt.subplots(figsize=(8, 7), facecolor='#0a0a1a')
    ax3.set_facecolor('#0a0a1a')
    
    r_horizon = 1 + np.sqrt(1 - a_spin**2)
    circle = Circle((0, 0), r_horizon, color='#555555', alpha=0.7)
    ax3.add_patch(circle)
    ax3.text(0, 0, 'BH', color='white', ha='center', va='center', fontsize=12)
    
    if a_spin <= 0.999:
        r_photon = 2 * (1 + np.cos(2/3 * np.arccos(-abs(a_spin))))
        theta_ph = np.linspace(0, 2*np.pi, 100)
        ax3.plot(r_photon * np.cos(theta_ph), r_photon * np.sin(theta_ph), 
                '#ff8888', linewidth=2, linestyle='--', label='Photon Sphere')
    
    for impact in [6, 8, 10]:
        t = np.linspace(0, 50, 400)
        r = 12 * np.exp(-t/35) + r_horizon + 0.5
        phi = (impact/10) * np.sin(t/25)
        ax3.plot(r * np.cos(phi), r * np.sin(phi), '#88ff88', linewidth=1.5, alpha=0.7)
    
    ax3.set_aspect('equal')
    ax3.set_xlim(-14, 14)
    ax3.set_ylim(-14, 14)
    ax3.set_title(f'Kerr Spacetime | a/M = {a_spin:.3f}', color='#00aaff')
    ax3.legend()
    ax3.axis('off')
    
    show_plot(fig3)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")


# ── PRIMORDIAL ENTANGLEMENT TAB ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🕳️ Primordial Entanglement")

primordial_mixing, primordial_entropy, t_evo = primordial_von_neumann_evolution(omega, m_dark)

col_p1, col_p2 = st.columns(2)

with col_p1:
    fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor='#0a0a1a')
    ax4.plot(t_evo, primordial_mixing, 'r-', linewidth=2)
    ax4.set_xlabel('Scale Factor', color='white')
    ax4.set_ylabel('Mixing Amplitude', color='white')
    ax4.set_title('von Neumann Mixing Evolution', color='#00aaff')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(colors='white')
    st.pyplot(fig4)
    plt.close(fig4)

with col_p2:
    fig5, ax5 = plt.subplots(figsize=(6, 4), facecolor='#0a0a1a')
    ax5.plot(t_evo, primordial_entropy, 'b-', linewidth=2)
    ax5.set_xlabel('Scale Factor', color='white')
    ax5.set_ylabel('Entanglement Entropy', color='white')
    ax5.set_title('Entropy Evolution', color='#00aaff')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(colors='white')
    st.pyplot(fig5)
    plt.close(fig5)


# ── QCIS FRAMEWORK TAB ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌌 QCIS Framework")

k = np.logspace(-3, 0, 100)
P_lcdm, P_quantum, P_tensor, q_corr = qcis_power_spectrum(k, f_nl=omega)
T_k = qcis_transfer_function(k)

col_q1, col_q2 = st.columns(2)

with col_q1:
    fig6, ax6 = plt.subplots(figsize=(6, 4), facecolor='#0a0a1a')
    ax6.loglog(k, P_lcdm, 'b-', linewidth=2, label='ΛCDM')
    ax6.loglog(k, P_quantum, 'r-', linewidth=2, label='Quantum-corrected')
    ax6.fill_between(k, P_lcdm, P_quantum, alpha=0.3, color='red')
    ax6.set_xlabel('k (Mpc⁻¹)', color='white')
    ax6.set_ylabel('P(k)', color='white')
    ax6.set_title('Matter Power Spectrum', color='#00aaff')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.tick_params(colors='white')
    st.pyplot(fig6)
    plt.close(fig6)

with col_q2:
    fig7, ax7 = plt.subplots(figsize=(6, 4), facecolor='#0a0a1a')
    ax7.loglog(k, T_k, 'g-', linewidth=2)
    ax7.set_xlabel('k (Mpc⁻¹)', color='white')
    ax7.set_ylabel('T(k)', color='white')
    ax7.set_title('Transfer Function', color='#00aaff')
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(colors='white')
    st.pyplot(fig7)
    plt.close(fig7)


# ── DOWNLOAD BUTTONS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💾 Download Results")

def save_array_png(arr, cmap=None):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    if len(arr.shape) == 3:
        ax.imshow(np.clip(arr, 0, 1))
    else:
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
    plt.close(fig)
    return buf.getvalue()

def save_pil_image(pil_img):
    buf = io.BytesIO()
    Image.fromarray((pil_img * 255).astype(np.uint8)).save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    st.download_button("📸 Annotated Comparison", save_pil_image(combined_img), "annotated_comparison.png", width='stretch')
with col_d2:
    st.download_button("⭐ Soliton Core", save_array_png(qci_results['soliton'], 'hot'), "soliton.png", width='stretch')
with col_d3:
    st.download_button("🌊 Dark Photon", save_array_png(qci_results['wave'], 'plasma'), "dark_photon.png", width='stretch')
with col_d4:
    st.download_button("⚡ Entangled", save_array_png(qci_results['entangled'], 'inferno'), "entangled.png", width='stretch')

st.markdown("---")
st.markdown("🔭 **QCAUS v6.0** | QCI + Stellaris + Primordial + QCIS | Tony Ford Model")
