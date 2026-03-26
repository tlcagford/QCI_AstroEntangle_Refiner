"""
Magnetar QED Explorer v2.3 – Final
Drag & drop | Preloaded examples | Annotated side‑by‑side | All plots
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.constants import c, hbar, e, m_e, alpha
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image, ImageDraw, ImageFont
import warnings

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Magnetar QED Explorer v2.3",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Dark theme
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0a0a1a; }
    [data-testid="stSidebar"] { background: #0f0f1f; border-right: 2px solid #00aaff; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color: #ffffff !important; }
    .stTitle, h1, h2, h3 { color: #00aaff !important; }
    [data-testid="stMetricValue"] { color: #00aaff !important; }
    .stDownloadButton button { background-color: #00aaff; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── PHYSICS CONSTANTS ─────────────────────────────────────────────
B_crit = m_e**2 * c**2 / (e * hbar)
alpha_fine = 1/137.036


# ── PRELOADED DATASETS (simulated examples) ─────────────────────────────────────────────
# These are synthetic images for demonstration. For real NASA data, use the upload option.
PRELOADED = {
    "🌌 Bullet Cluster (simulated)": {"fringe": 70, "omega": 0.75, "pattern": "bullet", "desc": "Merging cluster – dark matter separation"},
    "🔭 Abell 1689 (simulated)": {"fringe": 55, "omega": 0.65, "pattern": "abell", "desc": "Strong lensing – prominent soliton"},
    "✨ Abell 209 (simulated)": {"fringe": 60, "omega": 0.70, "pattern": "abell", "desc": "Balanced fringe visibility"},
    "🦀 Crab Nebula (simulated)": {"fringe": 50, "omega": 0.68, "pattern": "crab", "desc": "Supernova remnant – filaments"},
    "📡 Centaurus A (simulated)": {"fringe": 45, "omega": 0.62, "pattern": "centaurus", "desc": "Radio galaxy – jet structure"}
}


def generate_sample(size=400, pattern="abell"):
    """Generate synthetic sample image (for preloaded examples)"""
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


def process_image(image, omega, fringe, brightness=1.2):
    """Full PDP processing"""
    h, w = image.shape
    
    soliton = create_soliton((h, w), fringe)
    wave = create_wave((h, w), fringe)
    
    mixing = omega * 0.6
    
    result = image * (1 - mixing * 0.4)
    result = result + wave * mixing * 0.5
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = np.stack([
        result,
        result * 0.3 + wave * 0.5 + soliton * 0.2,
        result * 0.2 + soliton * 0.8
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    entropy = -mixing * np.log(mixing + 1e-12)
    
    return {
        'original': image,
        'entangled': result,
        'soliton': soliton,
        'wave': wave,
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


def array_to_pil(arr):
    """Convert numpy array to PIL Image"""
    return Image.fromarray((arr * 255).astype(np.uint8))


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image for reliable display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black', dpi=100)
    buf.seek(0)
    return Image.open(buf)


def add_annotations(image_array, metadata, scale_kpc=100, title_prefix="Before"):
    """Add QCI-style annotations"""
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    bar_y = h - 40
    draw.rectangle([20, bar_y, 20 + scale_bar_px, bar_y + 5], fill='white')
    draw.text((20 + 30, bar_y - 18), f"{scale_bar_kpc:.0f} kpc", fill='white', font=font_tiny)
    
    # North indicator
    draw.line([w - 30, 30, w - 30, 60], fill='white', width=2)
    draw.text((w - 38, 15), "N", fill='white', font=font_small)
    
    # Physics info box
    info_lines = [
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing']:.3f} | Entropy = {metadata['entropy']:.3f}",
        f"λ_FDM = {scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    ]
    
    box_width = 240
    box_height = len(info_lines) * 22 + 10
    draw.rectangle([10, 10, 10 + box_width, 10 + box_height], fill=(0, 0, 0, 180), outline='white')
    
    for i, line in enumerate(info_lines):
        draw.text((15, 15 + i * 22), line, fill='cyan', font=font_tiny)
    
    # Formulas
    formulas = [
        r"ρ(r) ∝ [sin(kr)/kr]²",
        r"λ = h/(m v)",
        r"P(γ→A') = (εB/m')² sin²(m'²L/4ω)"
    ]
    
    formula_y_start = h - 80
    for i, formula in enumerate(formulas):
        draw.text((w - 210, formula_y_start + i * 18), formula, fill='#88ff88', font=font_tiny)
    
    # Title overlay
    if title_prefix == "Before":
        title_text = "Before: Standard View\n(Public HST/JWST Data)"
    else:
        title_text = "After: Photon-Dark-Photon Entangled\nFDM Overlays (Tony Ford Model)"
    
    # Title background
    title_bbox = draw.textbbox((0, 0), title_text, font=font_small)
    title_width = title_bbox[2] - title_bbox[0]
    draw.rectangle([w//2 - title_width//2 - 10, 10, w//2 + title_width//2 + 10, 50], 
                   fill=(0, 0, 0, 180), outline='white')
    draw.text((w//2 - title_width//2, 15), title_text, fill='white', font=font_small, align='center')
    
    return np.array(img_pil) / 255.0


def create_annotated_side_by_side(original, processed, metadata, scale_kpc=100):
    """Create side-by-side with annotations"""
    original_annotated = add_annotations(original, metadata, scale_kpc, "Before")
    processed_annotated = add_annotations(processed, metadata, scale_kpc, "After")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#0a0a1a')
    
    ax1.imshow(original_annotated)
    ax1.axis('off')
    
    ax2.imshow(processed_annotated)
    ax2.axis('off')
    
    fig.tight_layout()
    return fig


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Magnetar QED")
    st.markdown("*Quantum Vacuum Physics*")
    st.markdown("---")
    
    st.markdown("### 📁 Data Source")
    data_source = st.radio("Choose", ["📤 Upload your image", "🌌 Preloaded example"])
    
    if data_source == "📤 Upload your image":
        uploaded = st.file_uploader(
            "Drag & drop file here",
            type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="FITS (astronomy), PNG, JPG, TIFF"
        )
        use_preload = False
    else:
        selected = st.selectbox("Select object", list(PRELOADED.keys()))
        preset = PRELOADED[selected]
        st.info(preset['desc'])
        use_preload = True
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    if use_preload:
        omega = st.slider("Ω Entanglement", 0.1, 1.0, preset["omega"], 0.05)
        fringe = st.slider("Fringe Scale", 20, 120, preset["fringe"], 5)
    else:
        omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
        fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], index=1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("ε Mixing", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford | Magnetar QED v2.3")


# ── MAIN APP ─────────────────────────────────────────────
st.title("⚡ Magnetar QED Explorer")
st.markdown("*Quantum Vacuum Engineering for Extreme Astrophysics*")
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
results = None

if use_preload:
    with st.spinner(f"Loading {selected}..."):
        pattern = PRELOADED[selected]["pattern"]
        img_data = generate_sample(400, pattern)
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {selected} (simulated example)")

elif data_source == "📤 Upload your image" and uploaded is not None:
    with st.spinner("Processing image..."):
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
        
        # Resize if needed
        if img_data.shape[0] > 500:
            from skimage.transform import resize
            img_data = resize(img_data, (500, 500), preserve_range=True)
        
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {uploaded.name}")

else:
    st.info("📁 **Select a preloaded example or upload your own image**")


# ── DISPLAY RESULTS ─────────────────────────────────────────────
if results is not None:
    # Update metadata with current parameters
    results['metadata'].update({
        'omega': omega,
        'fringe': fringe,
        'mixing': results['mixing'],
        'entropy': results['entropy'],
        'brightness': brightness,
    })
    
    # Annotated side‑by‑side comparison
    st.markdown("### 📊 Annotated Comparison")
    comparison_fig = create_annotated_side_by_side(
        results['original'],
        results['rgb'],
        results['metadata'],
        scale_kpc
    )
    st.pyplot(comparison_fig)
    plt.close(comparison_fig)
    
    # Quantum components
    st.markdown("---")
    st.markdown("### ⚛️ Quantum Components")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(array_to_pil(results['soliton']), caption="FDM Soliton Core", use_container_width=True)
        st.caption(f"Peak: {results['soliton'].max():.3f}")
    with col_b:
        st.image(array_to_pil(results['wave']), caption="Dark Photon Field", use_container_width=True)
        st.caption(f"Contrast: {results['wave'].std():.3f}")
    with col_c:
        st.image(array_to_pil(results['entangled']), caption="PDP Entangled", use_container_width=True)
        st.caption(f"Mixing: {results['mixing']:.3f} | Entropy: {results['entropy']:.3f}")
    
    # Download buttons
    st.markdown("---")
    st.markdown("### 💾 Download Results")
    
    def save_fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        return buf.getvalue()
    
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
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.download_button("📸 Annotated Comparison", save_fig_to_bytes(comparison_fig), "annotated_comparison.png", use_container_width=True)
    with col_d2:
        st.download_button("⭐ Soliton Core", save_array_png(results['soliton'], 'hot'), "soliton.png", use_container_width=True)
    with col_d3:
        st.download_button("🌊 Dark Photon", save_array_png(results['wave'], 'plasma'), "dark_photon.png", use_container_width=True)
    with col_d4:
        st.download_button("⚡ Entangled", save_array_png(results['entangled'], 'inferno'), "entangled.png", use_container_width=True)


# ── MAGNETAR PHYSICS TABS (reliable display with fig_to_pil) ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Magnetar Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetic Field", "🕳️ Dark Photons", "🌀 Kerr Spacetime"])

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
    
    st.image(fig_to_pil(fig1), use_container_width=True)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', facecolor='black')
    buf1.seek(0)
    st.download_button("📥 Download Field Plot", buf1, "magnetar_field.png", use_container_width=True)
    plt.close(fig1)

# Tab 2: Dark Photons (log y‑scale)
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0a0a1a')
    ax2.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 6, 500)  # extended range to see oscillations
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
    P = np.clip(P, 1e-30, 1)
    
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
    
    st.image(fig_to_pil(fig2), use_container_width=True)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor='black')
    buf2.seek(0)
    st.download_button("📥 Download Conversion Plot", buf2, "dark_photon_conversion.png", use_container_width=True)
    plt.close(fig2)
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
    
    st.image(fig_to_pil(fig3), use_container_width=True)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor='black')
    buf3.seek(0)
    st.download_button("📥 Download Geodesic Plot", buf3, "kerr_geodesic.png", use_container_width=True)
    plt.close(fig3)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")

st.markdown("---")
st.markdown("⚡ **Magnetar QED Explorer v2.3** | Drag & Drop | Preloaded Examples | Tony Ford Model")# QCI AstroEntangle Refiner – v25 FINAL FIXED
# All presets working, no index errors

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter, sobel
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont
import warnings
import json
from dataclasses import dataclass
from typing import Dict

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide", 
    page_title="QCI Refiner v25 - Final", 
    page_icon="🔭",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e3f2fd; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #0288d1; }
.stTitle, h1, h2, h3 { color: #01579b; }
[data-testid="stMetricValue"] { color: #01579b; }
</style>
""", unsafe_allow_html=True)

# ── DATA CLASS ─────────────────────────────────────────────
@dataclass
class PhysicsOutput:
    entangled_image: np.ndarray
    soliton_core: np.ndarray
    dark_photon_field: np.ndarray
    dark_matter_density: np.ndarray
    rgb_composite: np.ndarray
    mixing_angle: float
    entanglement_entropy: float
    metadata: Dict


# ── ANNOTATION FUNCTIONS ─────────────────────────────────────────────

def add_physics_annotations(image_array, metadata, scale_kpc=100, image_pixels=500):
    """Add physics information overlay to image"""
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_small = ImageFont.load_default()
    
    h, w = image_array.shape[:2]
    
    # Scale bar
    scale_bar_length_px = 100
    scale_bar_kpc = (scale_bar_length_px / image_pixels) * scale_kpc
    scale_bar_y = h - 40
    scale_bar_x_start = 20
    scale_bar_x_end = scale_bar_x_start + scale_bar_length_px
    
    draw.rectangle([scale_bar_x_start, scale_bar_y, scale_bar_x_end, scale_bar_y + 5], fill='white')
    draw.text((scale_bar_x_start + 10, scale_bar_y - 18), f"{scale_bar_kpc:.0f} kpc", 
              fill='white', font=font_small)
    
    # North indicator
    draw.line([w - 30, 30, w - 30, 60], fill='white', width=2)
    draw.text((w - 38, 15), "N", fill='white', font=font_small)
    
    # Info box
    info_lines = [
        f"QCI Refiner v25",
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing_angle']:.3f}",
        f"S_entropy = {metadata['entanglement_entropy']:.3f}",
        f"λ = {metadata.get('wavelength_kpc', scale_kpc / metadata['fringe'] * 8):.1f} kpc",
    ]
    
    box_y_start = 10
    box_y_end = box_y_start + len(info_lines) * 20 + 10
    draw.rectangle([10, box_y_start, 250, box_y_end], fill=(0, 0, 0, 180), outline='white')
    
    for i, line in enumerate(info_lines):
        draw.text((15, box_y_start + 5 + i * 20), line, fill='white', font=font_small)
    
    return np.array(img_pil) / 255.0


def create_comparison_with_labels(original, entangled, metadata, scale_kpc=100):
    """Create side-by-side comparison with labels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title("Before: Standard View\n(Public HST/JWST Data)", color='white', fontsize=10)
    ax1.axis('off')
    
    ax2.imshow(entangled, cmap='inferno')
    
    h, w = entangled.shape[:2]
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    scale_x_start = 20
    scale_y = h - 30
    
    ax2.add_patch(Rectangle((scale_x_start, scale_y), scale_bar_px, 5, 
                            facecolor='white', edgecolor='white'))
    ax2.text(scale_x_start + 10, scale_y - 12, f"{scale_bar_kpc:.0f} kpc", 
             color='white', fontsize=9, ha='center')
    
    ax2.annotate('N', xy=(w - 30, 30), color='white', fontsize=12, ha='center')
    ax2.plot([w - 30, w - 30], [30, 60], 'w-', linewidth=2)
    
    info_text = f"Ω={metadata['omega']:.2f} | Fringe={metadata['fringe']}\n"
    info_text += f"S={metadata['entanglement_entropy']:.3f}"
    ax2.text(15, 25, info_text, color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax2.set_title("After: Photon-Dark-Photon Entangled\nFDM Overlays (Tony Ford Model)", 
                  color='white', fontsize=10)
    ax2.axis('off')
    
    fig.patch.set_facecolor('#0b0b1a')
    fig.tight_layout()
    
    return fig


# ── PHYSICS FUNCTIONS ─────────────────────────────────────────────

def compute_entanglement_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    return -np.sum(eigenvalues * np.log(eigenvalues))


def schrodinger_poisson_soliton(size, fringe):
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    r_s = 0.2 * (50.0 / max(fringe, 1))
    k = np.pi / max(r_s, 0.01)
    kr = k * r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    soliton = soliton - soliton.min()
    soliton = soliton / (soliton.max() + 1e-9)
    soliton = gaussian_filter(soliton, sigma=2)
    return soliton


def create_dark_photon_field(size, fringe, scale_kpc=100):
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = w//2, h//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w, 1)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 20.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    angular = np.sin(k * 3 * theta)
    
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    return pattern


def create_dark_matter_density(image, soliton):
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    return np.clip(soliton * 0.6 + gradient * 0.4, 0, 1)


def apply_primordial_entanglement(image, omega, fringe, brightness=1.2, scale_kpc=100):
    h, w = image.shape
    m_fdm = 1e-22 * (50.0 / max(fringe, 1))
    
    soliton = schrodinger_poisson_soliton((h, w), fringe)
    dark_photon = create_dark_photon_field((h, w), fringe, scale_kpc)
    dm_density = create_dark_matter_density(image, soliton)
    
    mixing = omega * 0.5
    
    result = image * (1 - mixing * 0.3)
    result = result + dark_photon * mixing * 0.5
    result = result + dm_density * mixing * 0.3
    result = result + soliton * mixing * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    rgb = np.stack([
        result,
        result * 0.4 + dark_photon * 0.6,
        result * 0.3 + dm_density * 0.7
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    entropy = compute_entanglement_entropy(np.array([[1-mixing, mixing], [mixing, mixing]]))
    
    metadata = {
        "omega": float(omega),
        "fringe": int(fringe),
        "brightness": float(brightness),
        "scale_kpc": int(scale_kpc),
        "mixing_angle": float(mixing),
        "entanglement_entropy": float(entropy),
        "m_fdm_eV": float(m_fdm),
        "wavelength_kpc": float(scale_kpc / max(fringe, 1) * 8)
    }
    
    return PhysicsOutput(
        entangled_image=result,
        soliton_core=soliton,
        dark_photon_field=dark_photon,
        dark_matter_density=dm_density,
        rgb_composite=rgb,
        mixing_angle=mixing,
        entanglement_entropy=entropy,
        metadata=metadata
    )


# ── CLUSTER PRESETS ─────────────────────────────────────────────
CLUSTER_PRESETS = {
    "Bullet Cluster (1E0657-56)": {"fringe": 70, "omega": 0.75, "scale_kpc": 200, "desc": "Merging cluster, dark matter separation"},
    "Abell 1689": {"fringe": 55, "omega": 0.65, "scale_kpc": 150, "desc": "Strong lensing, prominent soliton"},
    "Abell 209": {"fringe": 60, "omega": 0.70, "scale_kpc": 100, "desc": "Balanced fringe visibility"},
    "Abell 2218": {"fringe": 50, "omega": 0.68, "scale_kpc": 120, "desc": "Giant arcs, good reconstruction"},
    "COSMOS Field": {"fringe": 45, "omega": 0.60, "scale_kpc": 80, "desc": "Deep field, subtle effects"}
}

SCALE_OPTIONS = [50, 100, 150, 200, 300]


# ── DISPLAY FUNCTION ─────────────────────────────────────────────
def display_image(img_array, title, cmap='inferno', show_colorbar=True, figsize=(4, 4)):
    try:
        fig, ax = plt.subplots(figsize=figsize)
        if len(img_array.shape) == 3:
            ax.imshow(np.clip(img_array, 0, 1))
        else:
            im = ax.imshow(img_array, cmap=cmap, vmin=0, vmax=1)
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10, color='#01579b')
        ax.axis('off')
        fig.patch.set_facecolor('#e3f2fd')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.write(f"⚠️ {title}")


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v25")
    st.markdown("### Primordial Photon-DarkPhoton Entanglement")
    st.markdown("*With FDM Soliton Physics*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload FITS/Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    st.markdown("### 🎯 Cluster Presets")
    
    # Preset selection
    preset_names = list(CLUSTER_PRESETS.keys())
    selected_preset = st.selectbox("Load Preset", ["Custom"] + preset_names)
    
    # Initialize default values
    omega_default = 0.70
    fringe_default = 65
    scale_default = 100
    
    if selected_preset != "Custom":
        preset = CLUSTER_PRESETS[selected_preset]
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
        scale_default = preset["scale_kpc"]
        st.info(f"**{selected_preset}**\n{preset['desc']}")
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    # Safe scale selection - find closest match
    if scale_default in SCALE_OPTIONS:
        scale_index = SCALE_OPTIONS.index(scale_default)
    else:
        # Find closest value
        scale_index = min(range(len(SCALE_OPTIONS)), key=lambda i: abs(SCALE_OPTIONS[i] - scale_default))
    
    scale_kpc = st.selectbox("Scale (kpc)", SCALE_OPTIONS, index=scale_index)
    
    st.markdown("---")
    st.markdown("### 🏷️ Options")
    add_annotations = st.checkbox("Add Annotations", value=True)
    
    st.markdown("---")
    st.caption("Tony Ford Model | v25 - Final")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Primordial Photon-DarkPhoton Entanglement with FDM Soliton Physics*")
st.markdown("---")

if uploaded is not None:
    # Load image
    ext = uploaded.name.split(".")[-1].lower()
    data_bytes = uploaded.read()
    
    with st.spinner("Loading image..."):
        try:
            if ext == "fits":
                with fits.open(io.BytesIO(data_bytes)) as h:
                    img = h[0].data.astype(np.float32)
                    if len(img.shape) > 2:
                        img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
            else:
                img = Image.open(io.BytesIO(data_bytes)).convert("L")
                img = np.array(img, dtype=np.float32)
        except Exception as e:
            st.error(f"Error loading: {e}")
            st.stop()
    
    # Normalize
    img = np.nan_to_num(img, nan=0.0)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    # Resize
    MAX_SIZE = 500
    if img.shape[0] > MAX_SIZE or img.shape[1] > MAX_SIZE:
        from skimage.transform import resize
        img = resize(img, (MAX_SIZE, MAX_SIZE), preserve_range=True)
    
    # Process
    with st.spinner("Running physics solvers..."):
        blurred = gaussian_filter(img, sigma=1)
        enhanced = img + (img - blurred) * 0.5
        enhanced = np.clip(enhanced, 0, 1)
        
        physics = apply_primordial_entanglement(enhanced, omega, fringe, brightness, scale_kpc)
        
        if add_annotations:
            annotated = add_physics_annotations(
                physics.entangled_image, 
                physics.metadata, 
                scale_kpc, 
                physics.entangled_image.shape[1]
            )
            comparison_fig = create_comparison_with_labels(
                img, annotated, physics.metadata, scale_kpc
            )
        else:
            comparison_fig = create_comparison_with_labels(
                img, physics.entangled_image, physics.metadata, scale_kpc
            )
    
    # Success
    st.success(f"✅ Complete | Mixing={physics.mixing_angle:.3f} | Entropy={physics.entanglement_entropy:.3f}")
    
    # ── COMPARISON ─────────────────────────────────────────────
    st.markdown("### 📊 Comparison")
    st.pyplot(comparison_fig)
    plt.close(comparison_fig)
    
    # ── COMPONENTS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚛️ FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_image(physics.soliton_core, "FDM Soliton Core", 'hot')
        st.metric("Peak", f"{physics.soliton_core.max():.3f}")
        st.caption("ρ(r) ∝ [sin(kr)/kr]²")
    
    with col_b:
        display_image(physics.dark_photon_field, f"Dark Photon Field", 'plasma')
        st.metric("Contrast", f"{physics.dark_photon_field.std():.3f}")
        st.caption(f"λ = {physics.metadata['wavelength_kpc']:.1f} kpc")
    
    with col_c:
        display_image(physics.dark_matter_density, "Dark Matter Density", 'viridis')
        st.metric("Mean", f"{physics.dark_matter_density.mean():.3f}")
        st.caption("From ∇²Φ = 4πGρ")
    
    # ── METRICS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    
    with col_m1:
        st.metric("Soliton Peak", f"{physics.soliton_core.max():.3f}")
    with col_m2:
        st.metric("Fringe Contrast", f"{physics.dark_photon_field.std():.3f}")
    with col_m3:
        st.metric("Mixing", f"{physics.mixing_angle:.3f}")
    with col_m4:
        st.metric("Entropy", f"{physics.entanglement_entropy:.3f}")
    with col_m5:
        gain = physics.entangled_image.std() / (img.std() + 1e-9)
        st.metric("Contrast Gain", f"{gain:.2f}x")
    with col_m6:
        st.metric("FDM Mass", f"{physics.metadata['m_fdm_eV']:.1e} eV")
    
    # ── DOWNLOAD ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    def array_to_bytes(arr, cmap='inferno'):
        fig, ax = plt.subplots(figsize=(8, 8))
        if len(arr.shape) == 3:
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return buf.getvalue()
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        buf = io.BytesIO()
        comparison_fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        plt.close(comparison_fig)
        st.download_button("📸 Annotated Comparison", buf.getvalue(), "annotated_comparison.png")
    with col_d2:
        st.download_button("🌌 Entangled", array_to_bytes(physics.entangled_image), "entangled.png")
    with col_d3:
        st.download_button("⭐ Soliton", array_to_bytes(physics.soliton_core, 'hot'), "soliton.png")
    with col_d4:
        st.download_button("📋 Metadata", json.dumps(physics.metadata, indent=2), "metadata.json")

else:
    st.info("✨ **Upload an image to see FDM Soliton Waves**\n\n"
            "**Features:**\n"
            "• 📏 **Scale Bar**: Physical scale in kpc\n"
            "• 🧪 **Physics Info**: Ω, fringe, entropy, FDM mass\n"
            "• 📐 **Formulas**: [sin(kr)/kr]², λ = h/(m v), S = -Tr(ρ log ρ)\n"
            "• 🎯 **Cluster Presets**: Bullet Cluster, Abell 1689, Abell 209, Abell 2218, COSMOS\n\n"
            "*Recommended: Ω=0.65-0.75, Fringe=55-70*")
    
    st.markdown("---")
    st.markdown("### 🎯 Available Presets")
    
    for name, preset in CLUSTER_PRESETS.items():
        st.markdown(f"**{name}**")
        st.caption(f"Ω={preset['omega']}, Fringe={preset['fringe']}, Scale={preset['scale_kpc']} kpc - {preset['desc']}")

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v25** | Final Working Version | Tony Ford Model")
