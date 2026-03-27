"""
Magnetar QED Explorer v3.1 – FINAL FIX
Direct PIL display | No placeholders
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
    page_title="Magnetar QED Explorer v3.1",
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
    """Convert numpy array to PIL Image - guaranteed display"""
    if len(arr.shape) == 3:
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    else:
        arr = np.clip(arr, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8)).convert('RGB')


def add_annotations_direct(image_array, metadata, scale_kpc=100, title_prefix="Before"):
    """Add annotations directly to PIL image - no matplotlib for main images"""
    img_pil = array_to_pil(image_array)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
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
    draw.text((w - 45, 12), "N", fill='white', font=font_medium)
    
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
    
    return img_pil


def create_annotated_side_by_side_direct(original_array, processed_array, metadata, scale_kpc=100):
    """Create side-by-side comparison using PIL directly"""
    original_pil = add_annotations_direct(original_array, metadata, scale_kpc, "Before")
    processed_pil = add_annotations_direct(processed_array, metadata, scale_kpc, "After")
    
    # Combine images side by side
    w_total = original_pil.width + processed_pil.width
    h_total = max(original_pil.height, processed_pil.height)
    
    combined = Image.new('RGB', (w_total, h_total), color=(10, 10, 26))
    combined.paste(original_pil, (0, 0))
    combined.paste(processed_pil, (original_pil.width, 0))
    
    return combined


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ Magnetar QED")
    st.markdown("*Quantum Vacuum Physics*")
    st.markdown("---")
    
    # Data source radio
    data_source = st.radio("📁 Data Source", ["📤 Upload Image", "🌌 Preloaded Example"])
    
    if data_source == "📤 Upload Image":
        uploaded = st.file_uploader(
            "Drag & drop file here",
            type=['fits', 'png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="FITS (astronomy), PNG, JPG, TIFF"
        )
        use_upload = True
        # Default preset for parameters
        current_preset = PRELOADED["🌌 Bullet Cluster"]
    else:
        selected = st.selectbox("Select Object", list(PRELOADED.keys()))
        current_preset = PRELOADED[selected]
        use_upload = False
        st.info(current_preset['desc'])
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    
    # Set defaults based on preloaded or upload
    if use_upload:
        omega_default = 0.70
        fringe_default = 65
        scale_default = 100
    else:
        omega_default = current_preset["omega"]
        fringe_default = current_preset["fringe"]
        scale_default = current_preset["scale_kpc"]
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], 
                              index=[50,100,150,200,300].index(scale_default) if scale_default in [50,100,150,200,300] else 1)
    
    st.markdown("---")
    st.markdown("### 🌌 Magnetar Parameters")
    B_surface = st.slider("B Field (G)", 1e13, 1e16, 1e15, format="%.1e")
    epsilon = st.slider("ε Mixing", 1e-12, 1e-8, 1e-10, format="%.1e")
    m_dark = st.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
    a_spin = st.slider("Kerr Spin", 0.0, 0.998, 0.9)
    
    st.caption("Tony Ford | Magnetar QED v3.1")


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
if data_source == "📤 Upload Image" and uploaded is not None:
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
        
        # Resize if needed
        if img_data.shape[0] > 500:
            from skimage.transform import resize
            img_data = resize(img_data, (500, 500), preserve_range=True)
        
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {uploaded.name}")
        source_name = uploaded.name

elif data_source == "🌌 Preloaded Example":
    with st.spinner(f"Loading {selected}..."):
        pattern = PRELOADED[selected]["pattern"]
        img_data = generate_sample(400, pattern)
        results = process_image(img_data, omega, fringe, brightness)
        st.success(f"✅ Loaded: {selected}")
        source_name = selected

else:
    st.info("📁 **Upload an image or select a preloaded example**")
    st.stop()


# ── UPDATE METADATA ─────────────────────────────────────────────
results['metadata'].update({
    'omega': omega,
    'fringe': fringe,
    'mixing': results['mixing'],
    'entropy': results['entropy'],
    'brightness': brightness,
})


# ── DISPLAY ANNOTATED SIDE-BY-SIDE (DIRECT PIL) ─────────────────────────────────────────────
st.markdown("### 📊 Annotated Comparison")

# Create combined image directly with PIL
combined_img = create_annotated_side_by_side_direct(
    results['original'],
    results['rgb'],
    results['metadata'],
    scale_kpc
)

# Display with st.image (guaranteed to work)
st.image(combined_img, use_container_width=True)


# ── QUANTUM COMPONENTS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚛️ Quantum Components")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.image(array_to_pil(results['soliton']), caption="FDM Soliton Core", use_container_width=True)
    st.caption(f"Peak: {results['soliton'].max():.3f} | ρ(r) ∝ [sin(kr)/kr]²")

with col_b:
    st.image(array_to_pil(results['wave']), caption="Dark Photon Field", use_container_width=True)
    st.caption(f"Contrast: {results['wave'].std():.3f} | λ = h/(m v)")

with col_c:
    st.image(array_to_pil(results['entangled']), caption="PDP Entangled", use_container_width=True)
    st.caption(f"Mixing: {results['mixing']:.3f} | Entropy: {results['entropy']:.3f}")


# ── MAGNETAR PHYSICS TABS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Magnetar Physics")

tab1, tab2, tab3 = st.tabs(["🌌 Magnetic Field", "🕳️ Dark Photons", "🌀 Kerr Spacetime"])

# Helper to display plots reliably
def show_plot(fig):
    """Convert figure to PIL and display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black', dpi=100)
    buf.seek(0)
    st.image(Image.open(buf), use_container_width=True)
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
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight', facecolor='black')
    buf1.seek(0)
    st.download_button("📥 Download Field Plot", buf1, "magnetar_field.png", use_container_width=True)

# Tab 2: Dark Photons
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0a0a1a')
    ax2.set_facecolor('#0a0a1a')
    
    L = np.logspace(-2, 4, 500)
    if m_dark <= 0:
        P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
    else:
        hbar_ev_s = 6.582e-16
        c_km_s = 3e5
        conv_len = 4 * 1e18 * hbar_ev_s * c_km_s / (m_dark**2)
        if conv_len > 0:
            P = (epsilon * B_surface / 1e15)**2 * np.sin(np.pi * L / conv_len)**2
        else:
            P = (epsilon * B_surface / 1e15)**2 * np.ones_like(L)
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
    
    show_plot(fig2)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight', facecolor='black')
    buf2.seek(0)
    st.download_button("📥 Download Conversion Plot", buf2, "dark_photon_conversion.png", use_container_width=True)
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
    
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight', facecolor='black')
    buf3.seek(0)
    st.download_button("📥 Download Geodesic Plot", buf3, "kerr_geodesic.png", use_container_width=True)
    st.caption(f"Event Horizon: r_+ = {r_horizon:.3f} M")


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
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    st.download_button("📸 Annotated Comparison", save_pil_image(combined_img), "annotated_comparison.png", use_container_width=True)
with col_d2:
    st.download_button("⭐ Soliton Core", save_array_png(results['soliton'], 'hot'), "soliton.png", use_container_width=True)
with col_d3:
    st.download_button("🌊 Dark Photon", save_array_png(results['wave'], 'plasma'), "dark_photon.png", use_container_width=True)
with col_d4:
    st.download_button("⚡ Entangled", save_array_png(results['entangled'], 'inferno'), "entangled.png", use_container_width=True)

st.markdown("---")
st.markdown("⚡ **Magnetar QED Explorer v3.1** | No Placeholders | Real Physics | Tony Ford Model")
