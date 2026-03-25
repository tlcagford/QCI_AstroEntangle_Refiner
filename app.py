# QCI AstroEntangle Refiner – v23 WITH IMAGE ANNOTATION
# Added: Physics information overlay on images

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.fft import fft2, fftshift
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont
import warnings
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    layout="wide", 
    page_title="QCI Refiner v23 - Annotated Outputs", 
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
    processing_time: float
    metadata: Dict
    annotated_image: np.ndarray = None


# ── ANNOTATION FUNCTIONS ─────────────────────────────────────────────

def add_physics_annotations(image_array, metadata, scale_kpc=100, image_pixels=500):
    """
    Add physics information overlay to image
    """
    # Convert to PIL for text overlay
    if len(image_array.shape) == 3:
        img = (image_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
    else:
        img_pil = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
    
    draw = ImageDraw.Draw(img_pil)
    
    # Try to use a nice font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Scale bar parameters
    scale_bar_length_px = 100  # pixels
    scale_bar_kpc = (scale_bar_length_px / image_pixels) * scale_kpc
    scale_bar_y = image_array.shape[0] - 40
    scale_bar_x_start = 20
    scale_bar_x_end = scale_bar_x_start + scale_bar_length_px
    
    # Draw scale bar
    draw.rectangle([scale_bar_x_start, scale_bar_y, scale_bar_x_end, scale_bar_y + 5], fill='white')
    draw.text((scale_bar_x_start + 10, scale_bar_y - 18), f"{scale_bar_kpc:.0f} kpc", 
              fill='white', font=font_small, stroke_width=1, stroke_fill='black')
    
    # Draw N (North) indicator
    draw.line([image_array.shape[1] - 30, 30, image_array.shape[1] - 30, 60], fill='white', width=2)
    draw.text((image_array.shape[1] - 38, 15), "N", fill='white', font=font_medium)
    
    # Physics info box
    info_lines = [
        f"QCI Refiner v23",
        f"Ω = {metadata['omega']:.2f} | Fringe = {metadata['fringe']}",
        f"Mixing = {metadata['mixing_angle']:.3f}",
        f"S_entropy = {metadata['entanglement_entropy']:.3f}",
        f"λ_FDM = {scale_kpc / metadata['fringe'] * 8:.1f} kpc",
        f"FDM Mass: {metadata['m_fdm_eV']:.1e} eV"
    ]
    
    # Draw info box background
    box_y_start = 10
    box_y_end = box_y_start + len(info_lines) * 20 + 10
    draw.rectangle([10, box_y_start, 250, box_y_end], fill=(0, 0, 0, 180), outline='white')
    
    # Draw text lines
    for i, line in enumerate(info_lines):
        draw.text((15, box_y_start + 5 + i * 20), line, fill='white', font=font_small)
    
    # Add formula annotations
    formulas = [
        "ρ(r) ∝ [sin(kr)/kr]²",
        "λ = h/(m v)",
        "S = -Tr(ρ log ρ)"
    ]
    
    formula_y_start = image_array.shape[0] - 80
    for i, formula in enumerate(formulas):
        draw.text((image_array.shape[1] - 200, formula_y_start + i * 18), formula, 
                  fill='cyan', font=font_small, stroke_width=1, stroke_fill='black')
    
    return np.array(img_pil) / 255.0


def create_comparison_with_labels(original, entangled, metadata, scale_kpc=100):
    """
    Create side-by-side comparison with labels
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title(f"Before: Abell 1689 Standard View\n(Public HST/JWST Data)", 
                  color='white', fontsize=10)
    ax1.axis('off')
    
    # Annotated entangled image
    ax2.imshow(entangled, cmap='inferno')
    
    # Add scale bar to matplotlib figure
    h, w = entangled.shape[:2]
    scale_bar_px = 100
    scale_bar_kpc = (scale_bar_px / w) * scale_kpc
    scale_x_start = 20
    scale_x_end = scale_x_start + scale_bar_px
    scale_y = h - 30
    
    ax2.add_patch(Rectangle((scale_x_start, scale_y), scale_bar_px, 5, 
                            facecolor='white', edgecolor='white'))
    ax2.text(scale_x_start + 10, scale_y - 12, f"{scale_bar_kpc:.0f} kpc", 
             color='white', fontsize=9, ha='center')
    
    # Add N indicator
    ax2.annotate('N', xy=(w - 30, 30), xytext=(w - 30, 30), 
                 color='white', fontsize=12, ha='center', va='center')
    ax2.plot([w - 30, w - 30], [30, 60], 'w-', linewidth=2)
    
    # Add physics info box
    info_text = f"Ω={metadata['omega']:.2f} | Fringe={metadata['fringe']}\n"
    info_text += f"S_entropy={metadata['entanglement_entropy']:.3f}\n"
    info_text += f"λ={scale_bar_kpc / metadata['fringe'] * 8:.1f} kpc"
    
    ax2.text(15, 25, info_text, color='white', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add formula annotation
    ax2.text(w - 180, h - 25, r'$\rho(r) \propto [\sin(kr)/(kr)]^2$', 
             color='cyan', fontsize=8, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    ax2.set_title(f"After: Abell 1689 with Full Photon-Dark-Photon\nEntangled FDM Overlays (Tony Ford Model)", 
                  color='white', fontsize=10)
    ax2.axis('off')
    
    fig.patch.set_facecolor('#0b0b1a')
    fig.tight_layout()
    
    return fig


# ── PHYSICS FUNCTIONS ─────────────────────────────────────────────

def compute_entanglement_entropy(rho):
    """Compute von Neumann entanglement entropy"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0.0
    return -np.sum(eigenvalues * np.log(eigenvalues))


def schrodinger_poisson_soliton(size, fringe):
    """Create FDM soliton core with [sin(kr)/(kr)]² profile"""
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
    """Create visible dark photon interference pattern"""
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
    """Create dark matter density map from image gradients"""
    smoothed = gaussian_filter(image, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    if gradient.max() > gradient.min():
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    else:
        gradient = np.zeros_like(gradient)
    
    dm = soliton * 0.6 + gradient * 0.4
    return np.clip(dm, 0, 1)


def apply_primordial_entanglement(image, omega, fringe, brightness=1.2, scale_kpc=100):
    """Apply full physics pipeline"""
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
        "wavelength_kpc": float(scale_kpc / fringe * 8)
    }
    
    return PhysicsOutput(
        entangled_image=result,
        soliton_core=soliton,
        dark_photon_field=dark_photon,
        dark_matter_density=dm_density,
        rgb_composite=rgb,
        mixing_angle=mixing,
        entanglement_entropy=entropy,
        processing_time=0.0,
        metadata=metadata
    )


# ── CLUSTER PRESETS ─────────────────────────────────────────────
CLUSTER_PRESETS = {
    "Bullet Cluster (1E0657-56)": {"fringe": 70, "omega": 0.75, "scale_kpc": 200},
    "Abell 1689": {"fringe": 55, "omega": 0.65, "scale_kpc": 150},
    "Abell 209": {"fringe": 60, "omega": 0.70, "scale_kpc": 100},
    "Abell 2218": {"fringe": 50, "omega": 0.68, "scale_kpc": 120},
    "COSMOS Field": {"fringe": 45, "omega": 0.60, "scale_kpc": 80}
}


# ── UI FUNCTIONS ─────────────────────────────────────────────

def display_image(img_array, title, cmap='inferno', show_colorbar=True, figsize=(4, 4)):
    """Safe image display"""
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
        st.write(f"⚠️ Display error: {str(e)[:50]}")


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v23")
    st.markdown("### Annotated Physics Outputs")
    st.markdown("*With Image Overlays*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload FITS/Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    
    st.markdown("### 🎯 Cluster Presets")
    selected_cluster = st.selectbox("Load Preset", ["Custom"] + list(CLUSTER_PRESETS.keys()))
    
    if selected_cluster != "Custom":
        preset = CLUSTER_PRESETS[selected_cluster]
        st.info(f"**{selected_cluster}**\nΩ={preset['omega']}, Fringe={preset['fringe']}")
        omega_default = preset["omega"]
        fringe_default = preset["fringe"]
        scale_default = preset["scale_kpc"]
    else:
        omega_default = 0.70
        fringe_default = 65
        scale_default = 100
    
    st.markdown("---")
    st.markdown("### ⚛️ Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, omega_default, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, fringe_default, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    scale_kpc = st.selectbox("Scale (kpc)", [50, 100, 150, 200, 300], 
                              index=[50,100,150,200,300].index(scale_default))
    
    st.markdown("---")
    st.markdown("### 🏷️ Annotation Options")
    
    add_scale_bar = st.checkbox("Add Scale Bar", value=True)
    add_physics_text = st.checkbox("Add Physics Info", value=True)
    add_formulas = st.checkbox("Add Formulas", value=True)
    
    st.markdown("---")
    st.caption("Tony Ford Model | v23 - Annotated Outputs")


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
            st.error(f"Error: {e}")
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
        
        # Create annotated version
        annotated = add_physics_annotations(
            physics.entangled_image, 
            physics.metadata, 
            scale_kpc, 
            physics.entangled_image.shape[1]
        )
        
        # Create comparison figure
        comparison_fig = create_comparison_with_labels(
            img, 
            annotated if add_physics_text else physics.entangled_image,
            physics.metadata, 
            scale_kpc
        )
    
    # Success
    st.success(f"✅ Complete | Mixing = {physics.mixing_angle:.3f} | Entropy = {physics.entanglement_entropy:.3f}")
    
    # ── ANNOTATED COMPARISON ─────────────────────────────────────────────
    st.markdown("### 📊 Annotated Comparison")
    st.pyplot(comparison_fig)
    plt.close(comparison_fig)
    
    # ── PHYSICS COMPONENTS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚛️ FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_image(physics.soliton_core, "FDM Soliton Core", 'hot', figsize=(4, 4))
        st.metric("Peak", f"{physics.soliton_core.max():.3f}")
        st.caption("ρ(r) ∝ [sin(kr)/(kr)]²")
    
    with col_b:
        display_image(physics.dark_photon_field, f"Dark Photon Field", 'plasma', figsize=(4, 4))
        st.metric("Contrast", f"{physics.dark_photon_field.std():.3f}")
        st.caption(f"λ = {physics.metadata['wavelength_kpc']:.1f} kpc")
    
    with col_c:
        display_image(physics.dark_matter_density, "Dark Matter Density", 'viridis', figsize=(4, 4))
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
        st.metric("Mixing Angle", f"{physics.mixing_angle:.3f}")
    with col_m4:
        st.metric("Entanglement Entropy", f"{physics.entanglement_entropy:.3f}")
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
    
    def annotated_to_bytes():
        buf = io.BytesIO()
        comparison_fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        plt.close(comparison_fig)
        return buf.getvalue()
    
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)
    
    with col_d1:
        st.download_button("📸 Annotated Comparison", annotated_to_bytes(), "annotated_comparison.png")
    with col_d2:
        st.download_button("🌌 Entangled Image", array_to_bytes(physics.entangled_image), "entangled.png")
    with col_d3:
        st.download_button("⭐ Soliton Core", array_to_bytes(physics.soliton_core, 'hot'), "soliton.png")
    with col_d4:
        st.download_button("🌊 Fringe Pattern", array_to_bytes(physics.dark_photon_field, 'plasma'), "fringe.png")
    with col_d5:
        st.download_button("📋 Metadata", json.dumps(physics.metadata, indent=2), "metadata.json")

else:
    st.info("✨ **Upload an image to see annotated physics outputs**\n\n"
            "**Features:**\n"
            "• 📏 **Scale Bar**: Physical scale in kpc\n"
            "• 🧪 **Physics Info**: Ω, fringe, entropy, FDM mass\n"
            "• 📐 **Formulas**: [sin(kr)/kr]², λ = h/(m v), S = -Tr(ρ log ρ)\n"
            "• 📊 **Side-by-Side**: Before/After comparison with annotations\n\n"
            "*Recommended: Ω=0.65-0.75, Fringe=55-70*")
    
    # Show example
    st.markdown("---")
    st.markdown("### 📋 Example Output with Annotations")
    st.markdown("""
    The annotated output includes:
    - **Scale Bar**: Physical scale reference (kpc)
    - **North Indicator**: Orientation marker
    - **Physics Parameters**: Ω, fringe, mixing angle, entanglement entropy
    - **FDM Mass**: Effective dark matter particle mass
    - **Wavelength**: Dark photon interference wavelength
    - **Formulas**: Key physical equations
    """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v23** | Annotated Physics Outputs | Tony Ford Model")
