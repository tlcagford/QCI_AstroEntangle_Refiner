# QCI AstroEntangle Refiner – v15 PIL DIRECT DISPLAY
# FIXED: Using PIL for direct image display instead of matplotlib

import io
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from astropy.io import fits
from scipy.ndimage import gaussian_filter, sobel, maximum_filter
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI AstroEntangle Refiner v15", page_icon="🔭")

# Light blue interface
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e3f2fd; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #0288d1; }
.stTitle, h1, h2, h3 { color: #01579b; }
[data-testid="stMetricValue"] { color: #01579b; }
</style>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────

def array_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def pil_to_array(img_pil):
    """Convert PIL Image to numpy array"""
    return np.array(img_pil).astype(np.float32) / 255.0


def load_image(uploaded_file):
    """Load image from uploaded file"""
    ext = uploaded_file.name.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    if ext == "fits":
        with fits.open(io.BytesIO(data_bytes)) as h:
            img = h[0].data.astype(np.float32)
            if len(img.shape) > 2:
                img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        return array_to_pil(img)
    else:
        return Image.open(io.BytesIO(data_bytes)).convert("L")


def create_soliton_core(size, fringe):
    """Create FDM soliton core as PIL Image"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w/2), int(h/2)
    
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
    r_s = 0.25 * (50.0 / max(fringe, 1))
    k = np.pi / r_s
    kr = k * r
    
    with np.errstate(divide='ignore', invalid='ignore'):
        soliton = np.where(kr > 1e-6, (np.sin(kr) / kr)**2, 1.0)
    
    soliton = (soliton - soliton.min()) / (soliton.max() - soliton.min())
    soliton = gaussian_filter(soliton, sigma=3)
    
    return array_to_pil(soliton)


def create_fringe_pattern(size, fringe, soliton_array):
    """Create visible dark photon fringe patterns"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w/2), int(h/2)
    
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 15.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    angular = np.sin(k * 3 * theta)
    
    if fringe < 50:
        pattern = radial * 0.6 + spiral * 0.4
    elif fringe < 80:
        pattern = radial * 0.4 + spiral * 0.4 + angular * 0.2
    else:
        pattern = spiral * 0.5 + angular * 0.3 + radial * 0.2
    
    pattern = pattern * soliton_array
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    
    return array_to_pil(pattern)


def apply_pdp_entanglement(img_pil, omega, fringe, brightness=1.2):
    """Apply PDP entanglement with soliton physics"""
    # Convert to array
    img_array = pil_to_array(img_pil)
    h, w = img_array.shape
    
    # Create soliton core
    soliton_array = pil_to_array(create_soliton_core((h, w), fringe))
    
    # Create fringe pattern
    fringe_pil = create_fringe_pattern((h, w), fringe, soliton_array)
    fringe_array = pil_to_array(fringe_pil)
    
    # Create dark matter map
    smoothed = gaussian_filter(img_array, sigma=10)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    dm_array = np.sqrt(grad_x**2 + grad_y**2)
    dm_array = (dm_array - dm_array.min()) / (dm_array.max() - dm_array.min() + 1e-9)
    
    # Add soliton to DM
    dm_array = soliton_array * 0.6 + dm_array * 0.4
    
    # Mix components
    mix = omega * 0.8
    result = img_array * (1 - mix * 0.3)
    result = result + fringe_array * mix * 0.5
    result = result + dm_array * mix * 0.3
    result = result + soliton_array * mix * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    # Create RGB composite
    rgb = np.stack([
        result,
        result * 0.4 + fringe_array * 0.6,
        result * 0.3 + dm_array * 0.7
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    return (array_to_pil(result), 
            array_to_pil(soliton_array), 
            array_to_pil(fringe_array), 
            array_to_pil(dm_array),
            array_to_pil(rgb))


def enhance_image(img_pil):
    """Simple image enhancement"""
    # Sharpen
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_sharp = enhancer.enhance(1.3)
    # Contrast
    enhancer = ImageEnhance.Contrast(img_sharp)
    img_enhanced = enhancer.enhance(1.2)
    return img_enhanced


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v15")
    st.markdown("### Photon-Dark-Photon Entangled FDM")
    st.markdown("*PIL Direct Display*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    st.markdown("### Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    st.markdown("---")
    st.caption("Tony Ford Model | v15 - PIL Direct")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled FDM with Soliton Core Waves*")
st.markdown("---")

if uploaded is not None:
    # Load and process
    with st.spinner("Processing image..."):
        # Load
        img_original = load_image(uploaded)
        
        # Enhance
        img_enhanced = enhance_image(img_original)
        
        # Apply PDP entanglement
        result, soliton, fringe_pattern, dm_map, rgb_composite = apply_pdp_entanglement(
            img_enhanced, omega, fringe, brightness
        )
    
    # Display parameters
    st.success(f"✅ Complete | Ω={omega:.2f} | Fringe={fringe}")
    
    # ── DISPLAY IMAGES (DIRECT PIL) ─────────────────────────────────────────────
    st.markdown("### 📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image(img_original, caption="Original", use_container_width=True)
    
    with col2:
        st.image(img_enhanced, caption="Enhanced", use_container_width=True)
    
    with col3:
        st.image(result, caption="PDP Entangled", use_container_width=True)
    
    with col4:
        st.image(rgb_composite, caption="RGB Composite", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🌌 FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.image(soliton, caption="FDM Soliton Core", use_container_width=True)
        st.caption(f"Peak: {np.max(pil_to_array(soliton)):.3f}")
    
    with col_b:
        st.image(fringe_pattern, caption=f"Dark Photon Field (fringe={fringe})", use_container_width=True)
        st.caption(f"Contrast: {np.std(pil_to_array(fringe_pattern)):.3f}")
    
    with col_c:
        st.image(dm_map, caption="Dark Matter Density", use_container_width=True)
        st.caption(f"Mean Density: {np.mean(pil_to_array(dm_map)):.3f}")
    
    # ── COMPARISON ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before vs After")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.image(img_original, caption="Original", use_container_width=True)
    
    with col_comp2:
        st.image(result, caption=f"PDP Entangled (Ω={omega:.2f}, Fringe={fringe})", use_container_width=True)
    
    # ── METRICS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    # Calculate metrics from arrays
    soliton_array = pil_to_array(soliton)
    fringe_array = pil_to_array(fringe_pattern)
    dm_array = pil_to_array(dm_map)
    result_array = pil_to_array(result)
    original_array = pil_to_array(img_original)
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    with col_m1:
        st.metric("Soliton Peak", f"{soliton_array.max():.3f}")
    
    with col_m2:
        st.metric("Fringe Contrast", f"{fringe_array.std():.3f}")
    
    with col_m3:
        st.metric("DM Mean Density", f"{dm_array.mean():.3f}")
    
    with col_m4:
        mixing = omega * 0.8
        st.metric("PDP Mixing", f"{mixing:.2f}")
    
    with col_m5:
        enhancement = result_array.std() / (original_array.std() + 1e-9)
        st.metric("Contrast Gain", f"{enhancement:.2f}x")
    
    # ── SOLITON PROFILE ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Soliton Radial Profile [sin(kr)/kr]²")
    
    # Generate profile data
    h, w = soliton_array.shape
    cx, cy = w//2, h//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    radii = np.arange(0, min(h, w)//2, 3)
    profile = []
    for rad in radii:
        mask = (r >= rad) & (r < rad + 3)
        if np.any(mask):
            profile.append(np.mean(soliton_array[mask]))
        else:
            profile.append(0)
    
    # Create profile plot (using matplotlib only for this one plot)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(radii[:len(profile)], profile, 'r-', linewidth=2)
    ax.fill_between(radii[:len(profile)], profile, alpha=0.3, color='red')
    ax.set_xlabel("Radius (pixels)", fontsize=11)
    ax.set_ylabel("Soliton Density", fontsize=11)
    ax.set_title("FDM Soliton Profile", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f5f5f5')
    st.pyplot(fig)
    plt.close(fig)
    
    # ── DOWNLOAD ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    # Convert PIL to bytes for download
    def pil_to_bytes(img_pil, format="PNG"):
        buf = io.BytesIO()
        img_pil.save(buf, format=format)
        return buf.getvalue()
    
    with col_d1:
        st.download_button("📸 PDP Image", pil_to_bytes(result), "pdp_result.png")
    
    with col_d2:
        st.download_button("⭐ Soliton Core", pil_to_bytes(soliton), "soliton_core.png")
    
    with col_d3:
        st.download_button("🌊 Fringe Pattern", pil_to_bytes(fringe_pattern), "fringe_pattern.png")
    
    with col_d4:
        st.download_button("🎨 RGB Composite", pil_to_bytes(rgb_composite), "rgb_composite.png")

else:
    st.info("✨ **Upload an image to start**\n\n"
            "This app applies Photon-Dark-Photon Entanglement with FDM Soliton Physics:\n\n"
            "• **FDM Soliton Core**: Ground state of fuzzy dark matter [sin(kr)/kr]²\n"
            "• **Dark Photon Fringes**: Wave interference patterns visible in the field\n"
            "• **Dark Matter Map**: Substructure from gravitational potential\n"
            "• **RGB Composite**: Visual separation of components\n\n"
            "*Recommended: Ω=0.7, Fringe=65 for optimal visibility*")
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Start")
    st.markdown("""
    1. Upload a galaxy cluster image (Bullet Cluster, Abell 1689, etc.)
    2. Adjust Ω to control dark matter visibility
    3. Adjust Fringe to change wave pattern density
    4. Download results as PNG files
    """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v15** | PIL Direct Display | Tony Ford Model")
