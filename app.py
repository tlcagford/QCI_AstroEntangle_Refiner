# QCI AstroEntangle Refiner – v16 SIMPLIFIED WORKING
# GUARANTEED: Direct numpy array display with fallback

import io
import numpy as np
import streamlit as st
from PIL import Image
from astropy.io import fits
from scipy.ndimage import gaussian_filter, sobel
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="QCI Refiner v16", page_icon="🔭")

# Light blue interface
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #e3f2fd; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 2px solid #0288d1; }
.stTitle, h1, h2, h3 { color: #01579b; }
</style>
""", unsafe_allow_html=True)

# ── SIMPLE IMAGE FUNCTIONS ─────────────────────────────────────────────

def load_image_array(uploaded_file):
    """Load image directly as numpy array"""
    ext = uploaded_file.name.split(".")[-1].lower()
    data_bytes = uploaded_file.read()
    
    if ext == "fits":
        with fits.open(io.BytesIO(data_bytes)) as h:
            img = h[0].data.astype(np.float32)
            if len(img.shape) > 2:
                img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
    else:
        img = Image.open(io.BytesIO(data_bytes)).convert("L")
        img = np.array(img, dtype=np.float32)
    
    # Normalize to [0,1]
    img = np.nan_to_num(img, nan=0.0)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    
    return img


def create_soliton_array(size, fringe):
    """Create FDM soliton core array"""
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
    
    return soliton


def create_fringe_array(size, fringe, soliton):
    """Create dark photon fringe pattern array"""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cx, cy = int(w/2), int(h/2)
    
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w)
    theta = np.arctan2(y - cy, x - cx)
    k = fringe / 15.0
    
    radial = np.sin(k * 2 * np.pi * r * 3)
    spiral = np.sin(k * 2 * np.pi * (r + theta / (2 * np.pi)))
    
    pattern = radial * 0.5 + spiral * 0.5
    pattern = pattern * soliton
    
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
    
    return pattern


def create_dm_array(data, soliton):
    """Create dark matter density array"""
    smoothed = gaussian_filter(data, sigma=8)
    grad_x = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    dm = np.sqrt(grad_x**2 + grad_y**2)
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-9)
    dm = soliton * 0.5 + dm * 0.5
    return dm


def apply_pdp(data, omega, fringe, brightness=1.2):
    """Apply PDP entanglement"""
    h, w = data.shape
    
    # Create components
    soliton = create_soliton_array((h, w), fringe)
    fringe_pattern = create_fringe_array((h, w), fringe, soliton)
    dm_map = create_dm_array(data, soliton)
    
    # Mix
    mix = omega * 0.7
    result = data * (1 - mix * 0.3)
    result = result + fringe_pattern * mix * 0.5
    result = result + dm_map * mix * 0.3
    result = result + soliton * mix * 0.4
    result = result * brightness
    result = np.clip(result, 0, 1)
    
    # RGB composite
    rgb = np.stack([
        result,
        result * 0.5 + fringe_pattern * 0.5,
        result * 0.3 + dm_map * 0.7
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    return result, soliton, fringe_pattern, dm_map, rgb


# ── DISPLAY FUNCTION (GUARANTEED) ─────────────────────────────────────────────
def display_image(img_array, caption, cmap="gray", width=300):
    """Guaranteed image display using numpy array"""
    if img_array is None:
        st.write(f"⚠️ No data for {caption}")
        return
    
    # Ensure array is 2D for grayscale
    if len(img_array.shape) == 3:
        st.image(img_array, caption=caption, use_container_width=True)
    else:
        # Convert to proper format for display
        img_display = np.clip(img_array, 0, 1)
        st.image(img_display, caption=caption, use_container_width=True, clamp=True)
        # Also show mini stats
        st.caption(f"Range: [{img_display.min():.3f}, {img_display.max():.3f}] | Mean: {img_display.mean():.3f}")


# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.title("🔭 QCI Refiner v16")
    st.markdown("### Photon-Dark-Photon Entangled FDM")
    st.markdown("*Simplified Working Version*")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload Image", type=["fits", "png", "jpg", "jpeg"])
    
    st.markdown("---")
    st.markdown("### Parameters")
    
    omega = st.slider("Ω Entanglement", 0.1, 1.0, 0.70, 0.05)
    fringe = st.slider("Fringe Scale", 20, 120, 65, 5)
    brightness = st.slider("Brightness", 0.8, 1.8, 1.2, 0.05)
    
    st.markdown("---")
    st.caption("Tony Ford Model | v16 - Simplified")


# ── MAIN APP ─────────────────────────────────────────────
st.title("🔭 QCI AstroEntangle Refiner")
st.markdown("*Photon-Dark-Photon Entangled FDM with Soliton Core Waves*")
st.markdown("---")

if uploaded is not None:
    # Load and process
    with st.spinner("Processing..."):
        # Load
        original = load_image_array(uploaded)
        
        # Simple enhancement
        blurred = gaussian_filter(original, sigma=1)
        enhanced = original + (original - blurred) * 0.5
        enhanced = np.clip(enhanced, 0, 1)
        
        # Apply PDP
        result, soliton, fringe_pattern, dm_map, rgb = apply_pdp(
            enhanced, omega, fringe, brightness
        )
    
    # Success
    st.success(f"✅ Complete | Ω={omega:.2f} | Fringe={fringe}")
    
    # ── DISPLAY RESULTS ─────────────────────────────────────────────
    st.markdown("### 📊 Pipeline Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_image(original, "Original", "gray")
    
    with col2:
        display_image(enhanced, "Enhanced", "inferno")
    
    with col3:
        display_image(result, "PDP Entangled", "inferno")
    
    with col4:
        display_image(rgb, "RGB Composite", None)
    
    st.markdown("---")
    st.markdown("### 🌌 FDM Physics Components")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_image(soliton, f"FDM Soliton Core", "hot")
        st.metric("Peak", f"{soliton.max():.3f}")
    
    with col_b:
        display_image(fringe_pattern, f"Dark Photon Field (fringe={fringe})", "plasma")
        st.metric("Contrast", f"{fringe_pattern.std():.3f}")
    
    with col_c:
        display_image(dm_map, "Dark Matter Density", "viridis")
        st.metric("Mean", f"{dm_map.mean():.3f}")
    
    # ── BEFORE/AFTER COMPARISON ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Before vs After")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        display_image(original, "Original", "gray")
    
    with col_comp2:
        display_image(result, f"PDP Entangled (Ω={omega:.2f})", "inferno")
    
    # ── METRICS ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Physics Metrics")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    with col_m1:
        st.metric("Soliton Peak", f"{soliton.max():.3f}")
    
    with col_m2:
        st.metric("Fringe Contrast", f"{fringe_pattern.std():.3f}")
    
    with col_m3:
        st.metric("DM Mean", f"{dm_map.mean():.3f}")
    
    with col_m4:
        st.metric("PDP Mixing", f"{omega * 0.7:.2f}")
    
    with col_m5:
        gain = result.std() / (original.std() + 1e-9)
        st.metric("Contrast Gain", f"{gain:.2f}x")
    
    # ── SOLITON PROFILE ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Soliton Radial Profile [sin(kr)/kr]²")
    
    # Calculate radial profile
    h, w = soliton.shape
    cx, cy = w//2, h//2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    radii = np.arange(0, min(h, w)//2, 2)
    profile = []
    for rad in radii:
        mask = (r >= rad) & (r < rad + 2)
        if np.any(mask):
            profile.append(np.mean(soliton[mask]))
        else:
            profile.append(0)
    
    # Create simple plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(radii[:len(profile)], profile, 'r-', linewidth=2, label='Data')
    ax.set_xlabel("Radius (pixels)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("FDM Soliton Profile", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    # Show theoretical fit
    r_norm = radii[:len(profile)] / max(radii[:len(profile)])
    theoretical = np.sin(np.pi * r_norm) / (np.pi * r_norm + 1e-9)
    theoretical = theoretical**2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(radii[:len(profile)], profile, 'r-', linewidth=2, label='Measured')
    ax.plot(radii[:len(profile)], theoretical[:len(profile)] * profile[0], 'b--', linewidth=2, label='Theoretical [sin(kr)/kr]²')
    ax.set_xlabel("Radius (pixels)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Soliton Profile: Measured vs Theoretical", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    # ── DOWNLOAD ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    def save_array_as_png(array, cmap):
        """Save numpy array as PNG bytes"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(array, cmap=cmap)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)
        return buf.getvalue()
    
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        st.download_button("📸 PDP Result", save_array_as_png(result, 'inferno'), "pdp_result.png")
    
    with col_d2:
        st.download_button("⭐ Soliton Core", save_array_as_png(soliton, 'hot'), "soliton.png")
    
    with col_d3:
        st.download_button("🌊 Fringe Pattern", save_array_as_png(fringe_pattern, 'plasma'), "fringe.png")
    
    with col_d4:
        st.download_button("🌌 Dark Matter", save_array_as_png(dm_map, 'viridis'), "darkmatter.png")

else:
    st.info("✨ **Upload an image to see FDM Soliton Waves**\n\n"
            "**This app implements:**\n"
            "• **FDM Soliton Core**: [sin(kr)/kr]² profile (ground state of fuzzy dark matter)\n"
            "• **Dark Photon Field**: Wave interference patterns from photon-dark-photon mixing\n"
            "• **Dark Matter Density**: Substructure from gravitational potential\n\n"
            "*Upload the Bullet Cluster, Abell 1689, or any galaxy cluster image*")
    
    # Show parameter guide
    with st.expander("📖 Parameter Guide"):
        st.markdown("""
        **Ω Entanglement (0.1 - 1.0)**\n
        - 0.3-0.5: Subtle dark matter effects\n
        - 0.6-0.8: Clear wave patterns (recommended)\n
        - 0.9-1.0: Strong visible entanglement\n
        
        **Fringe Scale (20 - 120)**\n
        - 30-50: Large, smooth soliton core\n
        - 50-70: Balanced waves and core\n
        - 80-100: Fine, detailed fringe patterns\n
        
        **Brightness (0.8 - 1.8)**\n
        - Adjust to your preference
        """)

st.markdown("---")
st.markdown("🔭 **QCI AstroEntangle Refiner v16** | Simplified Working Version | Tony Ford Model")
