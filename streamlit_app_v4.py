import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from PIL import Image
import io

st.set_page_config(page_title="Photon–Dark Photon + Lensing Simulator", layout="wide")
st.title("Photon–Dark Photon Cluster Simulator + Gravitational Lensing")

# ====================== SIDEBAR ======================
st.sidebar.header("Simulation Controls")

cluster = st.sidebar.selectbox(
    "Select Cluster", ["Bullet Cluster", "Abell 1689", "Abell 209"]
)

log_epsilon = st.sidebar.slider("log10(ε)", -12, -5, -6, step=1)
epsilon = 10 ** log_epsilon

log_m = st.sidebar.slider("log10(m_dark [eV])", -14, -10, -12, step=1)
m_dark = 10 ** log_m

E = st.sidebar.slider("Photon Energy (eV)", 1.0, 10000.0, 1000.0, step=10.0)

apply_lensing = st.sidebar.checkbox("Apply Gravitational Lensing", value=True)
kappa_scale = st.sidebar.slider("Lensing Strength (κ scale)", 0.0, 2.0, 0.8, 0.05) if apply_lensing else 0.0

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image (FITS, PNG, JPG, TIFF, etc.)",
    type=["fits", "fit", "png", "jpg", "jpeg", "tif", "tiff"],
    help="Now supports regular images + FITS"
)

show_overlay = st.sidebar.checkbox("Show Overlays (blended view)", value=True)
overlay_alpha = st.sidebar.slider("Overlay Transparency", 0.1, 0.9, 0.5, 0.05) if show_overlay else 0.0

# ====================== 1D SIMULATION (unchanged) ======================
@st.cache_data
def run_1d_simulation(epsilon, m_dark, E, cluster, kappa_scale):
    r = np.linspace(0, 10, 500)
    P = []
    kappa = []
    for x in r:
        ne = (2e-3 if cluster == "Bullet Cluster" else 3e-3 if cluster == "Abell 1689" else 1e-3) * (1 + x**2)**(-1.2)
        omega_p2 = ne * 1e-12
        delta = m_dark**2 - omega_p2
        phase = delta * x / (4 * E)
        P.append(epsilon**2 * np.sin(phase)**2)
        
        if cluster == "Bullet Cluster":
            k = 0.6 * np.exp(-x**2 / 4) + 0.4 * np.exp(-(x-3)**2 / 2)
        else:
            k = 0.8 * np.exp(-x**2 / 6)
        kappa.append(k * kappa_scale)
    return r, np.array(P), np.array(kappa)

r, P_curve, kappa_curve = run_1d_simulation(epsilon, m_dark, E, cluster, kappa_scale)

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(r, P_curve)
    ax.set_title("Conversion Probability")
    ax.set_xlabel("Radius")
    ax.set_ylabel("P(γ → A')")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(r, np.ones_like(r), label="Before")
    ax.plot(r, 1 - P_curve, label="After PDP")
    if apply_lensing:
        ax.plot(r, (1 - P_curve) * (1 + 2*kappa_curve), '--', label="After + Lensing")
    ax.set_title("Photon Intensity")
    ax.set_xlabel("Radius")
    ax.legend()
    st.pyplot(fig)

# ====================== IMAGE LOADING & PROCESSING ======================
if uploaded_file is not None:
    with st.spinner("Processing image..."):
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        # Load image (FITS or regular)
        if file_ext in ['fits', 'fit']:
            with fits.open(uploaded_file) as hdul:
                image = np.nan_to_num(hdul[0].data.astype(float))
        else:
            # Regular image via PIL
            img = Image.open(uploaded_file).convert("L")  # grayscale for astro style
            image = np.array(img, dtype=float)
        
        # Robust normalization to [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-12:
            img_norm = np.zeros_like(image, dtype=float)
        else:
            img_norm = (image - img_min) / (img_max - img_min)
        
        h, w = img_norm.shape
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        radius = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w) * 10
        
        # PDP conversion
        ne = (2e-3 if cluster == "Bullet Cluster" else 3e-3 if cluster == "Abell 1689" else 1e-3) * (1 + radius**2)**(-1.2)
        omega_p2 = ne * 1e-12
        delta = m_dark**2 - omega_p2
        phase = delta * radius / (4 * E)
        P_map = epsilon**2 * np.sin(phase)**2
        
        pdp_image = img_norm * (1 - P_map)
        
        # Lensing
        if apply_lensing and kappa_scale > 0:
            if cluster == "Bullet Cluster":
                r1 = np.sqrt((x - cx)**2 + (y - cy*0.9)**2) / max(w, h) * 10
                r2 = np.sqrt((x - cx*1.6)**2 + (y - cy*1.1)**2) / max(w, h) * 10
                kappa_map = (0.6 * np.exp(-r1**2/4) + 0.4 * np.exp(-r2**2/2)) * kappa_scale
            else:
                r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(w, h) * 10
                kappa_map = 0.8 * np.exp(-r**2/6) * kappa_scale
            magnification = 1 + 2 * kappa_map
            lensed_image = pdp_image * magnification
            lensed_image = np.clip(lensed_image, 0, 1)
        else:
            lensed_image = pdp_image
            kappa_map = np.zeros_like(img_norm)
        
        # ====================== DISPLAY ======================
        st.subheader("Before vs After with Visual Overlays")
        
        # Main side-by-side
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_norm, caption="📸 Original", clamp=True, use_column_width=True)
        with c2:
            st.image(pdp_image, caption="🔄 After PDP Conversion", clamp=True, use_column_width=True)
        with c3:
            st.image(lensed_image, caption="🌌 After PDP + Lensing", clamp=True, use_column_width=True)
        
        # Overlay section
        if show_overlay:
            st.subheader("Overlay Views")
            overlay_col1, overlay_col2 = st.columns(2)
            
            with overlay_col1:
                # PDP overlay on original (red tint where conversion happened)
                overlay_pdp = np.stack([img_norm, img_norm * (1 - P_map*2), img_norm * (1 - P_map*2)], axis=-1)
                st.image(overlay_pdp, caption="Original + PDP Conversion Overlay (red = converted)", clamp=True, use_column_width=True)
            
            with overlay_col2:
                # Lensing convergence as blue overlay
                if apply_lensing:
                    overlay_lens = np.stack([img_norm, img_norm, img_norm + kappa_map*0.8], axis=-1)
                    overlay_lens = np.clip(overlay_lens, 0, 1)
                    st.image(overlay_lens, caption="Original + Lensing Convergence Overlay (blue = mass)", clamp=True, use_column_width=True)
        
        # Detail maps
        st.subheader("Detail Maps")
        d1, d2 = st.columns(2)
        with d1:
            st.image(P_map, caption="Conversion Probability Map (brighter = more conversion)", clamp=True, use_column_width=True)
        with d2:
            if apply_lensing:
                st.image(kappa_map, caption="Convergence κ Map (Dark Matter)", clamp=True, use_column_width=True)

        st.success("✅ Now supports regular images + FITS with nice overlays!")

else:
    st.info("👆 Drag & drop or upload an image (FITS, PNG, JPG, TIFF, etc.) in the sidebar")

st.markdown("### Parameters")
st.write(f"**Cluster:** {cluster} | **ε:** {epsilon:.2e} | **m_dark:** {m_dark:.2e} eV | **Energy:** {E} eV")
