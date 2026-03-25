import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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
    "Upload FITS Image", type=["fits", "fit"], help="Limit 200 MB • FITS, FIT"
)

# ====================== 1D SIMULATION ======================
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
        
        # Simple convergence profile
        if cluster == "Bullet Cluster":
            k = 0.6 * np.exp(-x**2 / 4) + 0.4 * np.exp(-(x-3)**2 / 2)
        else:
            k = 0.8 * np.exp(-x**2 / 6)
        kappa.append(k * kappa_scale)
    return r, np.array(P), np.array(kappa)

r, P_curve, kappa_curve = run_1d_simulation(epsilon, m_dark, E, cluster, kappa_scale)

# 1D Plots
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(r, P_curve)
    ax.set_title("Conversion Probability P(γ → A')")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(r, np.ones_like(r), label="Before")
    ax.plot(r, 1 - P_curve, label="After PDP")
    if apply_lensing:
        ax.plot(r, (1 - P_curve) * (1 + 2*kappa_curve), '--', label="After PDP + Lensing")
    ax.set_title("Photon Intensity")
    ax.set_xlabel("Radius")
    ax.legend()
    st.pyplot(fig)

# ====================== IMAGE PROCESSING ======================
if uploaded_file is not None:
    with st.spinner("Applying PDP conversion + Lensing to FITS image..."):
        with fits.open(uploaded_file) as hdul:
            image = np.nan_to_num(hdul[0].data.astype(float))
        
        # Normalize
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-12)
        
        # Simple PDP conversion (using your existing physics style)
        # For demo we apply a radial probability map
        h, w = img_norm.shape
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w/2, h/2
        radius = np.sqrt((x - cx)**2 + (y - cy)**2) / max(h, w) * 10
        
        ne = 2e-3 * (1 + radius**2)**(-1.2) if cluster == "Bullet Cluster" else \
             3e-3 * (1 + radius**2)**(-1.5) if cluster == "Abell 1689" else \
             1e-3 * (1 + radius**2)**(-1.3)
        omega_p2 = ne * 1e-12
        delta = m_dark**2 - omega_p2
        phase = delta * radius / (4 * E)
        P_map = epsilon**2 * np.sin(phase)**2
        
        pdp_image = img_norm * (1 - P_map)   # simple depletion
        
        # Lensing (magnification only for now)
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
            lensed_image = np.clip(lensed_image, 0, None)
        else:
            lensed_image = pdp_image
            kappa_map = np.zeros_like(img_norm)
        
        # Display
        st.subheader("Processed FITS Image")
        c1, c2, c3 = st.columns(3)
        with c1: st.image(img_norm, caption="Original", clamp=True, use_column_width=True)
        with c2: st.image(pdp_image, caption="After PDP Conversion", clamp=True, use_column_width=True)
        with c3: st.image(lensed_image, caption="After PDP + Lensing", clamp=True, use_column_width=True)
        
        if apply_lensing:
            st.image(kappa_map, caption="Convergence κ Map (Dark Matter)", clamp=True, use_column_width=True)

st.success("✅ Running with simplified PDP + Lensing (compatible with your current pdp_physics_working.py)")
