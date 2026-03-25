import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pdp_physics_working import PhotonDarkPhotonEngine

st.set_page_config(page_title="Photon–Dark Photon + Lensing Simulator", layout="wide")
st.title("Photon–Dark Photon Cluster Simulator + Gravitational Lensing")

# ====================== SIDEBAR ======================
st.sidebar.header("Simulation Controls")

cluster = st.sidebar.selectbox(
    "Select Cluster",
    ["Bullet Cluster", "Abell 1689", "Abell 209"]
)

log_epsilon = st.sidebar.slider("log10(ε)", -12, -5, -6)
epsilon = 10 ** log_epsilon

log_m = st.sidebar.slider("log10(m_dark [eV])", -14, -10, -12)
m_dark = 10 ** log_m

E = st.sidebar.slider("Photon Energy (eV)", 1.0, 10000.0, 1000.0, step=1.0)

apply_lensing = st.sidebar.checkbox("Apply Gravitational Lensing", value=True)
kappa_scale = st.sidebar.slider("Lensing Strength (κ scale)", 0.0, 2.0, 0.8, 0.05) if apply_lensing else 0.0

# Data Input
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload FITS Image", type=["fits", "fit"], help="Limit 200 MB • FITS, FIT"
)

# ====================== PHYSICS ======================
@st.cache_data
def run_1d_simulation(epsilon, m_dark, E, cluster, kappa_scale):
    r = np.linspace(0, 10, 500)
    P = []
    kappa = []
    for x in r:
        # Simple plasma profile
        ne = 2e-3 * (1 + x**2)**(-1.2) if cluster == "Bullet Cluster" else \
             3e-3 * (1 + x**2)**(-1.5) if cluster == "Abell 1689" else \
             1e-3 * (1 + x**2)**(-1.3)
        omega_p2 = ne * 1e-12
        delta = m_dark**2 - omega_p2
        phase = delta * x / (4 * E)
        P.append(epsilon**2 * np.sin(phase)**2)
        
        # Approximate convergence (higher in center, offset for Bullet)
        if cluster == "Bullet Cluster":
            k = 0.6 * np.exp(-x**2 / 4) + 0.4 * np.exp(-(x-3)**2 / 2)  # two peaks
        else:
            k = 0.8 * np.exp(-x**2 / 6)
        kappa.append(k * kappa_scale)
    return r, np.array(P), np.array(kappa)

r, P_curve, kappa_curve = run_1d_simulation(epsilon, m_dark, E, cluster, kappa_scale)

# ====================== LAYOUT: 1D Plots ======================
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(r, P_curve, label="PDP Conversion")
    ax1.set_title("Conversion Probability")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("P(γ → A')")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(r, np.ones_like(P_curve), label="Before")
    ax2.plot(r, 1 - P_curve, label="After (PDP only)")
    if apply_lensing:
        ax2.plot(r, (1 - P_curve) * (1 + 2 * kappa_curve), label="After (PDP + Lensing)", linestyle="--")
    ax2.set_title("Photon Intensity")
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    st.pyplot(fig2)

# ====================== FULL IMAGE PROCESSING ======================
if uploaded_file is not None:
    with st.spinner("Processing: PDP Entanglement + Gravitational Lensing..."):
        # Load FITS
        with fits.open(uploaded_file) as hdul:
            image = np.nan_to_num(hdul[0].data.astype(float))
        
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-12)
        height, width = img_norm.shape
        
        # 1. Full PDP Engine (fringes + entanglement)
        engine = PhotonDarkPhotonEngine()
        metadata = engine.initialize_from_image(
            image_data=img_norm,
            dark_photon_mass_eV=m_dark,
            mixing_epsilon=epsilon,
            relative_velocity=2000000,
            redshift=0.206,
            distance_mpc=430
        )
        pdp_image = engine.get_entanglement_map()
        
        # 2. Apply Gravitational Lensing (simple weak lensing approximation)
        if apply_lensing and kappa_scale > 0:
            # Create coordinate grid
            y, x = np.mgrid[0:height, 0:width]
            cx, cy = width/2, height/2
            
            # Bullet Cluster: two offset mass peaks (main + subcluster)
            if cluster == "Bullet Cluster":
                r1 = np.sqrt((x - cx)**2 + (y - cy*0.9)**2) / max(width, height) * 10
                r2 = np.sqrt((x - cx*1.6)**2 + (y - cy*1.1)**2) / max(width, height) * 10
                kappa_map = 0.6 * np.exp(-r1**2 / 4) + 0.4 * np.exp(-r2**2 / 2)
            else:
                r = np.sqrt((x - cx)**2 + (y - cy)**2) / max(width, height) * 10
                kappa_map = 0.8 * np.exp(-r**2 / 6)
            
            kappa_map *= kappa_scale
            magnification = 1 + 2 * kappa_map  # weak lensing approx (μ ≈ 1 + 2κ)
            
            # Simple magnification: stretch intensity (for demo; real lensing would remap pixels)
            lensed_image = pdp_image * magnification
            lensed_image = np.clip(lensed_image, 0, 1)
            
            # Shear visualization (simple tangential shear for demo)
            shear_map = kappa_map * 0.3  # fake shear strength
        else:
            lensed_image = pdp_image
            kappa_map = np.zeros_like(img_norm)
            shear_map = np.zeros_like(img_norm)
            magnification = np.ones_like(img_norm)
        
        diff = img_norm - pdp_image
        
        # ====================== DISPLAY ======================
        st.subheader("FITS Image: PDP Entanglement + Gravitational Lensing")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_norm, caption="Original (normalized)", clamp=True)
        with c2:
            st.image(pdp_image, caption="After PDP Entanglement", clamp=True)
        with c3:
            st.image(lensed_image, caption="After PDP + Lensing", clamp=True)
        
        st.image(diff, caption="PDP Difference Map", clamp=True)
        
        if apply_lensing:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(kappa_map, caption="Convergence (κ) Map – Dark Matter Distribution", clamp=True)
            with col_b:
                st.image(magnification, caption="Magnification Map (μ ≈ 1 + 2κ)", clamp=True)
        
        st.caption(f"Entropy: {metadata['entropy']:.4f} bits | Fringe spacing: {metadata['fringe_spacing_px']:.1f} px | "
                   f"Lensing κ max: {kappa_map.max():.3f}")

# ====================== PARAMETERS ======================
st.markdown("### Simulation Parameters")
st.write({
    "Cluster": cluster,
    "ε": f"{epsilon:.2e}",
    "m_dark (eV)": f"{m_dark:.2e}",
    "Photon Energy (eV)": E,
    "Lensing Applied": apply_lensing,
    "Lensing Strength": f"{kappa_scale:.2f}"
})

st.success("✅ Full PDP conversion + Gravitational Lensing now active!")
