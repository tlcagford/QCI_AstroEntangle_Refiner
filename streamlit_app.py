import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Import the full physics engine
from pdp_physics_working import PhotonDarkPhotonEngine

st.set_page_config(page_title="Photon–Dark Photon Cluster Simulator", layout="wide")
st.title("Photon–Dark Photon Cluster Simulator")

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

# Data Input (exactly as in your screenshot)
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload FITS Image", type=["fits", "fit"], help="Limit 200 MB • FITS, FIT"
)

# ====================== PHYSICS ======================
@st.cache_data
def run_1d_simulation(epsilon, m_dark, E, cluster):
    r = np.linspace(0, 10, 500)
    P = []
    for x in r:
        ne = 2e-3 * (1 + x**2)**(-1.2) if cluster == "Bullet Cluster" else \
             3e-3 * (1 + x**2)**(-1.5) if cluster == "Abell 1689" else \
             1e-3 * (1 + x**2)**(-1.3)
        omega_p2 = ne * 1e-12
        delta = m_dark**2 - omega_p2
        phase = delta * x / (4 * E)
        P.append(epsilon**2 * np.sin(phase)**2)
    return r, np.array(P)

r, P_curve = run_1d_simulation(epsilon, m_dark, E, cluster)

# ====================== LAYOUT ======================
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(r, P_curve)
    ax1.set_title("Conversion Probability")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("P(γ → A')")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(r, np.ones_like(P_curve), label="Before")
    ax2.plot(r, 1 - P_curve, label="After")
    ax2.set_title("Photon Intensity")
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    st.pyplot(fig2)

# ====================== FULL IMAGE PROCESSING ======================
if uploaded_file is not None:
    with st.spinner("Applying full PDP entanglement..."):
        # Load FITS
        with fits.open(uploaded_file) as hdul:
            image = np.nan_to_num(hdul[0].data.astype(float))
        
        # Normalize once
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-12)
        
        # Use the FULL physics engine (fringes + entanglement)
        engine = PhotonDarkPhotonEngine()
        metadata = engine.initialize_from_image(
            image_data=img_norm,
            dark_photon_mass_eV=m_dark,
            mixing_epsilon=epsilon,
            # defaults tuned for Bullet Cluster
            relative_velocity=2000000,
            redshift=0.206,
            distance_mpc=430
        )
        
        processed = engine.get_entanglement_map()
        diff = img_norm - processed
        
        st.subheader("FITS Image Processing – Full PDP Entanglement")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_norm, caption="Before (normalized)", clamp=True)
        with c2:
            st.image(processed, caption="After (with fringes)", clamp=True)
        
        st.image(diff, caption="Difference Map", clamp=True)
        st.image(normalize := (processed - img_norm), caption="Conversion/Fringe Map", clamp=True)
        
        st.caption(f"Entropy: {metadata['entropy']:.4f} bits | Fringe spacing: {metadata['fringe_spacing_px']:.1f} px")

# ====================== PARAMETERS ======================
st.markdown("### Simulation Parameters")
st.write({
    "Cluster": cluster,
    "ε": f"{epsilon:.2e}",
    "m_dark (eV)": f"{m_dark:.2e}",
    "Photon Energy (eV)": E,
})

st.success("✅ Full PDP conversion + image processing now active!")
