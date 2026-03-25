import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Photon–Dark Photon Simulator", layout="wide")
st.title("Photon–Dark Photon Cluster Simulator")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("Simulation Controls")

cluster = st.sidebar.selectbox(
    "Select Cluster",
    ["Bullet Cluster", "Abell 1689", "Abell 209"]
)

# --- SAFE LOG SLIDERS ---
log_epsilon = st.sidebar.slider("log10(ε)", -12, -2, -6)
epsilon = 10 ** log_epsilon

log_m = st.sidebar.slider("log10(m_dark [eV])", -14, -10, -12)
m_dark = 10 ** log_m

E = st.sidebar.slider(
    "Photon Energy (eV)",
    min_value=1.0,
    max_value=10000.0,
    value=1000.0,
    step=1.0
)

# -------------------------------
# FITS Upload
# -------------------------------
st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload FITS Image",
    type=["fits", "fit"]
)

# -------------------------------
# Cluster Density Model
# -------------------------------
def ne_profile(r, cluster):
    if cluster == "Bullet Cluster":
        return 2e-3 * (1 + r**2)**(-1.2)
    elif cluster == "Abell 1689":
        return 3e-3 * (1 + r**2)**(-1.5)
    elif cluster == "Abell 209":
        return 1e-3 * (1 + r**2)**(-1.3)
    return 1e-3

# -------------------------------
# FITS Loader
# -------------------------------
def load_fits(file):
    with fits.open(file) as hdul:
        data = hdul[0].data
    return np.nan_to_num(data)

# -------------------------------
# Normalize Image
# -------------------------------
def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-12)

# -------------------------------
# Conversion Engine (FAST)
# -------------------------------
def apply_conversion(image, epsilon, m_dark, E, cluster):
    h, w = image.shape

    y, x = np.indices((h, w))
    r = np.sqrt((x - w/2)**2 + (y - h/2)**2) / 50.0

    ne = ne_profile(r, cluster)
    omega_p2 = ne * 1e-12

    delta = m_dark**2 - omega_p2
    phase = delta * r / (4 * E)

    P = epsilon**2 * np.sin(phase)**2

    return image * (1 - P), P

# -------------------------------
# Simulation Plot (1D)
# -------------------------------
@st.cache_data
def run_simulation(epsilon, m_dark, E, cluster):
    r = np.linspace(0, 10, 500)
    P = []

    for x in r:
        ne = ne_profile(x, cluster)
        omega_p2 = ne * 1e-12

        delta = m_dark**2 - omega_p2
        phase = delta * x / (4 * E)

        P.append(epsilon**2 * np.sin(phase)**2)

    return r, np.array(P)

r, P_curve = run_simulation(epsilon, m_dark, E, cluster)

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# --- Plot 1 ---
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(r, P_curve)
    ax1.set_title("Conversion Probability")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("P(γ → A')")
    st.pyplot(fig1)

# --- Plot 2 ---
with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(r, np.ones_like(P_curve), label="Before")
    ax2.plot(r, 1 - P_curve, label="After")
    ax2.set_title("Photon Intensity")
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    st.pyplot(fig2)

# -------------------------------
# FITS Processing Section
# -------------------------------
if uploaded_file is not None:
    image = load_fits(uploaded_file)
    processed, P_map = apply_conversion(image, epsilon, m_dark, E, cluster)
    diff = image - processed

    st.subheader("FITS Image Processing")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Before")
        st.image(normalize(image), clamp=True)

    with col4:
        st.subheader("After")
        st.image(normalize(processed), clamp=True)

    st.subheader("Difference Map")
    st.image(normalize(diff), clamp=True)

    st.subheader("Conversion Probability Map")
    st.image(normalize(P_map), clamp=True)

# -------------------------------
# Resonance Check
# -------------------------------
omega_avg = np.mean([ne_profile(x, cluster) for x in r]) * 1e-12

if abs(m_dark**2 - omega_avg) < 1e-24:
    st.warning("⚡ Resonance condition detected!")

# -------------------------------
# Parameters Display
# -------------------------------
st.markdown("### Simulation Parameters")
st.write({
    "Cluster": cluster,
    "epsilon": epsilon,
    "m_dark (eV)": m_dark,
    "Photon Energy (eV)": E
})
