import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Title
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
log_epsilon = st.sidebar.slider("log10(ε)", -12, -5, -6)
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
# Cluster Density Profiles
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
# Simulation (Cached)
# -------------------------------
@st.cache_data
def run_simulation(epsilon, m_dark, E, cluster):
    r = np.linspace(0, 10, 500)
    P = []

    for x in r:
        ne = ne_profile(x, cluster)
        omega_p2 = ne * 1e-12  # simplified plasma scaling

        delta = m_dark**2 - omega_p2
        phase = delta * x / (4 * E)

        P.append(epsilon**2 * np.sin(phase)**2)

    return r, np.array(P)


# Run simulation
r, P = run_simulation(epsilon, m_dark, E, cluster)

# -------------------------------
# BEFORE vs AFTER
# -------------------------------
before = np.ones_like(P)
after = 1 - P

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# --- Plot 1: Conversion Probability ---
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(r, P)
    ax1.set_title("Conversion Probability")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("P(γ → A')")
    st.pyplot(fig1)

# --- Plot 2: Before vs After ---
with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(r, before, label="Before")
    ax2.plot(r, after, label="After")
    ax2.set_title("Photon Intensity")
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    st.pyplot(fig2)

# -------------------------------
# Resonance Indicator
# -------------------------------
omega_avg = np.mean([ne_profile(x, cluster) for x in r]) * 1e-12

if abs(m_dark**2 - omega_avg) < 1e-24:
    st.warning("⚡ Resonance condition detected!")

# -------------------------------
# Info Panel
# -------------------------------
st.markdown("### Simulation Parameters")
st.write({
    "Cluster": cluster,
    "epsilon": epsilon,
    "m_dark (eV)": m_dark,
    "Photon Energy (eV)": E
})
