import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Photon–Dark Photon Cluster Simulator")

# --- Controls ---
cluster = st.selectbox(
    "Select Cluster",
    ["Bullet Cluster", "Abell 1689", "Abell 209"]
)

epsilon = st.slider("Mixing parameter ε", 1e-12, 1e-5, 1e-6, format="%.1e")
m_dark = st.slider("Dark photon mass (eV)", 1e-14, 1e-10, 1e-12, format="%.1e")
E = st.slider("Photon energy (eV)", 1, 1e4, 1000)

# --- Simple density model ---
def ne_profile(r):
    return 1e-3 * (1 + r**2)**(-1.5)

# --- Simulation ---
r = np.linspace(0, 10, 500)
P = []

for x in r:
    ne = ne_profile(x)
    omega_p2 = ne * 1e-12
    delta = m_dark**2 - omega_p2
    phase = delta * x / (4 * E)
    P.append(epsilon**2 * np.sin(phase)**2)

P = np.array(P)

# --- Plot ---
fig, ax = plt.subplots()
ax.plot(r, P)
ax.set_xlabel("Radius")
ax.set_ylabel("Conversion Probability")

st.pyplot(fig)
