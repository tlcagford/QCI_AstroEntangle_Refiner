import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Photon–Dark Photon Cluster Simulator")

# --- Controls ---
cluster = st.selectbox(
    "Select Cluster",
    ["Bullet Cluster", "Abell 1689", "Abell 209"]
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Photon–Dark Photon Cluster Simulator")

# --- Cluster selection ---
cluster = st.selectbox(
    "Select Cluster",
    ["Bullet Cluster", "Abell 1689", "Abell 209"]
)

# --- Log-scale sliders (stable + scientific) ---
log_epsilon = st.slider("log10(ε)", -12, -5, -6)
epsilon = 10**log_epsilon

log_m = st.slider("log10(m_dark [eV])", -14, -10, -12)
m_dark = 10**log_m

E = st.slider("Photon energy (eV)", 1.0, 1e4, 1000.0)

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
fig, ax = plt.subplots()
ax.plot(r, P)
ax.set_xlabel("Radius")
ax.set_ylabel("Conversion Probability")
ax.set_title(f"{cluster}")

st.pyplot(fig)
