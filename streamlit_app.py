import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="QCAUS", layout="wide")

# =========================
# HEADER
# =========================
st.title("QCAUS: Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("Interactive Multi-Scale Quantum Astrophysics Framework")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Global Controls")

mode = st.sidebar.radio("Mode", ["Physics Mode", "Exploration Mode"])

target = st.sidebar.selectbox(
    "Target",
    ["Crab Nebula", "Swift J1818.0-1607", "Custom Upload"]
)

st.sidebar.subheader("Core Parameters")

B_field = st.sidebar.slider("Magnetic Field (G)", 1e13, 1e16, 1e15, format="%.1e")
epsilon = st.sidebar.slider("Mixing ε", 1e-12, 1e-8, 1e-10, format="%.1e")
m_dark = st.sidebar.slider("Dark Photon Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
m_fdm = st.sidebar.slider("FDM Mass (eV)", 1e-23, 1e-20, 1e-22, format="%.1e")
scale = st.sidebar.slider("Scale (kpc)", 1, 100, 10)
a_star = st.sidebar.slider("Kerr Spin a*", 0.0, 1.0, 0.9)
omega = st.sidebar.slider("Photon Energy ω (arb)", 0.1, 10.0, 1.0)
L = st.sidebar.slider("Path Length L (arb)", 0.1, 10.0, 1.0)

show_overlays = (mode == "Exploration Mode")

# =========================
# MODEL LABEL
# =========================
def model_badge(label):
    if label == "established":
        st.success("✅ Established Physics")
    elif label == "approx":
        st.warning("⚠️ Approximation")
    else:
        st.info("🧠 Hypothesis / Extension")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Magnetar QED",
    "🌌 FDM Soliton",
    "🔄 Photon–Dark Photon",
    "🧪 Visualization Lab",
    "📈 Validation",
    "🌠 QCIS Cosmology"
])

# =========================
# 1. MAGNETAR QED
# =========================
with tab1:
    st.header("Magnetar QED Engine")
    model_badge("established")

    r = np.linspace(1, 20, 500)
    B = B_field / r**3

    Bcrit = 4.41e13
    ratio = B_field / Bcrit

    fig, ax = plt.subplots()
    ax.plot(r, B)
    ax.set_title("Dipole Field Scaling")
    ax.set_xlabel("Radius")
    ax.set_ylabel("B(r)")

    st.pyplot(fig)

    st.metric("B / Bcrit", f"{ratio:.2e}")

# =========================
# 2. FDM SOLITON
# =========================
with tab2:
    st.header("FDM Soliton Core")
    model_badge("approx")

    r = np.linspace(0.1, 20, 500)
    k = np.sqrt(m_fdm) / scale
    rho = (np.sin(k * r) / (k * r))**2

    fig, ax = plt.subplots()
    ax.plot(r, rho)
    ax.set_title("Soliton Density Profile")
    ax.set_xlabel("Radius")
    ax.set_ylabel("Density")

    st.pyplot(fig)

# =========================
# 3. PDP MIXING
# =========================
with tab3:
    st.header("Photon–Dark Photon Mixing")
    model_badge("hypothesis")

    x = np.linspace(0, 10, 500)

    P = (epsilon * B_field / m_dark)**2 * np.sin((m_dark * L) / (4 * omega))**2

    fig, ax = plt.subplots()
    ax.plot(x, P)
    ax.set_title("Conversion Probability")
    ax.set_xlabel("Distance")
    ax.set_ylabel("P(γ → A')")

    st.pyplot(fig)

# =========================
# 4. VISUALIZATION LAB
# =========================
with tab4:
    st.header("Visualization Lab")
    st.warning("⚠️ Model-dependent visualization (not observational evidence)")

    if show_overlays:
        size = 300
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)

        r_grid = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        fringe = np.sin(scale * r_grid)
        soliton = np.exp(-r_grid**2)

        kerr_mod = 1 + a_star * np.sin(theta)

        combined = fringe * soliton * kerr_mod

        fig, ax = plt.subplots()
        ax.imshow(combined, cmap='inferno')
        ax.set_title("FDM + PDP + Kerr Composite")

        st.pyplot(fig)
    else:
        st.info("Switch to Exploration Mode")

# =========================
# 5. VALIDATION
# =========================
with tab5:
    st.header("Parameter Sensitivity")

    eps_vals = np.logspace(-12, -8, 50)
    mix = (eps_vals * B_field / m_dark)**2

    fig, ax = plt.subplots()
    ax.plot(eps_vals, mix)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sensitivity to ε")

    st.pyplot(fig)

# =========================
# 6. QCIS COSMOLOGY
# =========================
with tab6:
    st.header("Quantum Cosmology Integration Suite")
    model_badge("approx")

    k_vals = np.logspace(-3, 1, 300)
    f_nl = 0.05
    k0 = 0.05

    P_lcdm = k_vals**(-3)
    P_quantum = P_lcdm * (1 + f_nl * (k_vals / k0)**0.1)

    fig, ax = plt.subplots()
    ax.loglog(k_vals, P_lcdm, label="ΛCDM")
    ax.loglog(k_vals, P_quantum, label="Quantum Corrected")

    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")

    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
QCAUS integrates:
• Magnetar QED  
• FDM Soliton Physics  
• Photon–Dark Photon Coupling  
• Quantum Cosmology  

Physics Mode: validated models  
Exploration Mode: extended theoretical visualization  
""")
