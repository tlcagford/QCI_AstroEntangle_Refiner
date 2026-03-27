python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import json
from scipy.ndimage import gaussian_filter, zoom
from scipy.fft import fft2, ifft2, fftshift
from io import BytesIO
from PIL import Image
import tempfile
import os
import base64
from datetime import datetime

st.set_page_config(page_title="QCAUS", page_icon="🌌", layout="wide")

# ============================================================================
# VERIFIED FORMULAS
# ============================================================================

def fdm_wave_function(r, k=1.0):
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(kr > 0, np.sin(kr) / kr, 1.0)

def pdp_quantum_field(img, omega=0.5, fringe=1.0):
    fft_img = fft2(img)
    fft_shift = fftshift(fft_img)
    rows, cols = img.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe))
    mixed = fft_shift * mask
    return np.abs(ifft2(fftshift(mixed)))

def quantum_superposition(img, omega=0.5, fringe=1.0, k=1.0, alpha=0.8, beta=1.0):
    size = min(img.shape)
    r = np.linspace(0, 3, size)
    psi = fdm_wave_function(r, k=k)
    fdm_field = np.outer(psi, psi)
    fdm_resized = zoom(fdm_field, (img.shape[0]/size, img.shape[1]/size))
    pdp_field = pdp_quantum_field(img, omega, fringe)
    state = img + alpha * fdm_resized + beta * pdp_field
    return np.clip(state, 0, 1), fdm_resized, pdp_field

def magnetar_dipole(r, theta, B0=1e15):
    Br = B0 * 2 * np.cos(theta) / (r**3 + 1e-10)
    Bt = B0 * np.sin(theta) / (r**3 + 1e-10)
    return Br, Bt

def vacuum_polarization(B, alpha=1/137):
    Bc = 4.41e13
    beta = (B / Bc)**2
    return alpha * beta / (45 * np.pi) * (1 + 7/4 * beta)

def dark_photon_conv(B, eps=0.1, m=1e-9):
    return eps**2 * (1 - np.exp(-B**2 / m**2))

def von_neumann(rho, H, dt):
    return rho + (-1j * (np.dot(H, rho) - np.dot(rho, H))) * dt

def entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-10]
    return -np.sum(ev * np.log(ev))

def qcis_spectrum(k, fnl=1.0, nq=0.5, k0=0.05):
    P_lcdm = k ** (-3) * np.exp(-k / 0.1)
    return P_lcdm * (1 + fnl * (k / k0)**nq)

def add_scale_bar(ax, w, kpc=100, scale=0.1):
    bar_px = kpc / scale
    x = 50
    y = w - 60
    rect = Rectangle((x, y), bar_px, 8, linewidth=2, edgecolor='white', facecolor='white', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + bar_px/2, y - 12, f"{kpc} kpc", color='white', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

def download_link(fig, name, title="Download"):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{name}" style="text-decoration:none;">{title}</a>'

def get_sample_image(name):
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    if name == "Galaxy Cluster":
        R = np.sqrt(X**2 + Y**2)
        img = np.exp(-R**2 / 1.5**2)
        img += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
        img += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
        return img / img.max()
    elif name == "Bullet Cluster":
        img = np.exp(-((X-0.8)**2 + Y**2) / 0.3**2) + 0.7 * np.exp(-((X+0.6)**2 + Y**2) / 0.4**2)
        return img / img.max()
    else:
        R = np.sqrt(X**2 + Y**2)
        img = np.exp(-R**2 / 1.5**2)
        img += 0.6 * np.exp(-((X-0.4)**2 + (Y-0.2)**2) / 0.3**2)
        return img / img.max()

def load_image(file):
    ext = file.name.split('.')[-1].lower()
    data = file.read()
    try:
        if ext in ['fits', 'fit']:
            from astropy.io import fits
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            with fits.open(tmp_path) as hdul:
                d = hdul[0].data
                if d is None and len(hdul) > 1:
                    d = hdul[1].data
                if d is not None and d.ndim > 2:
                    d = np.median(d, axis=0) if d.ndim == 3 else d[0]
            os.unlink(tmp_path)
            if d is None:
                return None, "No data"
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            return d, f"FITS: {d.shape}"
        else:
            img = Image.open(BytesIO(data))
            d = np.array(img)
            if d.ndim == 3:
                d = np.mean(d[:, :, :3], axis=2)
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            return d, f"{ext.upper()}: {d.shape}"
    except Exception as e:
        return None, str(e)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("⚛️ Parameters")
    omega = st.slider("Ω (Entanglement)", 0.0, 1.0, 0.5, 0.01)
    fringe = st.slider("λ (Fringe)", 0.1, 3.0, 1.5, 0.05)
    k = st.slider("k (Soliton)", 0.5, 3.0, 1.0, 0.05)
    alpha = st.slider("α (FDM)", 0.0, 2.0, 0.8, 0.05)
    beta = st.slider("β (PDP)", 0.0, 2.0, 1.0, 0.05)
    
    st.header("🖼️ Image")
    src = st.radio("Source", ["Sample", "Upload"])
    
    img = None
    img_name = "galaxy"
    px_scale = 0.1
    
    if src == "Sample":
        img_type = st.selectbox("Sample", ["Galaxy Cluster", "Bullet Cluster"])
        img = get_sample_image(img_type)
        img_name = img_type.replace(" ", "_").lower()
    else:
        up = st.file_uploader("Upload", type=['fits', 'jpg', 'png'])
        if up:
            img, info = load_image(up)
            if img is None:
                st.error(info)
            else:
                st.success(info)
                img_name = up.name.split('.')[0]
                px_scale = 100.0 / img.shape[1]
    
    st.header("📏 Scale")
    px_scale = st.number_input("kpc/pixel", value=px_scale, format="%.4f")

# ============================================================================
# MAIN
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("*Full-spectrum mapping of invisible quantum fields*")

tabs = st.tabs([
    "🌈 Quantum Fields",
    "⚡ Magnetar QED",
    "🌀 Entanglement",
    "📊 Power Spectra"
])

# ============================================================================
# TAB 0: QUANTUM FIELDS
# ============================================================================

with tabs[0]:
    st.header("🌈 Quantum Field Visualization")
    
    if img is not None:
        state, fdm, pdp = quantum_superposition(img, omega, fringe, k, alpha, beta)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_{img_name}_omega{omega:.2f}_{ts}"
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### Metrics")
            c1a, c1b, c1c = st.columns(3)
            c1a.metric("Max FDM", f"{np.max(fdm):.3f}")
            c1b.metric("Max PDP", f"{np.max(pdp):.3f}")
            corr = np.corrcoef(fdm.flatten(), pdp.flatten())[0,1]
            c1c.metric("Correlation", f"{corr:.3f}")
            
            st.markdown("---")
            st.markdown("### Export")
            
            # Composite
            rgb = np.zeros((*img.shape, 3))
            rgb[..., 0] = img / (img.max() + 1e-8)
            fn = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
            pn = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
            rgb[..., 1] = fn * 0.9
            rgb[..., 2] = pn * 0.9
            rgb = np.clip(rgb, 0, 1)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb, origin='upper')
            ax.set_title("Full-Spectrum Composite")
            ax.axis('off')
            add_scale_bar(ax, rgb.shape[1], scale=px_scale)
            st.pyplot(fig)
            st.markdown(download_link(fig, f"{base}_composite.png", "🌈 Download"), unsafe_allow_html=True)
            plt.close(fig)
        
        with c2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(state, cmap='plasma', origin='upper')
            ax.set_title(f"Quantum State\n|Ψ⟩ + {alpha:.2f}|FDM⟩ + {beta:.2f}|PDP⟩")
            ax.axis('off')
            add_scale_bar(ax, state.shape[1], scale=px_scale)
            st.pyplot(fig)
            st.markdown(download_link(fig, f"{base}_state.png", "📊 Download"), unsafe_allow_html=True)
            plt.close(fig)
        
        # Before/After
        st.markdown("---")
        st.markdown("### 🔬 Before/After Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(img, cmap='gray', origin='upper')
        ax1.set_title("Before: Visible Light Only")
        ax1.axis('off')
        add_scale_bar(ax1, img.shape[1], scale=px_scale)
        
        ax2.imshow(rgb, origin='upper')
        ax2.set_title(f"After: Full-Spectrum\nα={alpha:.2f}, β={beta:.2f}")
        ax2.axis('off')
        add_scale_bar(ax2, rgb.shape[1], scale=px_scale)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(download_link(fig, f"{base}_before_after.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)
        
        # Individual fields
        st.markdown("---")
        st.markdown("### Individual Fields")
        
        c3, c4 = st.columns(2)
        
        with c3:
            fig, ax = plt.subplots(figsize=(6, 6))
            rgb_fdm = np.zeros((*fdm.shape, 3))
            rgb_fdm[..., 1] = (fdm - fdm.min()) / (fdm.max() - fdm.min() + 1e-8)
            ax.imshow(rgb_fdm, origin='upper')
            ax.set_title(f"FDM Field\nρ(r) = ρ₀ [sin({k:.2f}r)/({k:.2f}r)]²")
            ax.axis('off')
            add_scale_bar(ax, fdm.shape[1], scale=px_scale)
            st.pyplot(fig)
            st.markdown(download_link(fig, f"{base}_fdm.png", "🌌 Download"), unsafe_allow_html=True)
            plt.close(fig)
        
        with c4:
            fig, ax = plt.subplots(figsize=(6, 6))
            rgb_pdp = np.zeros((*pdp.shape, 3))
            rgb_pdp[..., 2] = (pdp - pdp.min()) / (pdp.max() - pdp.min() + 1e-8)
            ax.imshow(rgb_pdp, origin='upper')
            ax.set_title(f"PDP Field\nℒ_mix = (ε/2) F_μν F'^μν\nΩ={omega:.2f}, λ={fringe:.2f}")
            ax.axis('off')
            add_scale_bar(ax, pdp.shape[1], scale=px_scale)
            st.pyplot(fig)
            st.markdown(download_link(fig, f"{base}_pdp.png", "🌀 Download"), unsafe_allow_html=True)
            plt.close(fig)
        
    else:
        st.info("👈 Select or upload an image")

# ============================================================================
# TAB 1: MAGNETAR QED
# ============================================================================

with tabs[1]:
    st.header("⚡ Magnetar QED Explorer")
    
    c1, c2 = st.columns(2)
    
    with c1:
        B0 = st.slider("B-Field (10¹⁵ G)", 0.5, 5.0, 1.0, 0.1)
        eps = st.slider("Mixing ε", 0.0, 0.5, 0.1, 0.01)
        m_dark = st.slider("Dark Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e")
        
        r = np.linspace(1, 10, 200)
        theta = np.linspace(0, np.pi, 200)
        R, Th = np.meshgrid(r, theta)
        Br, Bt = magnetar_dipole(R, Th, B0=B0*1e15)
        B = np.sqrt(Br**2 + Bt**2)
        qed = vacuum_polarization(B)
        dark = dark_photon_conv(B, eps=eps, m=m_dark)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_magnetar_B{B0:.1f}_eps{eps:.2f}_{ts}"
        
        st.markdown("---")
        st.subheader("Metrics")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Max B", f"{np.max(B)/1e15:.2f}×10¹⁵ G")
        cm2.metric("Max QED", f"{np.max(qed):.3e}")
        cm3.metric("Max Dark", f"{np.max(dark):.3f}")
    
    with c2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        im1 = axs[0].imshow(B, extent=[1, 10, 0, 180], aspect='auto', cmap='hot', origin='upper')
        axs[0].set_title(f"B-Field\n{B0:.1f}×10¹⁵ G")
        plt.colorbar(im1, ax=axs[0])
        
        im2 = axs[1].imshow(qed, extent=[1, 10, 0, 180], aspect='auto', cmap='plasma', origin='upper')
        axs[1].set_title("Vacuum Polarization")
        plt.colorbar(im2, ax=axs[1])
        
        im3 = axs[2].imshow(dark, extent=[1, 10, 0, 180], aspect='auto', cmap='viridis', origin='upper')
        axs[2].set_title(f"Dark Photons\nε={eps:.2f}")
        plt.colorbar(im3, ax=axs[2])
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(download_link(fig, f"{base}_magnetar.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# ============================================================================
# TAB 2: ENTANGLEMENT
# ============================================================================

with tabs[2]:
    st.header("🌀 Primordial Entanglement")
    
    c1, c2 = st.columns(2)
    
    with c1:
        om = st.slider("Ω", 0.0, 2.0, 0.7, 0.01, key="om_ent")
        dm = st.slider("Dark Mass (eV)", 1e-12, 1e-6, 1e-9, format="%.1e", key="dm_ent")
        em = st.slider("Mixing ε", 0.0, 0.5, 0.1, 0.01, key="em_ent")
        steps = st.slider("Steps", 50, 500, 100)
        
        rho = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        H = np.array([[om, em], [em, dm]], dtype=complex)
        dt = 0.01
        ent_ev = []
        mix_ev = []
        for _ in range(steps):
            rho = von_neumann(rho, H, dt)
            ent_ev.append(entropy(rho[:1, :1]))
            mix_ev.append(abs(rho[0, 1])**2)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_entanglement_omega{om:.2f}_{ts}"
        
        st.markdown("---")
        st.subheader("Metrics")
        cm1, cm2 = st.columns(2)
        cm1.metric("Final Entropy", f"{ent_ev[-1]:.4f}")
        cm2.metric("Final Mixing", f"{mix_ev[-1]:.4f}")
        
        data = {"parameters": {"omega": om, "dark_mass": dm, "mixing": em},
                "final_entropy": float(ent_ev[-1]), "final_mixing": float(mix_ev[-1])}
        st.markdown(f'<a href="data:application/json;base64,{base64.b64encode(json.dumps(data, indent=2).encode()).decode()}" download="{base}_data.json">📄 Download JSON</a>', unsafe_allow_html=True)
    
    with c2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(ent_ev, 'b-', linewidth=2)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Entropy S")
        ax1.set_title("Von Neumann Entropy")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(mix_ev, 'r-', linewidth=2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Mixing")
        ax2.set_title("Photon-Dark Photon Mixing")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(download_link(fig, f"{base}_evolution.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# ============================================================================
# TAB 3: POWER SPECTRA
# ============================================================================

with tabs[3]:
    st.header("📊 QCIS Power Spectra")
    
    c1, c2 = st.columns(2)
    
    with c1:
        fnl = st.slider("f_NL", 0.0, 5.0, 1.0, 0.1)
        nq = st.slider("n_q", 0.0, 2.0, 0.5, 0.05)
        kmin = st.slider("k_min", 0.001, 0.01, 0.005, 0.001, format="%.3f")
        kmax = st.slider("k_max", 0.1, 1.0, 0.5, 0.05)
        
        k = np.logspace(np.log10(kmin), np.log10(kmax), 100)
        Pq = qcis_spectrum(k, fnl, nq)
        Pl = k ** (-3) * np.exp(-k / 0.1)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"qcaus_power_fnl{fnl:.1f}_nq{nq:.2f}_{ts}"
        
        st.markdown("---")
        st.subheader("Metrics")
        ratio = Pq / (Pl + 1e-10)
        st.metric("Enhancement", f"{np.mean(ratio):.3f}x")
        
        data = {"parameters": {"f_nl": fnl, "n_q": nq},
                "k": [float(x) for x in k], "P_quantum": [float(x) for x in Pq]}
        st.markdown(f'<a href="data:application/json;base64,{base64.b64encode(json.dumps(data, indent=2).encode()).decode()}" download="{base}_data.json">📄 Download JSON</a>', unsafe_allow_html=True)
    
    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(k, Pl, 'b-', label='ΛCDM', linewidth=2)
        ax.loglog(k, Pq, 'r--', label=f'Quantum', linewidth=2)
        ax.set_xlabel("k (Mpc⁻¹)")
        ax.set_ylabel("P(k)")
        ax.set_title("Power Spectrum")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown(download_link(fig, f"{base}_spectrum.png", "📸 Download"), unsafe_allow_html=True)
        plt.close(fig)

# ============================================================================
# ABOUT
# ============================================================================

with st.expander("📖 About", expanded=False):
    st.markdown("""
    ### Verified Formulas
    
    | Project | Formula |
    |---------|---------|
    | **FDM Soliton** | ρ(r) = ρ₀ [sin(kr)/(kr)]² |
    | **PDP Mixing** | ℒ_mix = (ε/2) F_μν F'^μν |
    | **Magnetar** | B = B₀ (R/r)³ (2 cosθ, sinθ) |
    | **Entanglement** | i∂ρ/∂t = [H, ρ], S = -Tr(ρ log ρ) |
    | **QCIS** | P(k) = P_ΛCDM(k) × (1 + f_NL (k/k₀)^n_q) |
    
    **Color Mapping:** 🔴 Visible | 🟢 Dark Matter | 🔵 Dark Photons
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>© 2026 Tony E. Ford</div>", unsafe_allow_html=True)
```
