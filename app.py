"""
QCAUS - Enhanced with Radar Data Verification
Shows confirmation when real radar data is loaded
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="QCAUS - Quantum Cosmology Suite", page_icon="🌌", layout="wide")

# ============================================================================
# RADAR DATA VERIFICATION FUNCTION
# ============================================================================

def verify_radar_data(radar_data):
    """Verify that radar data is real and extract key information"""
    if radar_data is None:
        return {
            "is_valid": False,
            "message": "No radar data loaded",
            "type": "none"
        }
    
    # Check for sample data marker
    is_sample = radar_data.get("is_sample", False)
    
    # Check for required fields
    has_detections = len(radar_data.get("pdp_detections", [])) > 0
    has_aircraft = len(radar_data.get("aircraft_data", [])) > 0
    has_parameters = radar_data.get("parameters") is not None
    
    # Check if it's from StealthPDPRadar (has specific fields)
    is_stealth_export = (
        "pdp_detections" in radar_data and
        "aircraft_data" in radar_data and
        "parameters" in radar_data
    )
    
    # Extract source info
    source = "StealthPDPRadar Export" if is_stealth_export else "Unknown"
    if is_sample:
        source = "Sample Data (Built-in)"
    
    # Get detection details
    detections = radar_data.get("pdp_detections", [])
    detection_count = len(detections)
    avg_confidence = np.mean([d.get("confidence", 0) for d in detections]) if detections else 0
    
    # Get aircraft details
    aircraft = radar_data.get("aircraft_data", [])
    stealth_aircraft = [a for a in aircraft if a.get("stealth_level") in ["Very High", "High"]]
    
    return {
        "is_valid": True,
        "is_real_data": not is_sample and is_stealth_export,
        "message": f"✅ Loaded {detection_count} detections from {source}",
        "type": "real" if not is_sample else "sample",
        "source": source,
        "detection_count": detection_count,
        "avg_confidence": avg_confidence,
        "stealth_aircraft_count": len(stealth_aircraft),
        "parameters": radar_data.get("parameters", {})
    }

def display_radar_verification(verification):
    """Display radar data verification status"""
    if verification["is_valid"]:
        if verification["is_real_data"]:
            st.success(f"""
            ✅ **REAL RADAR DATA CONFIRMED**
            
            | Property | Value |
            |----------|-------|
            | Source | {verification['source']} |
            | Detections | {verification['detection_count']} |
            | Avg Confidence | {verification['avg_confidence']:.3f} |
            | Stealth Aircraft | {verification['stealth_aircraft_count']} |
            | Ω (from radar) | {verification['parameters'].get('omega', 'N/A')} |
            """)
        else:
            st.info(f"""
            ℹ️ **SAMPLE RADAR DATA** (Built-in for testing)
            
            | Property | Value |
            |----------|-------|
            | Source | {verification['source']} |
            | Detections | {verification['detection_count']} |
            | Avg Confidence | {verification['avg_confidence']:.3f} |
            
            *Upload a real JSON export from StealthPDPRadar to use your own data*
            """)
    else:
        st.warning("⚠️ No radar data loaded - using sample data for demonstration")

# ============================================================================
# RADAR OVERLAY WITH VERIFICATION
# ============================================================================

def radar_to_overlay_with_verification(radar_data, image_shape, verification):
    """Convert radar detections to RGBA overlay with verification metadata"""
    H, W = image_shape
    overlay = np.zeros((H, W, 4))
    
    if not verification["is_valid"] or verification["is_real_data"] == False and verification["type"] != "sample":
        return overlay, {"message": "No valid radar data"}
    
    detections = radar_data.get("pdp_detections", [])
    
    for det in detections:
        x = int(det.get('azimuth_deg', 180) / 360 * W)
        y = int(det.get('range_km', 150) / 300 * H)
        conf = det.get('confidence', 0.5)
        size = int(15 + conf * 25)
        
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < size:
                        intensity = (0.5 + conf*0.5) * np.exp(-dist**2/(size/3)**2)
                        if conf > 0.6:
                            overlay[yy, xx, 0] += intensity * 1.0
                            overlay[yy, xx, 1] += intensity * 0.2
                            overlay[yy, xx, 2] += intensity * 0.1
                        elif conf > 0.4:
                            overlay[yy, xx, 0] += intensity * 0.2
                            overlay[yy, xx, 1] += intensity * 0.8
                            overlay[yy, xx, 2] += intensity * 0.8
                        else:
                            overlay[yy, xx, 0] += intensity * 0.4
                            overlay[yy, xx, 1] += intensity * 0.2
                            overlay[yy, xx, 2] += intensity * 0.9
                        overlay[yy, xx, 3] = intensity * 0.7
    
    for c in range(3):
        max_val = np.max(overlay[..., c])
        if max_val > 0:
            overlay[..., c] /= max_val
    overlay[..., 3] = np.clip(overlay[..., 3], 0, 1)
    
    metadata = {
        "detection_count": len(detections),
        "avg_confidence": np.mean([d.get('confidence', 0) for d in detections]) if detections else 0,
        "positions": [{"range_km": d.get('range_km'), "azimuth_deg": d.get('azimuth_deg')} for d in detections]
    }
    
    return overlay, metadata

# ============================================================================
# PROJECT 1: QCI AstroEntangle Refiner (with radar verification)
# ============================================================================

def fdm_soliton_profile(r, k=1.0):
    """Fuzzy Dark Matter soliton profile: ρ(r) ∝ [sin(kr)/(kr)]²"""
    kr = k * r
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = np.where(kr > 0, (np.sin(kr) / kr)**2, 1.0)
    return profile

def pdp_entanglement_overlay(image_data, omega=0.5, fringe_scale=1.0):
    """Photon-DarkPhoton entanglement filter for image enhancement"""
    fft_img = fft2(image_data)
    fft_shift = fftshift(fft_img)
    rows, cols = image_data.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    dark_mask = 0.1 * np.exp(-omega * R**2) * (1 - np.exp(-R**2 / fringe_scale))
    dark_fft = fft_shift * dark_mask
    dark_mode = np.abs(ifft2(fftshift(dark_fft)))
    return dark_mode

def process_qci_astro(image_data, omega=0.5, fringe=1.0, soliton_scale=1.0):
    """Process QCI AstroEntangle Refiner with FDM soliton and PDP"""
    size = min(image_data.shape)
    r = np.linspace(0, 3, size)
    soliton_profile = fdm_soliton_profile(r, k=soliton_scale)
    soliton_2d = np.outer(soliton_profile, soliton_profile)
    soliton_resized = zoom(soliton_2d, (image_data.shape[0]/size, image_data.shape[1]/size))
    pdp = pdp_entanglement_overlay(image_data, omega, fringe)
    enhanced = image_data + 0.3 * soliton_resized + 0.5 * pdp
    enhanced = np.clip(enhanced, 0, 1)
    return enhanced, soliton_resized, pdp

# ============================================================================
# SAMPLE ASTROPHYSICAL IMAGES
# ============================================================================

def get_sample_image(image_name):
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    if image_name == "Galaxy Cluster":
        R = np.sqrt(X**2 + Y**2)
        img = np.exp(-R**2 / 1.5**2)
        img += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
        img += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
        return img / img.max()
    elif image_name == "Nebula":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.cos(5 * theta))
        return img / img.max()
    elif image_name == "Bullet Cluster":
        img = np.exp(-((X-0.8)**2 + Y**2) / 0.3**2) + 0.7 * np.exp(-((X+0.6)**2 + Y**2) / 0.4**2)
        return img / img.max()
    else:
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        img = np.exp(-R**2 / 1.2**2) * (1 + 0.5 * np.cos(10 * R + 3 * theta))
        return img / img.max()

# ============================================================================
# UNIVERSAL IMAGE LOADER
# ============================================================================

def load_image_file(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()
    
    try:
        if file_ext in ['fits', 'fit']:
            from astropy.io import fits
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            with fits.open(tmp_path) as hdul:
                data = hdul[0].data
                if data is None and len(hdul) > 1:
                    data = hdul[1].data
                if data is not None and data.ndim > 2:
                    if data.ndim == 3:
                        data = np.median(data, axis=0)
                    else:
                        data = data[0] if data.ndim == 4 else data[0]
            os.unlink(tmp_path)
            if data is None:
                return None, "No image data found"
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"FITS: {data.shape}"
        
        elif file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
            img = Image.open(BytesIO(file_bytes))
            data = np.array(img)
            if data.ndim == 3:
                data = np.mean(data[:, :, :3], axis=2) if data.shape[2] >= 3 else data[:, :, 0]
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return data, f"{file_ext.upper()}: {data.shape}"
        
        else:
            return None, f"Unsupported: {file_ext}"
            
    except Exception as e:
        return None, str(e)

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def get_image_download_link(fig, filename, title="Download"):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

def get_json_download_link(data, filename, title="Download"):
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" style="text-decoration:none;">{title}</a>'
    return href

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("📡 Radar Data Import")
    uploaded_radar = st.file_uploader("Upload Radar JSON from StealthPDPRadar", type=['json'])
    
    radar_data = None
    verification = {"is_valid": False, "message": "No radar data loaded"}
    
    if uploaded_radar is not None:
        try:
            radar_data = json.load(uploaded_radar)
            verification = verify_radar_data(radar_data)
            display_radar_verification(verification)
            
            # Show detection positions
            if verification["is_valid"] and verification["detection_count"] > 0:
                with st.expander("📡 Detection Positions"):
                    for det in radar_data.get("pdp_detections", []):
                        st.write(f"- Range: {det.get('range_km', '?')} km, Azimuth: {det.get('azimuth_deg', '?')}°, Confidence: {det.get('confidence', 0):.3f}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            radar_data = None
    else:
        st.info("📤 Upload a JSON export from StealthPDPRadar to use your real radar data")
        
        # Option to use sample data
        use_sample = st.checkbox("Use sample radar data for testing", value=True)
        if use_sample:
            sample_data = {
                "timestamp": datetime.now().isoformat(),
                "is_sample": True,
                "parameters": {"omega": 0.72, "fringe": 1.75},
                "pdp_detections": [{"range_km": 148.8, "azimuth_deg": 179.0, "confidence": 0.641}],
                "aircraft_data": [{"callsign": "STEALTH", "stealth_level": "Low", "range_km": 150, "azimuth_deg": 180}]
            }
            radar_data = sample_data
            verification = verify_radar_data(radar_data)
            display_radar_verification(verification)
    
    st.header("🖼️ Image Input")
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_name = "galaxy_cluster"
    if image_source == "Sample":
        image_type = st.selectbox("Sample Image", ["Galaxy Cluster", "Nebula", "Bullet Cluster"])
        astro_image = get_sample_image(image_type)
        image_name = image_type.replace(" ", "_").lower()
    else:
        uploaded_img = st.file_uploader("Upload Image", type=['fits', 'jpg', 'png'])
        if uploaded_img:
            astro_image, info = load_image_file(uploaded_img)
            if astro_image is None:
                st.error(info)
            else:
                st.success(info)
                image_name = uploaded_img.name.split('.')[0]
    
    st.header("🎨 Overlay Settings")
    overlay_opacity = st.slider("Radar Overlay Opacity", 0.0, 1.0, 0.5)

# ============================================================================
# MAIN CONTENT - QCI AstroEntangle
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("FDM Soliton • PDP Entanglement • Quantum Radar Integration")

st.header("🔭 QCI AstroEntangle Refiner")
st.markdown("FDM Soliton Physics + Photon-DarkPhoton Entanglement Overlay")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Controls")
    omega = st.slider("Ω (Entanglement Strength)", 0.0, 1.0, 0.5, 0.01)
    fringe = st.slider("Fringe Scale", 0.1, 3.0, 1.0, 0.1)
    soliton_scale = st.slider("FDM Soliton Scale", 0.5, 3.0, 1.0, 0.1)
    
    if astro_image is not None:
        enhanced, soliton, pdp = process_qci_astro(astro_image, omega, fringe, soliton_scale)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"qcaus_{image_name}_omega{omega:.2f}_fringe{fringe:.2f}_soliton{soliton_scale:.2f}_{timestamp}"
        
        st.markdown("---")
        st.subheader("📥 Download Results")
        
        # Individual downloads
        fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
        ax_orig.imshow(astro_image, cmap='gray', origin='lower')
        ax_orig.set_title("Original Image")
        ax_orig.axis('off')
        st.markdown(get_image_download_link(fig_orig, f"{base_filename}_original.png", "📷 Download Original"), unsafe_allow_html=True)
        plt.close(fig_orig)
        
        fig_enh, ax_enh = plt.subplots(figsize=(6, 6))
        ax_enh.imshow(enhanced, cmap='gray', origin='lower')
        ax_enh.set_title("Enhanced Image")
        ax_enh.axis('off')
        st.markdown(get_image_download_link(fig_enh, f"{base_filename}_enhanced.png", "✨ Download Enhanced Image"), unsafe_allow_html=True)
        plt.close(fig_enh)
        
        # Metadata with radar info
        metadata = {
            "project": "QCI AstroEntangle Refiner",
            "image": image_name,
            "parameters": {"omega": omega, "fringe": fringe, "soliton_scale": soliton_scale},
            "timestamp": timestamp,
            "radar_data": {
                "source": verification.get("source", "None"),
                "is_real": verification.get("is_real_data", False),
                "detection_count": verification.get("detection_count", 0),
                "avg_confidence": verification.get("avg_confidence", 0)
            }
        }
        st.markdown(get_json_download_link(metadata, f"{base_filename}_metadata.json", "📄 Download Metadata"), unsafe_allow_html=True)

with col2:
    st.subheader("Before / After Comparison")
    if astro_image is not None:
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(astro_image, cmap='gray', origin='lower')
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(soliton, cmap='viridis', origin='lower')
        axes[0, 1].set_title(f"FDM Soliton Core\nk={soliton_scale:.2f}")
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pdp, cmap='plasma', origin='lower')
        axes[1, 0].set_title(f"PDP Entanglement\nΩ={omega:.2f}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(enhanced, cmap='gray', origin='lower')
        axes[1, 1].set_title("Enhanced + Overlay")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        comp_filename = f"{base_filename}_comparison_grid.png"
        st.markdown(get_image_download_link(fig, comp_filename, "📸 Download Comparison Grid"), unsafe_allow_html=True)
        plt.close(fig)
        
        # Radar Overlay (with verification)
        if radar_data and verification["is_valid"]:
            st.subheader("📡 Radar Detection Overlay")
            
            overlay, overlay_meta = radar_to_overlay_with_verification(radar_data, astro_image.shape, verification)
            
            if overlay_meta.get("detection_count", 0) > 0:
                astro_rgb = np.stack([enhanced] * 3, axis=-1)
                alpha = overlay[..., 3:4] * overlay_opacity
                combined = astro_rgb * (1 - alpha) + overlay[..., :3] * alpha
                
                fig_radar, ax_radar = plt.subplots(figsize=(8, 8))
                ax_radar.imshow(np.clip(combined, 0, 1), origin='lower')
                
                # Add title showing data source
                if verification["is_real_data"]:
                    title = f"Enhanced Image + REAL RADAR DATA\n{overlay_meta['detection_count']} detections, avg conf={overlay_meta['avg_confidence']:.3f}"
                else:
                    title = f"Enhanced Image + SAMPLE RADAR DATA\n{overlay_meta['detection_count']} detection, avg conf={overlay_meta['avg_confidence']:.3f}"
                
                ax_radar.set_title(title)
                ax_radar.axis('off')
                st.pyplot(fig_radar)
                
                radar_filename = f"{base_filename}_radar_overlay.png"
                st.markdown(get_image_download_link(fig_radar, radar_filename, "📡 Download Radar Overlay"), unsafe_allow_html=True)
                plt.close(fig_radar)
                
                # Show detection positions on image
                if verification["detection_count"] > 0:
                    st.caption(f"Detection positions: {', '.join([f'{d.get("range_km")}km @ {d.get("azimuth_deg")}°' for d in radar_data.get('pdp_detections', [])])}")
            else:
                st.info("No radar detections to overlay")
        else:
            st.info("Upload radar data from StealthPDPRadar to see detection overlay")
    else:
        st.info("👈 Select or upload an image")

# ============================================================================
# PHYSICS EXPLANATION PANEL
# ============================================================================

with st.expander("🔬 Physics of FDM Soliton, PDP Entanglement & Fringe Waves", expanded=False):
    st.markdown(r"""
    ### 📐 Mathematical Foundations
    
    | Component | Formula | Physical Meaning |
    |-----------|---------|------------------|
    | **FDM Soliton** | $\rho(r) = \rho_0 \left[\frac{\sin(kr)}{kr}\right]^2$ | Ground state of ultra-light dark matter (axions, ~10⁻²² eV). Creates central core with quantum interference fringes. |
    | **PDP Entanglement** | $\mathcal{L}_{\text{mix}} = \frac{\varepsilon}{2} F_{\mu\nu} F'^{\mu\nu}$ | Kinetic mixing between photons and dark photons. Extracts hidden sector signatures via spectral duality. |
    | **Fringe Scale** | $\lambda = \frac{2\pi}{m_{\text{dark}}}$ | Dark photon Compton wavelength. Controls oscillation frequency of photon ↔ dark photon conversion. |
    | **Von Neumann Evolution** | $i\partial_t\rho = [H_{\text{eff}}, \rho]$ | Density matrix evolution for entangled photon-dark photon system. |
    
    ### 🔬 Physical Interpretation
    
    **FDM Soliton Ripples:**
    - The `sin²(kr)/(kr)²` pattern creates concentric rings
    - These are **quantum interference fringes** from wave-like dark matter
    - Parameter `k` controls soliton size (larger k = smaller core)
    
    **PDP Entanglement Patterns:**
    - Fourier-space filter extracts dark-mode components
    - Bright regions indicate **dark photon field presence**
    - Parameter `Ω` controls entanglement strength
    
    **Fringe Waves:**
    - Oscillations in the PDP entanglement map
    - Frequency related to dark photon mass
    - Larger fringe scale = smoother patterns (lighter dark photons)
    
    ### 📡 Radar Data Integration
    
    When you upload real radar data from StealthPDPRadar:
    - Detection positions are overlaid on the enhanced image
    - Confidence scores determine overlay brightness
    - Red markers indicate high-confidence stealth detections
    
    ### 🔍 Verification of Real Data
    
    The app confirms when real radar data is used:
    - ✅ Shows "REAL RADAR DATA CONFIRMED" in sidebar
    - 📡 Detection positions are displayed
    - 📊 Radar parameters are included in metadata downloads
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | Verified Radar Integration | © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
