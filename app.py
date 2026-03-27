"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Universal Image Processor - Supports FITS, CSV, JPEG, PNG, BMP, DICOM, TIFF, NPY, NPZ
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.ndimage import gaussian_filter, zoom
from io import BytesIO
from PIL import Image
import tempfile
import os

# Page config must be first Streamlit command
st.set_page_config(
    page_title="QCAUS - Quantum Cosmology Suite",
    page_icon="🌌",
    layout="wide"
)

# ============================================================================
# UNIVERSAL IMAGE LOADER
# ============================================================================

def load_image_file(uploaded_file):
    """Universal image loader - handles FITS, CSV, JPEG, PNG, BMP, DICOM, TIFF, NPY, NPZ"""
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()
    
    try:
        # ========== FITS Files ==========
        if file_ext in ['fits', 'fit']:
            try:
                from astropy.io import fits
                with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                with fits.open(tmp_path) as hdul:
                    data = hdul[0].data
                    if data is None and len(hdul) > 1:
                        data = hdul[1].data
                    
                    # Handle multi-dimensional FITS
                    if data is not None:
                        if data.ndim == 3:
                            data = np.median(data, axis=0)
                        elif data.ndim > 3:
                            data = data[0, 0] if data.ndim == 4 else data[0]
                
                os.unlink(tmp_path)
                
                if data is None:
                    return None, "FITS: No image data found"
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"FITS: {data.shape}"
                
            except ImportError:
                return None, "astropy not installed. Install with: pip install astropy"
            except Exception as e:
                return None, f"FITS error: {str(e)}"
        
        # ========== DICOM Files ==========
        elif file_ext in ['dcm', 'dicom']:
            try:
                import pydicom
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                ds = pydicom.dcmread(tmp_path)
                data = ds.pixel_array
                
                os.unlink(tmp_path)
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"DICOM: {data.shape}"
                
            except ImportError:
                return None, "pydicom not installed. Install with: pip install pydicom"
            except Exception as e:
                return None, f"DICOM error: {str(e)}"
        
        # ========== CSV Files ==========
        elif file_ext == 'csv':
            try:
                df = pd.read_csv(BytesIO(file_bytes), header=None)
                data = df.values.astype(float)
                data = np.nan_to_num(data)
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"CSV: {data.shape}"
                
            except Exception as e:
                return None, f"CSV error: {str(e)}"
        
        # ========== NumPy Files ==========
        elif file_ext == 'npy':
            try:
                data = np.load(BytesIO(file_bytes))
                if data.ndim > 2:
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)
                    else:
                        data = data[0] if data.shape[0] > 0 else data
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"NPY: {data.shape}"
                
            except Exception as e:
                return None, f"NPY error: {str(e)}"
        
        # ========== NPZ Archives ==========
        elif file_ext == 'npz':
            try:
                npz = np.load(BytesIO(file_bytes))
                first_key = list(npz.keys())[0]
                data = npz[first_key]
                if data.ndim > 2:
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)
                    else:
                        data = data[0] if data.shape[0] > 0 else data
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"NPZ: {data.shape} (key: {first_key})"
                
            except Exception as e:
                return None, f"NPZ error: {str(e)}"
        
        # ========== TIFF/TIF Files ==========
        elif file_ext in ['tiff', 'tif']:
            try:
                img = Image.open(BytesIO(file_bytes))
                data = np.array(img)
                if data.ndim == 3:
                    data = np.mean(data, axis=2)
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"TIFF: {data.shape}"
                
            except Exception as e:
                return None, f"TIFF error: {str(e)}"
        
        # ========== Standard Images ==========
        elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            try:
                img = Image.open(BytesIO(file_bytes))
                data = np.array(img)
                if data.ndim == 3:
                    if data.shape[2] >= 3:
                        data = np.mean(data[:, :, :3], axis=2)
                    else:
                        data = data[:, :, 0]
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"{file_ext.upper()}: {data.shape}"
                
            except Exception as e:
                return None, f"Image error: {str(e)}"
        
        else:
            return None, f"Unsupported format: {file_ext}"
            
    except Exception as e:
        return None, f"General error: {str(e)}"

# ============================================================================
# SAMPLE ASTROPHYSICAL IMAGES
# ============================================================================

def get_sample_astrophysical_image(image_name):
    """Generate sample astrophysical images for demonstration"""
    
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    if image_name == "Galaxy Cluster":
        R = np.sqrt(X**2 + Y**2)
        galaxy = np.exp(-R**2 / 1.5**2)
        galaxy += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
        galaxy += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
        galaxy += 0.3 * np.exp(-((X+0.2)**2 + (Y-0.7)**2) / 0.35**2)
        galaxy += np.random.randn(size, size) * 0.05
        return galaxy / galaxy.max()
    
    elif image_name == "Nebula":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        nebula = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.cos(5 * theta))
        nebula += 0.2 * np.exp(-((X-0.8)**2 + Y**2) / 0.2**2)
        nebula += 0.2 * np.exp(-((X+0.8)**2 + Y**2) / 0.2**2)
        nebula += np.random.randn(size, size) * 0.05
        return nebula / nebula.max()
    
    elif image_name == "CMB":
        cmb = np.random.randn(size, size)
        cmb = gaussian_filter(cmb, sigma=5)
        return (cmb - cmb.min()) / (cmb.max() - cmb.min())
    
    elif image_name == "Spiral Galaxy":
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        spiral = np.exp(-R**2 / 1.2**2) * (1 + 0.5 * np.cos(10 * R + 3 * theta))
        spiral += 0.3 * np.exp(-((X-0.3)**2 + (Y-0.2)**2) / 0.2**2)
        return spiral / spiral.max()
    
    else:  # Supernova Remnant
        R = np.sqrt(X**2 + Y**2)
        snr = np.exp(-((R-0.8)**2) / 0.2**2) * (1 + 0.3 * np.cos(8 * np.arctan2(Y, X)))
        snr += 0.2 * np.random.randn(size, size)
        return snr / snr.max()

# ============================================================================
# RADAR OVERLAY FUNCTIONS
# ============================================================================

def radar_to_optical_overlay_enhanced(radar_data, image_shape, radar_range_km=300):
    """Convert radar detection signatures to optical overlay"""
    
    pdp_detections = radar_data.get('pdp_detections', [])
    aircraft_data = radar_data.get('aircraft_data', [])
    
    H, W = image_shape
    optical_overlay = np.zeros((H, W, 4))
    
    for det in pdp_detections:
        range_km = det.get('range_km', 150)
        azimuth_deg = det.get('azimuth_deg', 180)
        confidence = det.get('confidence', 0.5)
        
        x = int(azimuth_deg / 360 * W)
        y = int(range_km / radar_range_km * H)
        size = int(max(10, min(50, confidence * H / 10)))
        brightness = 0.5 + confidence * 0.5
        
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < size:
                        intensity = brightness * np.exp(-dist**2 / (size/3)**2)
                        
                        if confidence > 0.6:
                            optical_overlay[yy, xx, 0] += intensity * 1.0
                            optical_overlay[yy, xx, 1] += intensity * 0.2
                            optical_overlay[yy, xx, 2] += intensity * 0.1
                        elif confidence > 0.4:
                            optical_overlay[yy, xx, 0] += intensity * 0.2
                            optical_overlay[yy, xx, 1] += intensity * 0.8
                            optical_overlay[yy, xx, 2] += intensity * 0.8
                        else:
                            optical_overlay[yy, xx, 0] += intensity * 0.4
                            optical_overlay[yy, xx, 1] += intensity * 0.2
                            optical_overlay[yy, xx, 2] += intensity * 0.9
                        
                        optical_overlay[yy, xx, 3] += intensity * 0.7
    
    for aircraft in aircraft_data:
        stealth_level = aircraft.get('stealth_level', 'None')
        if stealth_level in ['Very High', 'High']:
            range_km = aircraft.get('range_km', 150)
            azimuth_deg = aircraft.get('azimuth_deg', 180)
            
            x = int(azimuth_deg / 360 * W)
            y = int(range_km / radar_range_km * H)
            
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < W and 0 <= yy < H:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= 6:
                            optical_overlay[yy, xx, 0] = 1.0
                            optical_overlay[yy, xx, 1] = 0.2
                            optical_overlay[yy, xx, 2] = 0.2
                            optical_overlay[yy, xx, 3] = 0.9
    
    for c in range(3):
        max_val = np.max(optical_overlay[..., c])
        if max_val > 0:
            optical_overlay[..., c] = optical_overlay[..., c] / max_val
    
    optical_overlay[..., 3] = np.clip(optical_overlay[..., 3], 0, 1)
    
    return optical_overlay

def create_sample_radar_data():
    """Create sample radar detection data"""
    return {
        "timestamp": "2026-03-27T12:00:00",
        "parameters": {"omega": 0.72, "fringe": 1.75, "entanglement": 0.44, "mixing": 0.17, "threshold": 0.30},
        "detection_summary": {"max_probability": 1.0, "mean_probability": 0.65, "total_detections": 1},
        "pdp_detections": [{"id": 1, "range_km": 148.8, "azimuth_deg": 179.0, "confidence": 0.641, "size_pixels": 37}],
        "aircraft_data": [{"callsign": "STEALTH", "aircraft_name": "Stealth Target", "stealth_level": "Low", "range_km": 150, "azimuth_deg": 180, "rcs_m2": 8.5}]
    }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("📡 Radar Data Import")
    use_sample = st.checkbox("Use Sample Radar Data", value=False)
    
    radar_data = None
    if use_sample:
        radar_data = create_sample_radar_data()
        st.success("✅ Using sample radar data")
    else:
        uploaded_radar = st.file_uploader("Upload Radar JSON Export", type=['json'])
        if uploaded_radar:
            try:
                radar_data = json.load(uploaded_radar)
                st.success(f"✅ Loaded {len(radar_data.get('pdp_detections', []))} detections")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if radar_data:
        params = radar_data.get('parameters', {})
        st.metric("Detections", radar_data.get('detection_summary', {}).get('total_detections', 0))
        st.caption(f"Ω={params.get('omega', 0.72)} | ε={params.get('mixing', 0.17)}")
    
    st.header("🖼️ Astrophysical Image")
    
    image_source = st.radio("Image Source", ["Sample", "Upload File"])
    
    astro_image = None
    image_info = None
    
    if image_source == "Sample":
        image_type = st.selectbox(
            "Select Sample Image",
            ["Galaxy Cluster", "Nebula", "CMB", "Spiral Galaxy", "Supernova Remnant"]
        )
        astro_image = get_sample_astrophysical_image(image_type)
        image_info = f"Sample: {image_type}"
        
    else:
        st.caption("Supported: FITS, CSV, JPEG, PNG, BMP, DICOM, TIFF, NPY, NPZ")
        uploaded_image = st.file_uploader(
            "Upload Image File",
            type=['fits', 'fit', 'csv', 'jpg', 'jpeg', 'png', 'bmp', 'dcm', 'tiff', 'tif', 'npy', 'npz']
        )
        if uploaded_image:
            with st.spinner("Loading image..."):
                astro_image, image_info = load_image_file(uploaded_image)
                if astro_image is None:
                    st.error(image_info)
                else:
                    st.success(image_info)
    
    st.header("🎨 Overlay Settings")
    overlay_opacity = st.slider("Opacity", 0.0, 1.0, 0.6, 0.05)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("Integrating quantum radar detection with astrophysical observations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Astrophysical Image")
    
    if astro_image is not None:
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(astro_image, cmap='gray', origin='lower')
        ax1.set_title(image_info)
        ax1.axis('off')
        st.pyplot(fig1)
    else:
        st.info("👈 Select or upload an image")

with col2:
    st.subheader("Radar Detection Overlay")
    
    if astro_image is not None and radar_data is not None:
        # Create overlay
        overlay = radar_to_optical_overlay_enhanced(radar_data, astro_image.shape[:2])
        
        # Convert to 3-channel for display
        if len(astro_image.shape) == 2:
            astro_display = np.stack([astro_image] * 3, axis=-1)
        else:
            astro_display = astro_image.copy()
        
        # Apply alpha blending
        alpha = overlay[..., 3:4] * overlay_opacity
        combined = astro_display * (1 - alpha) + overlay[..., :3] * alpha
        combined = np.clip(combined, 0, 1)
        
        # Display
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(combined, origin='lower')
        ax2.set_title("Astrophysical Image + Radar Overlay")
        ax2.axis('off')
        st.pyplot(fig2)
        
        # Legend
        st.caption("🔴 Red: Stealth targets | 🟢 Cyan: Entanglement | 🔵 Violet: Dark mode")
        
        # Detection details
        if radar_data.get('pdp_detections'):
            with st.expander("📋 Detection Details"):
                for det in radar_data.get('pdp_detections', []):
                    st.write(f"- Range: {det.get('range_km', '?')} km")
                    st.write(f"- Azimuth: {det.get('azimuth_deg', '?')}°")
                    st.write(f"- Confidence: {det.get('confidence', 0):.3f}")
        
    elif astro_image is None:
        st.info("👈 Select or upload an astrophysical image")
    elif radar_data is None:
        st.info("👈 Upload radar data from the sidebar")

# ============================================================================
# DETECTION ANALYSIS TAB
# ============================================================================

st.subheader("📊 Detection Analysis")

if radar_data:
    detections = radar_data.get('pdp_detections', [])
    aircraft = radar_data.get('aircraft_data', [])
    params = radar_data.get('parameters', {})
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Detections", len(detections))
    col_b.metric("Aircraft", len(aircraft))
    col_c.metric("Max Probability", f"{radar_data.get('detection_summary', {}).get('max_probability', 0):.3f}")
    col_d.metric("Ω", f"{params.get('omega', 0.72):.2f}")
    
    if detections:
        with st.expander("📋 PDP Detections"):
            st.dataframe(pd.DataFrame(detections))
    
    if aircraft:
        with st.expander("✈️ Aircraft Data"):
            st.dataframe(pd.DataFrame(aircraft))
else:
    st.info("Upload radar data to see detection analysis")

# ============================================================================
# ABOUT SECTION
# ============================================================================

with st.expander("📖 About QCAUS", expanded=False):
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite
    
    **Supported Image Formats:**
    | Format | Extension | Use Case |
    |--------|-----------|----------|
    | FITS | .fits, .fit | Astronomical images, spectroscopy |
    | CSV | .csv | Data grids, simulation outputs |
    | JPEG/PNG/BMP | .jpg, .png, .bmp | Standard images |
    | TIFF | .tiff, .tif | GeoTIFF, microscopy |
    | DICOM | .dcm | Medical imaging (CT, MRI) |
    | NumPy | .npy, .npz | Scientific arrays |
    
    **Color Mapping:**
    - 🔴 Red: High-confidence stealth targets
    - 🟢 Cyan: Entanglement residuals (quantum interference)
    - 🔵 Violet: Dark-mode leakage (dark photon signatures)
    
    **Integration with StealthPDPRadar:**
    1. Export detection data as Complete JSON
    2. Upload to this app
    3. Overlay on any astrophysical image
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | Universal Image Processor | FITS • CSV • DICOM • TIFF • NPY<br>
    Integrated with StealthPDPRadar | © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
