"""
Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
Enhanced with Radar Data Import from StealthPDPRadar
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.ndimage import gaussian_filter
from io import BytesIO
from PIL import Image
import requests

# Page config must be first Streamlit command
st.set_page_config(
    page_title="QCAUS - Quantum Cosmology Suite",
    page_icon="🌌",
    layout="wide"
)

# ============================================================================
# RADAR TO OPTICAL COLOR CONVERSION FUNCTIONS
# ============================================================================

def radar_to_optical_overlay(radar_data):
    """
    Convert radar detection signatures to optical color representation
    
    Parameters:
    -----------
    radar_data : dict
        Loaded JSON data from StealthPDPRadar export
    
    Returns:
    --------
    optical_overlay : numpy.ndarray
        RGBA overlay for astrophysical images (H, W, 4)
    """
    
    # Extract components from radar data
    pdp_detections = radar_data.get('pdp_detections', [])
    aircraft_data = radar_data.get('aircraft_data', [])
    detection_summary = radar_data.get('detection_summary', {})
    parameters = radar_data.get('parameters', {})
    
    # Create overlay dimensions (default 512x512 for astrophysical images)
    H, W = 512, 512
    optical_overlay = np.zeros((H, W, 4))
    
    # 1. Create Gaussian blobs for each detection
    for det in pdp_detections:
        range_km = det.get('range_km', 150)
        azimuth_deg = det.get('azimuth_deg', 180)
        confidence = det.get('confidence', 0.5)
        
        # Convert radar coordinates to image coordinates
        # Range: 0-300km -> 0-512 pixels
        # Azimuth: 0-360° -> 0-512 pixels
        x = int(azimuth_deg / 360 * W)
        y = int(range_km / 300 * H)
        
        # Confidence determines brightness and size
        size = int(20 + confidence * 30)
        brightness = 0.5 + confidence * 0.5
        
        # Create Gaussian blob for detection
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < size:
                        # Gaussian falloff
                        intensity = brightness * np.exp(-dist**2 / (size/3)**2)
                        
                        # Color based on confidence and stealth level
                        if confidence > 0.6:
                            # High confidence - Red/IR
                            optical_overlay[yy, xx, 0] += intensity * 1.0  # Red
                            optical_overlay[yy, xx, 1] += intensity * 0.2  # Green
                            optical_overlay[yy, xx, 2] += intensity * 0.1  # Blue
                        elif confidence > 0.4:
                            # Medium confidence - Cyan/Teal (entanglement)
                            optical_overlay[yy, xx, 0] += intensity * 0.2  # Red
                            optical_overlay[yy, xx, 1] += intensity * 0.8  # Green
                            optical_overlay[yy, xx, 2] += intensity * 0.8  # Blue
                        else:
                            # Low confidence - Violet/Blue (dark mode)
                            optical_overlay[yy, xx, 0] += intensity * 0.4  # Red
                            optical_overlay[yy, xx, 1] += intensity * 0.2  # Green
                            optical_overlay[yy, xx, 2] += intensity * 0.9  # Blue
                        
                        # Set alpha (transparency)
                        optical_overlay[yy, xx, 3] += intensity * 0.7
    
    # 2. Add detection markers for stealth aircraft
    for aircraft in aircraft_data:
        stealth_level = aircraft.get('stealth_level', 'None')
        if stealth_level in ['Very High', 'High']:
            range_km = aircraft.get('range_km', 150)
            azimuth_deg = aircraft.get('azimuth_deg', 180)
            
            x = int(azimuth_deg / 360 * W)
            y = int(range_km / 300 * H)
            
            # Add red circle marker for stealth
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < W and 0 <= yy < H:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= 4:
                            optical_overlay[yy, xx, 0] = 1.0  # Full red
                            optical_overlay[yy, xx, 1] = 0.2
                            optical_overlay[yy, xx, 2] = 0.2
                            optical_overlay[yy, xx, 3] = 0.9
    
    # Normalize RGB channels
    for c in range(3):
        max_val = np.max(optical_overlay[..., c])
        if max_val > 0:
            optical_overlay[..., c] = optical_overlay[..., c] / max_val
    
    # Clip alpha
    optical_overlay[..., 3] = np.clip(optical_overlay[..., 3], 0, 1)
    
    return optical_overlay

def overlay_on_astrophysical_image(astrophysical_image, optical_overlay):
    """
    Overlay radar signatures on astrophysical images
    
    Parameters:
    -----------
    astrophysical_image : numpy.ndarray
        RGB astrophysical image (H, W, 3) or grayscale (H, W)
    optical_overlay : numpy.ndarray
        RGBA overlay from radar data (H, W, 4)
    
    Returns:
    --------
    combined : numpy.ndarray
        Combined RGB image with overlay
    """
    # Convert grayscale to RGB if needed
    if len(astrophysical_image.shape) == 2:
        astrophysical_image = np.stack([astrophysical_image] * 3, axis=-1)
    
    # Normalize
    if astrophysical_image.max() > 1:
        astrophysical_image = astrophysical_image / astrophysical_image.max()
    
    # Resize overlay to match astrophysical image if needed
    if astrophysical_image.shape[:2] != optical_overlay.shape[:2]:
        from skimage.transform import resize
        optical_overlay = resize(optical_overlay, astrophysical_image.shape[:2], preserve_range=True)
    
    # Apply alpha blending
    alpha = optical_overlay[..., 3:4]
    combined = astrophysical_image * (1 - alpha) + optical_overlay[..., :3] * alpha
    
    return np.clip(combined, 0, 1)

# ============================================================================
# EXAMPLE ASTROPHYSICAL IMAGES
# ============================================================================

def get_sample_astrophysical_image(image_name):
    """Get sample astrophysical images for demonstration"""
    
    # Create a synthetic galaxy-like image for demonstration
    size = 512
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    if image_name == "Galaxy Cluster":
        # Simulate a galaxy cluster with multiple cores
        R = np.sqrt(X**2 + Y**2)
        galaxy = np.exp(-R**2 / 1.5**2)
        
        # Add secondary peaks
        galaxy += 0.5 * np.exp(-((X-0.5)**2 + (Y-0.3)**2) / 0.3**2)
        galaxy += 0.4 * np.exp(-((X+0.4)**2 + (Y+0.6)**2) / 0.4**2)
        galaxy += 0.3 * np.exp(-((X+0.2)**2 + (Y-0.7)**2) / 0.35**2)
        
        # Add noise
        galaxy += np.random.randn(size, size) * 0.05
        
        return galaxy / galaxy.max()
    
    elif image_name == "Nebula":
        # Simulate a nebula with filamentary structure
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        nebula = np.exp(-R**2 / 1.2**2) * (1 + 0.3 * np.cos(5 * theta))
        nebula += 0.2 * np.exp(-((X-0.8)**2 + Y**2) / 0.2**2)
        nebula += 0.2 * np.exp(-((X+0.8)**2 + Y**2) / 0.2**2)
        
        # Add noise
        nebula += np.random.randn(size, size) * 0.05
        
        return nebula / nebula.max()
    
    else:  # Cosmic Microwave Background
        # Simulate CMB-like fluctuations
        cmb = np.random.randn(size, size)
        cmb = gaussian_filter(cmb, sigma=5)
        return (cmb - cmb.min()) / (cmb.max() - cmb.min())

# ============================================================================
# SIDEBAR - RADAR DATA IMPORT
# ============================================================================

with st.sidebar:
    st.title("🌌 QCAUS")
    st.markdown("Quantum Cosmology & Astrophysics Unified Suite")
    
    st.header("📡 Radar Data Import")
    st.caption("Import detection data from StealthPDPRadar")
    
    # File uploader for radar data
    uploaded_radar = st.file_uploader(
        "Upload Radar JSON Export",
        type=['json'],
        help="Export from StealthPDPRadar as Complete JSON"
    )
    
    if uploaded_radar is not None:
        try:
            radar_data = json.load(uploaded_radar)
            st.success(f"✅ Loaded {len(radar_data.get('pdp_detections', []))} detections")
            
            # Display detection summary
            summary = radar_data.get('detection_summary', {})
            st.metric("Total Detections", summary.get('total_detections', 0))
            st.metric("Max Probability", f"{summary.get('max_probability', 0):.3f}")
            
            # Parameter display
            params = radar_data.get('parameters', {})
            st.caption(f"Ω={params.get('omega', 0.72)} | ε={params.get('mixing', 0.17)}")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            radar_data = None
    else:
        radar_data = None
        st.info("📤 Export data from StealthPDPRadar as Complete JSON")
    
    st.header("🖼️ Astrophysical Image")
    image_type = st.selectbox(
        "Select Image",
        ["Galaxy Cluster", "Nebula", "CMB"]
    )
    
    st.header("🎨 Overlay Settings")
    overlay_opacity = st.slider("Opacity", 0.0, 1.0, 0.6, 0.05)
    show_entanglement = st.checkbox("Show Entanglement (Cyan)", True)
    show_darkmode = st.checkbox("Show Dark Mode (Violet)", True)
    show_stealth = st.checkbox("Show Stealth (Red)", True)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🌌 Quantum Cosmology & Astrophysics Unified Suite")
st.markdown("Integrating quantum radar detection with astrophysical observations")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔭 Image Overlay", "📊 Detection Analysis", "📖 About"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Astrophysical Image")
        
        # Get sample astrophysical image
        astro_image = get_sample_astrophysical_image(image_type)
        
        # Display as grayscale
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(astro_image, cmap='gray', origin='lower')
        ax1.set_title(f"{image_type}")
        ax1.axis('off')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Radar Detection Overlay")
        
        if radar_data is not None:
            # Convert radar data to optical overlay
            optical_overlay = radar_to_optical_overlay(radar_data)
            
            # Combine with astrophysical image
            combined = overlay_on_astrophysical_image(astro_image, optical_overlay)
            
            # Apply opacity
            alpha = overlay_opacity
            combined = astro_image[:, :, np.newaxis] * (1 - alpha) + combined * alpha
            combined = np.clip(combined, 0, 1)
            
            # Display combined image
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(combined, origin='lower')
            ax2.set_title(f"{image_type} + Radar Overlay")
            ax2.axis('off')
            st.pyplot(fig2)
            
            # Show detection info
            st.caption(f"🔴 Red markers: Stealth targets | 🟢 Cyan: Entanglement | 🔵 Violet: Dark mode")
            
        else:
            st.info("👈 Upload radar data from the sidebar to see overlay")
            
            # Show just the astrophysical image
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(astro_image, cmap='gray', origin='lower')
            ax2.set_title(f"{image_type} (No overlay)")
            ax2.axis('off')
            st.pyplot(fig2)

with tab2:
    st.subheader("📊 Detection Analysis")
    
    if radar_data is not None:
        # Display detection data
        detections = radar_data.get('pdp_detections', [])
        aircraft = radar_data.get('aircraft_data', [])
        params = radar_data.get('parameters', {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Detections", len(detections))
        col2.metric("Aircraft", len(aircraft))
        col3.metric("PDP Parameters", f"Ω={params.get('omega', 0.72):.2f}")
        
        # Detections table
        if detections:
            st.subheader("PDP Detections")
            det_df = pd.DataFrame(detections)
            st.dataframe(det_df, use_container_width=True)
        
        # Aircraft table
        if aircraft:
            st.subheader("Aircraft Data")
            ac_df = pd.DataFrame(aircraft)
            st.dataframe(ac_df, use_container_width=True)
        
        # Parameter summary
        with st.expander("PDP Filter Parameters"):
            st.json(params)
        
    else:
        st.info("Upload radar data to see detection analysis")

with tab3:
    st.subheader("📖 About")
    st.markdown("""
    ### Quantum Cosmology & Astrophysics Unified Suite (QCAUS)
    
    This suite integrates quantum radar detection with astrophysical observations:
    
    #### Features
    - **Radar Data Import**: Load detection data from StealthPDPRadar
    - **Optical Color Conversion**: Convert radar signatures to visible spectrum
    - **Astrophysical Overlay**: Combine with galaxy clusters, nebulae, and CMB
    - **Quantum Signatures**: Visualize entanglement residuals and dark-mode leakage
    
    #### Color Mapping
    | Signature | Radar Color | Optical Color | Meaning |
    |-----------|-------------|---------------|---------|
    | Entanglement | Green | Cyan/Teal | Quantum interference patterns |
    | Dark Mode | Blue | Violet | Dark photon signatures |
    | Stealth | Red | Infrared/Red | High-confidence detections |
    
    #### Integration with StealthPDPRadar
    1. Run detection in StealthPDPRadar
    2. Export as **Complete JSON**
    3. Upload to this app
    4. View overlays on astrophysical images
    
    #### References
    - Photon-DarkPhoton Entanglement
    - Fuzzy Dark Matter Soliton Physics
    - Quantum Cosmology Integration Suite
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <b>QCAUS</b> | Quantum Cosmology & Astrophysics Unified Suite<br>
    Integrated with StealthPDPRadar | © 2026 Tony E. Ford
</div>
""", unsafe_allow_html=True)
