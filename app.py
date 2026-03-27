"""
Universal Image Processor for QCAUS
Handles: FITS, CSV, JPEG, PNG, BMP, DICOM, TIFF, NPY, NPZ
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from scipy.ndimage import gaussian_filter, zoom
from io import BytesIO
from PIL import Image
import requests
import os
import tempfile

# ============================================================================
# UNIVERSAL IMAGE LOADER
# ============================================================================

def load_image_file(uploaded_file):
    """
    Universal image loader - handles multiple formats
    
    Supported formats:
    - FITS: Astronomical FITS files
    - CSV: CSV data grids
    - JPEG/PNG/BMP: Standard images
    - DICOM: Medical imaging
    - TIFF: GeoTIFF, microscopy
    - NPY/NPZ: NumPy arrays
    """
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.read()
    
    # Reset file pointer for potential re-reading
    uploaded_file.seek(0)
    
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
                    if data.ndim == 3:
                        # Take first slice or median
                        data = np.median(data, axis=0)
                    elif data.ndim > 3:
                        data = data[0, 0] if data.ndim == 4 else data[0]
                
                os.unlink(tmp_path)
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"FITS: {data.shape}"
                
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
                # Try to load as numeric grid
                df = pd.read_csv(BytesIO(file_bytes), header=None)
                data = df.values.astype(float)
                
                # Handle missing values
                data = np.nan_to_num(data)
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"CSV: {data.shape}"
                
            except Exception as e:
                return None, f"CSV error: {str(e)}"
        
        # ========== NumPy Files ==========
        elif file_ext == 'npy':
            try:
                data = np.load(BytesIO(file_bytes))
                
                # Handle multi-dimensional
                if data.ndim > 2:
                    # Take first slice or average
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)
                    else:
                        data = data[0] if data.shape[0] > 0 else data
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"NPY: {data.shape}"
                
            except Exception as e:
                return None, f"NPY error: {str(e)}"
        
        # ========== NPZ Archives ==========
        elif file_ext == 'npz':
            try:
                npz = np.load(BytesIO(file_bytes))
                # Get first array
                first_key = list(npz.keys())[0]
                data = npz[first_key]
                
                # Handle multi-dimensional
                if data.ndim > 2:
                    if data.ndim == 3:
                        data = np.mean(data, axis=0)
                    else:
                        data = data[0] if data.shape[0] > 0 else data
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"NPZ: {data.shape} (key: {first_key})"
                
            except Exception as e:
                return None, f"NPZ error: {str(e)}"
        
        # ========== TIFF/TIF Files ==========
        elif file_ext in ['tiff', 'tif']:
            try:
                from PIL import Image
                img = Image.open(BytesIO(file_bytes))
                data = np.array(img)
                
                # Convert to grayscale if RGB
                if data.ndim == 3:
                    data = np.mean(data, axis=2)
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"TIFF: {data.shape}"
                
            except Exception as e:
                return None, f"TIFF error: {str(e)}"
        
        # ========== Standard Images (JPEG, PNG, BMP, GIF) ==========
        elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            try:
                img = Image.open(BytesIO(file_bytes))
                data = np.array(img)
                
                # Convert to grayscale if RGB/RGBA
                if data.ndim == 3:
                    if data.shape[2] >= 3:
                        data = np.mean(data[:, :, :3], axis=2)
                    else:
                        data = data[:, :, 0]
                
                # Normalize
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
                return data, f"{file_ext.upper()}: {data.shape}"
                
            except Exception as e:
                return None, f"Image error: {str(e)}"
        
        else:
            return None, f"Unsupported format: {file_ext}"
            
    except Exception as e:
        return None, f"General error: {str(e)}"

# ============================================================================
# RESIZE FUNCTION FOR OVERLAY
# ============================================================================

def resize_to_match(image, target_shape):
    """Resize image to match target shape for overlay"""
    if image.shape == target_shape:
        return image
    
    from scipy.ndimage import zoom
    factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
    resized = zoom(image, factors, order=1)
    return resized

# ============================================================================
# ENHANCED RADAR OVERLAY WITH BETTER COORDINATE MAPPING
# ============================================================================

def radar_to_optical_overlay_enhanced(radar_data, image_shape, radar_range_km=300):
    """
    Enhanced overlay with better coordinate mapping for any image size
    """
    
    pdp_detections = radar_data.get('pdp_detections', [])
    aircraft_data = radar_data.get('aircraft_data', [])
    
    H, W = image_shape
    optical_overlay = np.zeros((H, W, 4))
    
    # Radar coordinates: range 0-300km, azimuth 0-360°
    # Map to image coordinates: y (range) 0-H, x (azimuth) 0-W
    
    for det in pdp_detections:
        range_km = det.get('range_km', 150)
        azimuth_deg = det.get('azimuth_deg', 180)
        confidence = det.get('confidence', 0.5)
        
        # Convert to pixel coordinates
        x = int(azimuth_deg / 360 * W)
        y = int(range_km / radar_range_km * H)
        
        # Size based on confidence and image dimensions
        size = int(max(10, min(50, confidence * H / 10)))
        brightness = 0.5 + confidence * 0.5
        
        # Create Gaussian blob
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < size:
                        intensity = brightness * np.exp(-dist**2 / (size/3)**2)
                        
                        # Color based on confidence
                        if confidence > 0.6:
                            optical_overlay[yy, xx, 0] += intensity * 1.0  # Red
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
    
    # Add stealth aircraft markers
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
    
    # Normalize
    for c in range(3):
        max_val = np.max(optical_overlay[..., c])
        if max_val > 0:
            optical_overlay[..., c] = optical_overlay[..., c] / max_val
    
    optical_overlay[..., 3] = np.clip(optical_overlay[..., 3], 0, 1)
    
    return optical_overlay

# ============================================================================
# SAMPLE DATA CREATION
# ============================================================================

def create_sample_radar_data():
    """Create sample radar detection data"""
    return {
        "timestamp": "2026-03-27T12:00:00",
        "parameters": {
            "omega": 0.72,
            "fringe": 1.75,
            "entanglement": 0.44,
            "mixing": 0.17,
            "threshold": 0.30
        },
        "detection_summary": {
            "max_probability": 1.0,
            "mean_probability": 0.65,
            "total_detections": 1
        },
        "pdp_detections": [
            {"id": 1, "range_km": 148.8, "azimuth_deg": 179.0, "confidence": 0.641, "size_pixels": 37}
        ],
        "aircraft_data": [
            {
                "callsign": "STEALTH",
                "aircraft_name": "Stealth Target",
                "aircraft_type": "Stealth Demonstrator",
                "country": "Test",
                "stealth_level": "Low",
                "range_km": 150,
                "azimuth_deg": 180,
                "rcs_m2": 8.5
            }
        ]
    }

# ============================================================================
# SIDEBAR - UPDATED WITH ALL FORMATS
# ============================================================================

# Add this to your sidebar in the main app
# (Replace the existing image source section)

"""
st.header("🖼️ Astrophysical Image")

# Image source selection
image_source = st.radio("Image Source", ["Sample", "Upload File"])

if image_source == "Sample":
    image_type = st.selectbox(
        "Select Sample Image",
        ["Galaxy Cluster", "Nebula", "CMB"]
    )
    uploaded_image = None
    file_format = None
else:
    st.caption("Supported formats: FITS, CSV, JPEG, PNG, BMP, DICOM, TIFF, NPY, NPZ")
    uploaded_image = st.file_uploader(
        "Upload Image File",
        type=['fits', 'fit', 'csv', 'jpg', 'jpeg', 'png', 'bmp', 'dcm', 'tiff', 'tif', 'npy', 'npz'],
        help="Any astrophysical or scientific image format"
    )
    if uploaded_image:
        file_format = uploaded_image.name.split('.')[-1].upper()
        st.info(f"Format: {file_format}")

# Load sample images function
def get_sample_astrophysical_image(image_name):
    # ... existing sample image code ...
    pass
"""

# ============================================================================
# MAIN DISPLAY FUNCTION
# ============================================================================

def display_image_with_overlay(astro_image, radar_data, opacity=0.6, title="Image with Radar Overlay"):
    """Display image with radar overlay"""
    
    if astro_image is None or radar_data is None:
        return None
    
    # Convert to 3-channel for display
    if len(astro_image.shape) == 2:
        astro_display = np.stack([astro_image] * 3, axis=-1)
    else:
        astro_display = astro_image.copy()
    
    # Create overlay
    overlay = radar_to_optical_overlay_enhanced(radar_data, astro_image.shape[:2])
    
    # Apply alpha blending
    alpha = overlay[..., 3:4]
    combined = astro_display * (1 - alpha * opacity) + overlay[..., :3] * alpha * opacity
    
    # Display
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.clip(combined, 0, 1), origin='lower')
    ax.set_title(title)
    ax.axis('off')
    
    return fig
