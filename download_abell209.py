# download_abell209.py
"""
Download Abell 209 HST data from MAST for processing with QCI AstroEntangle Refiner
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
import matplotlib.pyplot as plt

def search_abell209_data():
    """Search for Abell 209 observations in MAST"""
    print("Searching for Abell 209 data in MAST...")
    
    # Abell 209 coordinates (RA, Dec in degrees)
    # From literature: Abell 209 center ~ RA 01:31:52.5, Dec -13:36:38
    coord = SkyCoord("01h31m52.5s", "-13d36m38s", frame='icrs')
    
    # Search for HST observations
    obs_table = Observations.query_criteria(
        coordinates=coord,
        radius=0.1 * u.deg,
        obs_collection="HST",
        dataproduct_type="image"
    )
    
    print(f"Found {len(obs_table)} observations")
    
    if len(obs_table) > 0:
        print("\nAvailable observations:")
        for i, obs in enumerate(obs_table[:10]):  # Show first 10
            print(f"  {i+1}. {obs['obs_id']} - {obs['obs_collection']}")
    
    return obs_table

def download_abell209_data(output_dir="./abell209_data"):
    """Download Abell 209 HST data"""
    os.makedirs(output_dir, exist_ok=True)
    
    coord = SkyCoord("01h31m52.5s", "-13d36m38s", frame='icrs')
    
    # Get products
    obs_table = Observations.query_criteria(
        coordinates=coord,
        radius=0.05 * u.deg,
        obs_collection="HST",
        dataproduct_type="image"
    )
    
    if len(obs_table) == 0:
        print("No data found. Trying alternative search...")
        # Try without radius constraint
        obs_table = Observations.query_criteria(
            coordinates=coord,
            obs_collection="HST",
            dataproduct_type="image"
        )
    
    if len(obs_table) > 0:
        # Get product list
        products = Observations.get_product_list(obs_table)
        
        # Filter for calibrated FITS files (FLT, FLC, or DRZ)
        mask = ['.fits' in p['productFilename'].lower() for p in products]
        products = products[mask]
        
        mask = ['flt' in p['productFilename'].lower() or 
                'flc' in p['productFilename'].lower() or 
                'drz' in p['productFilename'].lower() 
                for p in products]
        products = products[mask]
        
        print(f"Found {len(products)} FITS files to download")
        
        # Download first few files (limit to avoid too much data)
        for i, product in enumerate(products[:3]):
            print(f"Downloading {product['productFilename']}...")
            try:
                Observations.download_products(
                    product,
                    download_dir=output_dir,
                    cache=False
                )
            except Exception as e:
                print(f"  Error downloading: {e}")
    else:
        print("No Abell 209 observations found.")
        
    return output_dir

def create_sample_fits_if_needed(output_dir="./abell209_data"):
    """
    If no data is available from MAST, create a sample FITS file
    for demonstration purposes
    """
    sample_file = os.path.join(output_dir, "abell209_sample.fits")
    
    if os.path.exists(sample_file):
        return sample_file
    
    print("\nCreating sample Abell 209-like FITS file for demonstration...")
    
    # Create synthetic image of a galaxy cluster
    size = 1024
    image = np.zeros((size, size))
    
    # Center coordinates
    cx, cy = size//2, size//2
    
    # Main cluster component (cD galaxy)
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    r2 = (X - cx)**2 + (Y - cy)**2
    
    # Bright central galaxy
    central = 1000 * np.exp(-r2 / (2 * 30**2))
    image += central
    
    # Multiple galaxy components (subclusters)
    subclusters = [
        (cx - 80, cy - 60, 25, 300),
        (cx + 70, cy - 50, 28, 250),
        (cx - 40, cy + 90, 32, 200),
        (cx + 100, cy + 40, 22, 180),
        (cx + 30, cy - 120, 35, 150),
    ]
    
    for xc, yc, sigma, amp in subclusters:
        r2_sub = (X - xc)**2 + (Y - yc)**2
        image += amp * np.exp(-r2_sub / (2 * sigma**2))
    
    # Add diffuse intracluster light
    image += 50 * np.exp(-r2 / (2 * 200**2))
    
    # Add background noise
    noise = np.random.normal(0, 10, image.shape)
    image += noise
    
    # Add some gradient (simulating background variations)
    gradient = np.outer(np.linspace(0, 20, size), np.ones(size))
    image += gradient
    
    # Ensure positive values
    image = np.maximum(image, 0)
    
    # Create FITS header with WCS info
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    hdu.header['OBJECT'] = 'Abell 209'
    hdu.header['RA'] = 22.96875  # ~01:31:52.5 in degrees
    hdu.header['DEC'] = -13.61056
    hdu.header['CDELT1'] = 0.0000278  # ~0.1 arcsec/pixel
    hdu.header['CDELT2'] = 0.0000278
    hdu.header['FILTER'] = 'F814W'
    hdu.header['TELESCOP'] = 'HST'
    hdu.header['INSTRUME'] = 'ACS/WFC'
    
    hdu.writeto(sample_file, overwrite=True)
    print(f"Created sample file: {sample_file}")
    
    return sample_file

def display_fits_info(fits_file):
    """Display information about a FITS file"""
    print(f"\nFITS File: {fits_file}")
    with fits.open(fits_file) as hdul:
        print(f"  Number of HDUs: {len(hdul)}")
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'data') and hdu.data is not None:
                print(f"  HDU {i}: {hdu.name}, shape={hdu.data.shape}, dtype={hdu.data.dtype}")
                if 'OBJECT' in hdu.header:
                    print(f"    OBJECT: {hdu.header['OBJECT']}")
                if 'FILTER' in hdu.header:
                    print(f"    FILTER: {hdu.header['FILTER']}")
                break

if __name__ == "__main__":
    print("=" * 60)
    print("Abell 209 Data Download for QCI AstroEntangle Refiner")
    print("=" * 60)
    
    # Try to download real data
    print("\n[1] Searching for Abell 209 HST data...")
    obs_table = search_abell209_data()
    
    if len(obs_table) > 0:
        print("\n[2] Downloading data...")
        output_dir = download_abell209_data()
    else:
        print("\n[2] No real data found. Creating sample FITS file...")
        output_dir = "./abell209_data"
    
    # Create sample if needed
    sample_file = create_sample_fits_if_needed(output_dir)
    
    # Display info about the file
    display_fits_info(sample_file)
    
    print("\n" + "=" * 60)
    print("Data ready! You can now process with QCI AstroEntangle Refiner")
    print(f"FITS file: {sample_file}")
    print("=" * 60)
