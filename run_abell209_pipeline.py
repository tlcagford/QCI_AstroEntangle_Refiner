# run_abell209_pipeline.py
"""
Run QCI AstroEntangle Refiner pipeline on Abell 209 data
"""

import sys
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Add the repository to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the physics engine
try:
    from pdp_physics_working import PhotonDarkPhotonEngine, PhysicalConstants, DarkPhotonParameters
    print("✓ Physics engine loaded")
except ImportError as e:
    print(f"✗ Could not load physics engine: {e}")
    print("  Make sure pdp_physics_working.py is in the same directory")
    sys.exit(1)

def load_fits_data(fits_file):
    """Load FITS data and extract image"""
    print(f"\nLoading {fits_file}...")
    
    with fits.open(fits_file) as hdul:
        # Find the first image HDU
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'data') and hdu.data is not None:
                if len(hdu.data.shape) >= 2:
                    data = hdu.data
                    header = hdu.header
                    print(f"  Image shape: {data.shape}")
                    print(f"  Filter: {header.get('FILTER', 'Unknown')}")
                    print(f"  Exposure: {header.get('EXPTIME', 'Unknown')} s")
                    
                    # Handle 3D data (take first slice)
                    if len(data.shape) == 3:
                        data = data[0]
                        print(f"  Using first slice: {data.shape}")
                    
                    # Normalize
                    data = data.astype(np.float32)
                    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
                    
                    return data, header
    
    raise ValueError(f"No image data found in {fits_file}")

def run_physics_on_abell209(fits_file, output_dir="./abell209_output"):
    """Run the full entanglement physics on Abell 209"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    image_data, header = load_fits_data(fits_file)
    
    print(f"\n{'='*60}")
    print("Running Photon-Dark-Photon Entanglement Pipeline on Abell 209")
    print(f"{'='*60}")
    
    # Initialize physics engine
    engine = PhotonDarkPhotonEngine()
    
    # Get pixel scale from header (if available)
    pixel_scale_arcsec = 0.1  # Default for ACS/WFC
    if 'CDELT1' in header:
        pixel_scale_arcsec = abs(header['CDELT1']) * 3600
    elif 'CD1_1' in header:
        pixel_scale_arcsec = abs(header['CD1_1']) * 3600
    
    print(f"\n[1] Physics Parameters:")
    print(f"    Pixel scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
    print(f"    Image size: {image_data.shape[0]}x{image_data.shape[1]}")
    
    # Configure dark photon parameters
    # For galaxy clusters like Abell 209, we expect:
    # - Dark matter mass ~10^-22 eV (FDM scale)
    # - Relative velocities ~100-1000 km/s
    # - Mixing parameter from experimental bounds
    
    dark_photon_mass_eV = 1e-22  # FDM scale
    mixing_epsilon = 1e-8         # Typical bound
    relative_velocity = 5e5       # 500 km/s for cluster
    
    print(f"    Dark photon mass: {dark_photon_mass_eV:.1e} eV")
    print(f"    Mixing ε: {mixing_epsilon:.1e}")
    print(f"    Relative velocity: {relative_velocity/1000:.0f} km/s")
    
    # Initialize with image
    print(f"\n[2] Initializing physics engine...")
    metadata = engine.initialize_from_image(
        image_data=image_data,
        pixel_scale_arcsec=pixel_scale_arcsec,
        dark_photon_mass_eV=dark_photon_mass_eV,
        mixing_epsilon=mixing_epsilon,
        relative_velocity=relative_velocity
    )
    
    # Display results
    print(f"\n[3] Entanglement Results:")
    print(f"    von Neumann Entropy: {metadata['entropy']:.6f} bits")
    print(f"    Concurrence: {metadata['concurrence']:.6f}")
    print(f"    de Broglie wavelength: {metadata['de_broglie_wavelength_m']:.2e} m")
    print(f"    Predicted fringe spacing: {metadata['predicted_fringe_spacing_px']:.1f} pixels")
    
    # Get entanglement map
    entanglement_map = engine.get_entanglement_map()
    
    print(f"\n[4] Entanglement Map Statistics:")
    print(f"    Shape: {entanglement_map.shape}")
    print(f"    Range: [{entanglement_map.min():.3f}, {entanglement_map.max():.3f}]")
    print(f"    Mean: {entanglement_map.mean():.3f}")
    print(f"    Std: {entanglement_map.std():.3f}")
    
    # Save results
    print(f"\n[5] Saving results to {output_dir}/")
    
    # Save as FITS
    output_fits = os.path.join(output_dir, "abell209_entangled.fits")
    hdu = fits.PrimaryHDU(entanglement_map.astype(np.float32))
    hdu.header['OBJECT'] = 'Abell 209'
    hdu.header['COMMENT'] = 'Photon-Dark-Photon Entanglement Map'
    hdu.header['ENTROPY'] = metadata['entropy']
    hdu.header['CONCURR'] = metadata['concurrence']
    hdu.header['MIXING'] = metadata['mixing_epsilon']
    hdu.header['DARKMASS'] = metadata['dark_photon_mass_eV']
    hdu.writeto(output_fits, overwrite=True)
    print(f"    Saved: {output_fits}")
    
    # Save metadata as text
    metadata_file = os.path.join(output_dir, "abell209_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("Abell 209 - Photon-Dark-Photon Entanglement Results\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"    Saved: {metadata_file}")
    
    # Create visualization
    print(f"\n[6] Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    im1 = axes[0].imshow(image_data, cmap='gray', origin='lower')
    axes[0].set_title('Original Abell 209')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Entanglement map
    im2 = axes[1].imshow(entanglement_map, cmap='plasma', origin='lower')
    axes[1].set_title('Entanglement Map\n(Photon-Dark-Photon)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference (enhanced)
    diff = entanglement_map - image_data
    im3 = axes[2].imshow(diff, cmap='RdBu', origin='lower')
    axes[2].set_title('Difference Map\n(Entanglement - Original)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Abell 209 - Photon-Dark-Photon Entanglement\n'
                 f'S = {metadata["entropy"]:.4f} bits, C = {metadata["concurrence"]:.4f}')
    
    plt.tight_layout()
    output_png = os.path.join(output_dir, "abell209_entanglement.png")
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_png}")
    
    print(f"\n{'='*60}")
    print("✅ Abell 209 processing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return metadata, entanglement_map

def main():
    """Main execution"""
    
    # Find the FITS file
    fits_file = None
    
    # Check for downloaded data
    data_dirs = ["./abell209_data", "./mastDownload/HST"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.fits'):
                        fits_file = os.path.join(root, f)
                        break
            if fits_file:
                break
    
    # If no data found, create sample
    if not fits_file or not os.path.exists(fits_file):
        print("No Abell 209 data found. Running with sample data...")
        from download_abell209 import create_sample_fits_if_needed
        sample_file = create_sample_fits_if_needed("./abell209_data")
        fits_file = sample_file
    
    # Run the pipeline
    metadata, entanglement_map = run_physics_on_abell209(fits_file)
    
    return metadata, entanglement_map

if __name__ == "__main__":
    main()
