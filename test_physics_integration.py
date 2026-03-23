"""
Test the physics integration with sample data
"""

import numpy as np
from pdp_physics import PhotonDarkPhotonSystem
import matplotlib.pyplot as plt

# Create synthetic photon field (Gaussian blob)
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
photon_density = np.exp(-(X**2 + Y**2) / 20)  # Simple Gaussian

# Initialize physics system
pdp = PhotonDarkPhotonSystem(
    image_data=photon_density,
    pixel_scale_arcsec=0.1,
    dark_photon_mass=1e-28,  # kg (~10⁻¹⁰ eV)
    mixing_parameter=1e-8,
    relative_velocity=1e5,
    redshift=0.1
)

# Generate entanglement map
entanglement = pdp.generate_entanglement_map()

# Get metadata
print(f"Entanglement Entropy: {pdp.metadata['entropy']:.4f}")
print(f"Predicted fringe spacing: {pdp.metadata['fringe_spacing_px']:.1f} pixels")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(photon_density, cmap='gray')
axes[0].set_title('Original Photon Field')
axes[1].imshow(entanglement, cmap='plasma')
axes[1].set_title('Entangled Photon-Dark Photon Map')
plt.show()
