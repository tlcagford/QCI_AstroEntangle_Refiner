# Create the physics engine file
cat > pdp_physics_working.py << 'EOF'
"""
Photon-Dark-Photon Entanglement Physics Engine
Simple but complete working implementation
"""

import numpy as np
import math

class PhysicalConstants:
    """Physical constants in SI units"""
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    
    @staticmethod
    def eV_to_kg(eV):
        """Convert eV/c² to kg"""
        return eV * 1.78266192e-36
    
    @staticmethod
    def kg_to_eV(kg):
        """Convert kg to eV/c²"""
        return kg / 1.78266192e-36


class PhotonDarkPhotonEngine:
    """Main physics engine for photon-dark-photon entanglement"""
    
    def __init__(self):
        self.const = PhysicalConstants()
        self.metadata = {}
        self.entanglement_map = None
        
    def initialize_from_image(self, image_data, pixel_scale_arcsec=0.05,
                              dark_photon_mass_eV=1e-22, mixing_epsilon=1e-8,
                              relative_velocity=2000000, redshift=0.206, distance_mpc=430):
        """
        Calculate entanglement map from astronomical image
        
        Parameters:
        - image_data: 2D numpy array of the image
        - pixel_scale_arcsec: angular resolution (arcseconds per pixel)
        - dark_photon_mass_eV: dark photon mass in eV/c²
        - mixing_epsilon: kinetic mixing parameter
        - relative_velocity: merger velocity in m/s
        - redshift: cosmological redshift
        - distance_mpc: distance to cluster in Mpc
        """
        
        # Normalize image
        img_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-10)
        
        # Calculate de Broglie wavelength in pixels
        distance_m = distance_mpc * 3.085677581e22  # Mpc to meters
        pixel_scale_rad = pixel_scale_arcsec * (math.pi / (180 * 3600))
        pixel_scale_m = pixel_scale_rad * distance_m
        
        mass_kg = self.const.eV_to_kg(dark_photon_mass_eV)
        de_broglie = self.const.h / (mass_kg * relative_velocity)
        fringe_px = de_broglie / pixel_scale_m
        
        # Create interference pattern
        ny, nx = img_norm.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        
        pattern = np.sin(2 * np.pi * X / max(fringe_px, 1)) * np.sin(2 * np.pi * Y / max(fringe_px, 1))
        
        # Apply mixing to create entanglement map
        entanglement = img_norm + mixing_epsilon * 10 * pattern * img_norm
        entanglement = np.clip(entanglement, 0, 1)
        
        # Store result
        self.entanglement_map = entanglement
        
        # Calculate entropy (simplified)
        hist, _ = np.histogram(img_norm.flatten(), bins=50)
        hist = hist[hist > 0]
        prob = hist / len(img_norm.flatten())
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        
        # Store metadata
        self.metadata = {
            'entropy': entropy,
            'concurrence': mixing_epsilon * 0.1,
            'purity': 1 - mixing_epsilon**2,
            'de_broglie_wavelength_px': fringe_px,
            'dark_photon_mass_eV': dark_photon_mass_eV,
            'mixing_epsilon': mixing_epsilon,
            'fringe_spacing_px': fringe_px
        }
        
        return self.metadata
    
    def get_entanglement_map(self):
        """Return the computed entanglement map"""
        return self.entanglement_map
    
    def get_metadata(self):
        """Return physics metadata"""
        return self.metadata

# Test function
if __name__ == "__main__":
    # Quick test
    test_img = np.random.rand(100, 100)
    engine = PhotonDarkPhotonEngine()
    metadata = engine.initialize_from_image(test_img)
    print("Physics engine working!")
    print(f"Entropy: {metadata['entropy']:.5f} bits")
    print(f"Fringe spacing: {metadata['fringe_spacing_px']:.1f} pixels")
EOF

# Verify it was created
echo "Checking file..."
cat pdp_physics_working.py | head -20
# Compatibility aliases for v4
PhotonDarkPhotonModel = PhotonDarkPhotonEngine
H = PhysicalConstants.h
HBAR = PhysicalConstants.hbar
C = PhysicalConstants.c
ALPHA = 1/137.035999084  # Fine structure constant
M_E = 9.1093837015e-31   # Electron mass in kg
EV = 1.602176634e-19     # Electron volt in Joules
K_B = 1.380649e-23       # Boltzmann constant
EPS0 = 8.8541878128e-12  # Vacuum permittivity
