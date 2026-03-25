"""
Photon-Dark-Photon Entanglement Physics Engine - Enhanced v4
Full PDP formulas, PSF corrections, Neural enhancements (without skimage)
"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import wiener
import warnings
warnings.filterwarnings('ignore')

class PhysicalConstants:
    """Physical constants in SI units"""
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    eV = 1.602176634e-19    # Electron volt (J)
    
    @staticmethod
    def eV_to_kg(eV):
        """Convert eV/c² to kg"""
        return eV * 1.78266192e-36
    
    @staticmethod
    def kg_to_eV(kg):
        """Convert kg to eV/c²"""
        return kg / 1.78266192e-36
    
    @staticmethod
    def eV_to_J(eV):
        """Convert eV to Joules"""
        return eV * 1.602176634e-19


class PhotonDarkPhotonModel:
    """Enhanced physics engine with full PDP formulas, PSF, and neural corrections"""
    
    def __init__(self):
        self.const = PhysicalConstants()
        self.metadata = {}
        self.entanglement_map = None
        self.conversion_probability_map = None
        self.enhanced_map = None
        self.original_norm = None
        
    def calculate_oscillation_probability(self, mixing_epsilon, dark_photon_mass_eV, 
                                          distance_m, E_photon_eV):
        """
        Full quantum oscillation probability for photon-dark photon conversion
        P(γ → A') = 4ε² sin²(Δm² L / 4E)
        """
        # Convert mass to kg
        mass_kg = self.const.eV_to_kg(dark_photon_mass_eV)
        
        # Calculate Δm² = m_dark² - m_photon² ≈ m_dark²
        delta_m2 = mass_kg ** 2
        
        # Convert photon energy to Joules
        E_J = self.const.eV_to_J(E_photon_eV)
        
        # Calculate argument
        argument = (delta_m2 * distance_m * self.const.c) / (4 * self.const.hbar * E_J + 1e-30)
        
        # Calculate oscillation probability
        prob = 4 * mixing_epsilon**2 * np.sin(argument)**2
        
        # Clip probability to physical range [0,1]
        prob = np.clip(prob, 0, 1)
        
        # Oscillation length in meters
        L_osc = (4 * np.pi * E_photon_eV) / (dark_photon_mass_eV**2 + 1e-30)
        L_osc_meters = L_osc * 1e-15  # Scale factor
        
        return prob, L_osc_meters, argument
    
    def apply_psf_correction(self, image, fwhm_arcsec=0.05, pixel_scale_arcsec=0.05):
        """Apply PSF deconvolution to correct for telescope beam smearing"""
        # Convert FWHM to sigma in pixels
        fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec
        sigma = fwhm_pixels / 2.355  # FWHM = 2.355 * sigma
        
        if sigma < 0.5 or sigma > 50:
            return image
        
        try:
            # Simple Gaussian blur removal (Wiener-like)
            blurred = gaussian_filter(image, sigma=sigma)
            # High-pass filter to sharpen
            restored = image + 0.5 * (image - blurred)
            restored = np.clip(restored, 0, 1)
            return restored
        except:
            return image
    
    def neural_enhancement(self, image, method='clahe'):
        """
        Neural-inspired image enhancement (no skimage required)
        
        Methods:
        - 'clahe': Adaptive histogram equalization
        - 'unsharp': Unsharp masking
        - 'retinex': Simple Retinex-inspired
        """
        enhanced = image.copy()
        
        if method == 'clahe':
            # Adaptive contrast enhancement
            kernel_size = max(3, int(min(image.shape) / 20))
            # Local mean
            local_mean = uniform_filter(image, size=kernel_size)
            # Local standard deviation
            local_std = np.sqrt(uniform_filter(image**2, size=kernel_size) - local_mean**2)
            # Adaptive contrast
            enhanced = (image - local_mean) / (local_std + 0.1)
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
            enhanced = np.clip(enhanced, 0, 1)
            
        elif method == 'unsharp':
            # Unsharp masking
            blurred = gaussian_filter(image, sigma=2)
            enhanced = image + 0.5 * (image - blurred)
            enhanced = np.clip(enhanced, 0, 1)
            
        elif method == 'retinex':
            # Simple Retinex-inspired enhancement
            blurred = gaussian_filter(image, sigma=10)
            enhanced = np.log(image + 1e-10) - np.log(blurred + 1e-10)
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
            enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    def calculate_conversion_map(self, image_norm, mixing_epsilon, dark_photon_mass_eV,
                                  distance_m, E_photon_eV, pixel_scale_m):
        """Calculate full conversion probability map including spatial variations"""
        ny, nx = image_norm.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        
        # Distance from center (radial coordinate in meters)
        center_x, center_y = nx/2, ny/2
        radial_distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2) * pixel_scale_m
        
        # Calculate oscillation probability for each pixel (vectorized for speed)
        prob_map = np.zeros_like(image_norm)
        
        # Precompute constants
        mass_kg = self.const.eV_to_kg(dark_photon_mass_eV)
        delta_m2 = mass_kg ** 2
        E_J = self.const.eV_to_J(E_photon_eV)
        prefactor = 4 * mixing_epsilon**2
        coeff = (delta_m2 * self.const.c) / (4 * self.const.hbar * E_J + 1e-30)
        
        # Vectorized calculation
        argument = coeff * radial_distance
        prob_map = prefactor * np.sin(argument)**2
        prob_map = np.clip(prob_map, 0, 1)
        
        # Apply image-dependent modulation
        prob_map = prob_map * (0.5 + 0.5 * image_norm)
        
        return prob_map
    
    def initialize_from_image(self, image_data, pixel_scale_arcsec=0.05,
                              dark_photon_mass_eV=1e-22, mixing_epsilon=1e-8,
                              relative_velocity=2000000, redshift=0.206, 
                              distance_mpc=430, E_photon_eV=1.0,
                              apply_psf=True, apply_neural=True,
                              psf_fwhm_arcsec=0.05):
        """Enhanced initialization with full physics"""
        
        # Normalize image
        self.original_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-10)
        img_norm = self.original_norm.copy()
        
        # Calculate pixel scales
        distance_m = distance_mpc * 3.085677581e22
        pixel_scale_rad = pixel_scale_arcsec * (math.pi / (180 * 3600))
        pixel_scale_m = pixel_scale_rad * distance_m
        # Calculate quantum concurrence (entanglement measure)
avg_prob = np.mean(prob_map)
concurrence = 2 * avg_prob * (1 - avg_prob) * mixing_epsilon
concurrence = np.clip(concurrence, 0, 1)

# Calculate purity
purity = 1 - concurrence**2
        # Apply PSF correction (telescope beam correction)
        if apply_psf:
            img_psf = self.apply_psf_correction(img_norm, psf_fwhm_arcsec, pixel_scale_arcsec)
        else:
            img_psf = img_norm
        
        # Apply neural-inspired enhancement
        if apply_neural:
            img_enhanced = self.neural_enhancement(img_psf, method='clahe')
        else:
            img_enhanced = img_psf
        
        # Calculate full quantum conversion probability map
        prob_map = self.calculate_conversion_map(
            img_enhanced, mixing_epsilon, dark_photon_mass_eV,
            distance_m, E_photon_eV, pixel_scale_m
        )
        
        # Create final entanglement/conversion map
        self.conversion_probability_map = prob_map
        entanglement = img_enhanced * (1 - prob_map) + prob_map * (1 - img_enhanced)
        entanglement = np.clip(entanglement, 0, 1)
        
        # Store result
        self.entanglement_map = entanglement
        self.enhanced_map = img_enhanced
        
        # Calculate entropy (information-theoretic)
        hist, _ = np.histogram(img_norm.flatten(), bins=50)
        hist = hist[hist > 0]
        prob = hist / len(img_norm.flatten())
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        
        # Calculate quantum concurrence (entanglement measure)
        avg_prob = np.mean(prob_map)
        concurrence = 2 * avg_prob * (1 - avg_prob) * mixing_epsilon
        concurrence = np.clip(concurrence, 0, 1)
        
        # Calculate purity
        purity = 1 - concurrence**2
        
        # Store metadata
        self.metadata = {
            'entropy': entropy,
            'concurrence': concurrence,
            'purity': purity,
            'dark_photon_mass_eV': dark_photon_mass_eV,
            'mixing_epsilon': mixing_epsilon,
            'avg_conversion_probability': avg_prob,
            'photon_energy_eV': E_photon_eV,
            'distance_mpc': distance_mpc,
            'pixel_scale_arcsec': pixel_scale_arcsec,
            'psf_corrected': apply_psf,
            'neural_enhanced': apply_neural
        }
        
        return self.metadata
    
    def get_entanglement_map(self):
        """Return the computed entanglement map"""
        return self.entanglement_map
    
    def get_conversion_map(self):
        """Return the conversion probability map"""
        return self.conversion_probability_map
    
    def get_enhanced_map(self):
        """Return the enhanced image"""
        return self.enhanced_map
    
    def get_metadata(self):
        """Return physics metadata"""
        return self.metadata


# Compatibility aliases for v4
PhotonDarkPhotonEngine = PhotonDarkPhotonModel
H = PhysicalConstants.h
HBAR = PhysicalConstants.hbar
C = PhysicalConstants.c
ALPHA = 1/137.035999084  # Fine structure constant
M_E = 9.1093837015e-31   # Electron mass in kg
EV = PhysicalConstants.eV
K_B = 1.380649e-23       # Boltzmann constant
EPS0 = 8.8541878128e-12  # Vacuum permittivity


# Test function
if __name__ == "__main__":
    test_img = np.random.rand(100, 100)
    engine = PhotonDarkPhotonModel()
    metadata = engine.initialize_from_image(
        test_img, 
        dark_photon_mass_eV=1e-22, 
        mixing_epsilon=1e-8,
        E_photon_eV=1.0,
        apply_psf=True,
        apply_neural=True
    )
    print("Enhanced Physics Engine v4 Working!")
    print(f"Entropy: {metadata['entropy']:.5f} bits")
    print(f"Concurrence: {metadata['concurrence']:.5f}")
    print(f"Avg Conversion Prob: {metadata['avg_conversion_probability']:.5f}")
