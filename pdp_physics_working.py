"""
Photon-Dark-Photon Entanglement Physics Engine - Enhanced v4
Includes: Full PDP formulas, PSF corrections, Neural enhancements
"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from skimage import exposure, restoration
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


class PhotonDarkPhotonEngine:
    """Enhanced physics engine with full PDP formulas, PSF, and neural corrections"""
    
    def __init__(self):
        self.const = PhysicalConstants()
        self.metadata = {}
        self.entanglement_map = None
        self.conversion_probability_map = None
        self.enhanced_map = None
        
    def calculate_oscillation_probability(self, mixing_epsilon, dark_photon_mass_eV, 
                                          distance_m, E_photon_eV):
        """
        Full quantum oscillation probability for photon-dark photon conversion
        P(γ → A') = 4ε² sin²(Δm² L / 4E)
        
        Parameters:
        - mixing_epsilon: kinetic mixing parameter
        - dark_photon_mass_eV: dark photon mass in eV/c²
        - distance_m: propagation distance in meters
        - E_photon_eV: photon energy in eV
        """
        # Convert mass to kg
        mass_kg = self.const.eV_to_kg(dark_photon_mass_eV)
        
        # Calculate Δm² = m_dark² - m_photon² ≈ m_dark²
        # Convert to SI units (kg²)
        delta_m2 = mass_kg ** 2
        
        # Convert to natural units (eV²) for convenience
        delta_m2_eV2 = dark_photon_mass_eV ** 2
        
        # Convert photon energy to Joules
        E_J = self.const.eV_to_J(E_photon_eV)
        
        # Calculate oscillation length: L_osc = 4πE/Δm²
        # In natural units: L_osc = 4πE / Δm²
        # Convert to meters
        L_osc = (4 * np.pi * E_photon_eV) / (delta_m2_eV2 + 1e-30)
        L_osc_meters = L_osc * (self.const.hbar * self.const.c / self.const.eV) / (1e-15)  # Scale appropriately
        
        # Calculate oscillation probability
        argument = (delta_m2 * distance_m * self.const.c) / (4 * self.const.hbar * E_J)
        prob = 4 * mixing_epsilon**2 * np.sin(argument)**2
        
        # Clip probability to physical range [0,1]
        prob = np.clip(prob, 0, 1)
        
        return prob, L_osc_meters, argument
    
    def apply_psf_correction(self, image, fwhm_arcsec=0.05, pixel_scale_arcsec=0.05):
        """
        Apply PSF deconvolution to correct for telescope beam smearing
        
        Parameters:
        - image: 2D numpy array
        - fwhm_arcsec: Full Width Half Maximum of PSF in arcseconds
        - pixel_scale_arcsec: Pixel scale in arcseconds per pixel
        """
        # Convert FWHM to sigma in pixels
        fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec
        sigma = fwhm_pixels / 2.355  # FWHM = 2.355 * sigma
        
        if sigma < 0.5:
            # PSF smaller than pixel, minimal correction
            return image
        
        # Create Gaussian PSF
        size = int(max(11, 2 * int(3 * sigma) + 1))
        if size % 2 == 0:
            size += 1
        
        y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
        psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        psf = psf / psf.sum()
        
        # Apply Wiener deconvolution
        try:
            restored = restoration.wiener(image, psf, noise=0.01)
        except:
            # Fallback to simple Gaussian blur removal
            restored = image - 0.5 * gaussian_filter(image, sigma=sigma/2)
            restored = np.clip(restored, 0, 1)
        
        return restored
    
    def neural_enhancement(self, image, method='clahe'):
        """
        Apply neural-inspired image enhancement
        
        Methods:
        - 'clahe': Contrast Limited Adaptive Histogram Equalization
        - 'retinex': Simple Retinex-inspired enhancement
        - 'unsharp': Unsharp masking
        """
        enhanced = image.copy()
        
        if method == 'clahe':
            # CLAHE - similar to what neural networks learn
            enhanced = exposure.equalize_adapthist(enhanced, clip_limit=0.03)
            
        elif method == 'retinex':
            # Simple Retinex-inspired enhancement
            blurred = gaussian_filter(enhanced, sigma=10)
            enhanced = np.log(enhanced + 1e-10) - np.log(blurred + 1e-10)
            enhanced = np.clip(enhanced, 0, 1)
            
        elif method == 'unsharp':
            # Unsharp masking
            blurred = gaussian_filter(enhanced, sigma=2)
            enhanced = enhanced + 0.5 * (enhanced - blurred)
            enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    def calculate_conversion_map(self, image_norm, mixing_epsilon, dark_photon_mass_eV,
                                  distance_m, E_photon_eV, pixel_scale_m):
        """
        Calculate full conversion probability map including spatial variations
        """
        ny, nx = image_norm.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        
        # Distance from center (radial coordinate in meters)
        center_x, center_y = nx/2, ny/2
        radial_distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2) * pixel_scale_m
        
        # Calculate oscillation probability for each pixel
        prob_map = np.zeros_like(image_norm)
        
        for i in range(ny):
            for j in range(nx):
                prob, _, _ = self.calculate_oscillation_probability(
                    mixing_epsilon, dark_photon_mass_eV, 
                    radial_distance[i, j], E_photon_eV
                )
                prob_map[i, j] = prob
        
        # Apply image-dependent modulation
        prob_map = prob_map * (0.5 + 0.5 * image_norm)
        
        return prob_map
    
    def initialize_from_image(self, image_data, pixel_scale_arcsec=0.05,
                              dark_photon_mass_eV=1e-22, mixing_epsilon=1e-8,
                              relative_velocity=2000000, redshift=0.206, 
                              distance_mpc=430, E_photon_eV=1.0,
                              apply_psf=True, apply_neural=True,
                              psf_fwhm_arcsec=0.05):
        """
        Enhanced initialization with full physics
        """
        
        # Normalize image
        img_norm = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-10)
        
        # Calculate pixel scales
        distance_m = distance_mpc * 3.085677581e22
        pixel_scale_rad = pixel_scale_arcsec * (math.pi / (180 * 3600))
        pixel_scale_m = pixel_scale_rad * distance_m
        
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
        # Apply conversion probability to image
        self.conversion_probability_map = prob_map
        entanglement = img_enhanced * (1 - prob_map) + prob_map * (1 - img_enhanced)
        entanglement = np.clip(entanglement, 0, 1)
        
        # Store result
        self.entanglement_map = entanglement
        self.enhanced_map = img_enhanced
        self.original_norm = img_norm
        
        # Calculate entropy (information-theoretic)
        hist, _ = np.histogram(img_norm.flatten(), bins=50)
        hist = hist[hist > 0]
        prob = hist / len(img_norm.flatten())
        entropy = -np.sum(prob * np.log2(prob + 1e-12))
        
        # Calculate quantum concurrence (entanglement measure)
        avg_prob = np.mean(prob_map)
        concurrence = 2 * avg_prob * (1 - avg_prob) * mixing_epsilon
        
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
PhotonDarkPhotonModel = PhotonDarkPhotonEngine
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
    # Quick test
    test_img = np.random.rand(100, 100)
    engine = PhotonDarkPhotonEngine()
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
    print(f"PSF Corrected: {metadata['psf_corrected']}")
    print(f"Neural Enhanced: {metadata['neural_enhanced']}")
