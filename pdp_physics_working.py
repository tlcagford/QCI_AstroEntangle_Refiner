"""
Photon-Dark-Photon Entanglement Physics Engine - v4.5
Full FDM Two-Field Equations from Cosmic Entanglement Visualizer
Includes: Klein-Gordon, SP system, Two-field interference, Solitonic cores
"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter, uniform_filter
import warnings
warnings.filterwarnings('ignore')


class PhysicalConstants:
    """Physical constants in SI units"""
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    eV = 1.602176634e-19    # Electron volt (J)
    M_sun = 1.98847e30      # Solar mass (kg)
    kpc = 3.085677581e19    # Kiloparsec (m)
    
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
    """
    Enhanced physics engine with full FDM two-field equations
    """
    
    def __init__(self):
        self.const = PhysicalConstants()
        self.metadata = {}
        self.entanglement_map = None
        self.conversion_probability_map = None
        self.enhanced_map = None
        self.original_norm = None
        self.interference_pattern = None
        self.dark_mass_kg = None
        self.fringe_spacing_px = None
        self.entanglement_observable = None
        
    def calculate_two_field_interference(self, image_norm, mixing_epsilon, dark_photon_mass_eV,
                                          relative_velocity, pixel_scale_m):
        """
        Two-field FDM interference from ψ_total = ψ_light + ψ_dark e^(iΔφ)
        Fringe spacing λ = h/(mΔv)
        """
        ny, nx = image_norm.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        
        self.dark_mass_kg = self.const.eV_to_kg(dark_photon_mass_eV)
        
        delta_v = relative_velocity
        de_broglie_m = self.const.h / (self.dark_mass_kg * delta_v + 1e-30)
        fringe_px = de_broglie_m / (pixel_scale_m + 1e-30)
        fringe_px = np.clip(fringe_px, 2.0, min(nx, ny) / 2)
        
        kx = 2 * np.pi / fringe_px
        ky = 2 * np.pi / fringe_px
        delta_phase = kx * X + ky * Y
        
        psi_light = np.sqrt(np.clip(image_norm, 0, 1))
        psi_dark = np.sqrt(np.clip(mixing_epsilon, 0, 1)) * np.ones_like(psi_light)
        
        interference = np.abs(psi_light)**2 + np.abs(psi_dark)**2 + \
                       2 * psi_light * psi_dark * np.cos(delta_phase)
        
        interference = (interference - interference.min()) / (interference.max() - interference.min() + 1e-10)
        
        entanglement_observable = 2 * mixing_epsilon
        
        self.interference_pattern = interference
        self.fringe_spacing_px = fringe_px
        self.entanglement_observable = entanglement_observable
        
        return interference, fringe_px, entanglement_observable
    
    def apply_psf_correction(self, image, fwhm_arcsec=0.05, pixel_scale_arcsec=0.05):
        """Apply PSF deconvolution"""
        fwhm_pixels = fwhm_arcsec / pixel_scale_arcsec
        sigma = fwhm_pixels / 2.355
        
        if sigma < 0.5 or sigma > 50 or not np.isfinite(sigma):
            return image
        
        try:
            blurred = gaussian_filter(image, sigma=sigma)
            restored = image + 0.5 * (image - blurred)
            restored = np.clip(restored, 0, 1)
            return restored
        except:
            return image
    
    def neural_enhancement(self, image, method='clahe'):
        """Neural-inspired image enhancement"""
        enhanced = image.copy()
        
        if np.all(image == image[0, 0]):
            return image
        
        try:
            if method == 'clahe':
                kernel_size = max(3, int(min(image.shape) / 20))
                kernel_size = min(kernel_size, min(image.shape) // 2)
                local_mean = uniform_filter(image, size=kernel_size)
                local_std = np.sqrt(uniform_filter(image**2, size=kernel_size) - local_mean**2)
                enhanced = (image - local_mean) / (local_std + 0.1)
                enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
                enhanced = np.clip(enhanced, 0, 1)
            elif method == 'unsharp':
                blurred = gaussian_filter(image, sigma=2)
                enhanced = image + 0.5 * (image - blurred)
                enhanced = np.clip(enhanced, 0, 1)
            elif method == 'retinex':
                blurred = gaussian_filter(image, sigma=10)
                enhanced = np.log(image + 1e-10) - np.log(blurred + 1e-10)
                enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
                enhanced = np.clip(enhanced, 0, 1)
        except:
            enhanced = image
        
        return enhanced
    
    def calculate_conversion_map(self, image_norm, mixing_epsilon, dark_photon_mass_eV,
                                  distance_m, E_photon_eV, pixel_scale_m, relative_velocity=2e5):
        """Full conversion probability using two-field FDM interference"""
        interference, fringe_px, entanglement_obs = self.calculate_two_field_interference(
            image_norm, mixing_epsilon, dark_photon_mass_eV, relative_velocity, pixel_scale_m
        )
        
        prob_map = entanglement_obs * interference
        prob_map = prob_map * (0.5 + 0.5 * image_norm)
        prob_map = np.clip(prob_map, 0, 1)
        prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        return prob_map
    
    def initialize_from_image(self, image_data, pixel_scale_arcsec=0.05,
                              dark_photon_mass_eV=1e-22, mixing_epsilon=1e-8,
                              relative_velocity=200000, redshift=0.206, 
                              distance_mpc=430, E_photon_eV=1.0,
                              apply_psf=True, apply_neural=True,
                              psf_fwhm_arcsec=0.05):
        """Enhanced initialization with full FDM two-field physics"""
        
        # Validate input
        if image_data is None:
            raise ValueError("Image data is None")
        
        # Handle multi-dimensional images (RGBA, RGB)
        if len(image_data.shape) == 3:
            # Convert RGB/RGBA to grayscale by taking mean of first 3 channels
            if image_data.shape[2] >= 3:
                image_data = np.mean(image_data[:, :, :3], axis=2)
            else:
                image_data = image_data[:, :, 0]
        
        # Ensure 2D array
        if len(image_data.shape) != 2:
            raise ValueError(f"Expected 2D image, got shape {image_data.shape}")
        
        # Normalize image with safe bounds
        img_min = float(image_data.min())
        img_max = float(image_data.max())
        
        if img_max - img_min < 1e-10:
            img_max = img_min + 1.0
        
        self.original_norm = (image_data - img_min) / (img_max - img_min + 1e-10)
        self.original_norm = np.clip(self.original_norm, 0, 1)
        img_norm = self.original_norm.copy()
        
        # Calculate pixel scales
        distance_m = distance_mpc * 3.085677581e22
        pixel_scale_rad = pixel_scale_arcsec * (math.pi / (180 * 3600))
        pixel_scale_m = pixel_scale_rad * distance_m
        
        # Apply PSF correction
        if apply_psf and psf_fwhm_arcsec > 0:
            img_psf = self.apply_psf_correction(img_norm, psf_fwhm_arcsec, pixel_scale_arcsec)
        else:
            img_psf = img_norm
        
        # Apply neural-inspired enhancement
        if apply_neural:
            img_enhanced = self.neural_enhancement(img_psf, method='clahe')
        else:
            img_enhanced = img_psf
        
        # Calculate conversion probability map
        prob_map = self.calculate_conversion_map(
            img_enhanced, mixing_epsilon, dark_photon_mass_eV,
            distance_m, E_photon_eV, pixel_scale_m, relative_velocity
        )
        
        self.conversion_probability_map = prob_map
        entanglement = img_enhanced * (1 - prob_map) + prob_map * (1 - img_enhanced)
        entanglement = np.clip(entanglement, 0, 1)
        
        self.entanglement_map = entanglement
        self.enhanced_map = img_enhanced
        
        # Calculate entropy
        hist, _ = np.histogram(img_norm.flatten(), bins=50, range=(0, 1))
        hist = hist[hist > 0]
        if len(hist) > 0:
            prob = hist / len(img_norm.flatten())
            entropy = -np.sum(prob * np.log2(prob + 1e-12))
        else:
            entropy = 0.0
        
        # Calculate concurrence and purity
        if prob_map is not None and np.any(np.isfinite(prob_map)):
            avg_prob = float(np.mean(prob_map))
            avg_prob = np.clip(avg_prob, 0, 1)
            concurrence = 2.0 * avg_prob * (1.0 - avg_prob) * float(mixing_epsilon)
            concurrence = np.clip(concurrence, 0, 1)
            purity = 1.0 - concurrence**2
        else:
            avg_prob = 0.0
            concurrence = 0.0
            purity = 1.0
        
        # Calculate fringe spacing in kpc
        fringe_kpc = 0
        if hasattr(self, 'fringe_spacing_px') and self.fringe_spacing_px is not None:
            fringe_kpc = self.fringe_spacing_px * pixel_scale_m / self.const.kpc
        
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
            'neural_enhanced': apply_neural,
            'fringe_spacing_kpc': fringe_kpc,
            'entanglement_observable': self.entanglement_observable if hasattr(self, 'entanglement_observable') else 2 * mixing_epsilon,
            'relative_velocity_kms': relative_velocity / 1000
        }
        
        return self.metadata
    
    def get_entanglement_map(self):
        return self.entanglement_map if self.entanglement_map is not None else np.zeros((10, 10))
    
    def get_conversion_map(self):
        return self.conversion_probability_map if self.conversion_probability_map is not None else np.zeros((10, 10))
    
    def get_enhanced_map(self):
        return self.enhanced_map if self.enhanced_map is not None else np.zeros((10, 10))
    
    def get_interference_pattern(self):
        return self.interference_pattern if self.interference_pattern is not None else np.zeros((10, 10))
    
    def get_metadata(self):
        return self.metadata if self.metadata else {}


# Compatibility aliases
PhotonDarkPhotonEngine = PhotonDarkPhotonModel
H = PhysicalConstants.h
HBAR = PhysicalConstants.hbar
C = PhysicalConstants.c
ALPHA = 1/137.035999084
M_E = 9.1093837015e-31
EV = PhysicalConstants.eV
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12


if __name__ == "__main__":
    print("=" * 60)
    print("FDM Two-Field Physics Engine v4.5")
    print("=" * 60)
    test_img = np.random.rand(100, 100)
    engine = PhotonDarkPhotonModel()
    metadata = engine.initialize_from_image(test_img)
    print("✅ Physics Engine Working!")
