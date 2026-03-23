"""
Photon-Dark-Photon Entanglement Physics Module
Based on Primordial-Photon-DarkPhoton-Entanglement theory repository
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import root
import warnings

# Physical constants (in natural units ħ = c = 1, but we'll keep SI for clarity)
class PhysicalConstants:
    h = 6.62607015e-34      # Planck constant (J·s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    c = 299792458           # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    
    # Dark matter parameters (FDM mass scale ~10⁻²² eV)
    # Convert eV to kg: 1 eV/c² = 1.78266192e-36 kg
    eV_to_kg = 1.78266192e-36
    m_fdm_default = 1e-22 * eV_to_kg  # 10⁻²² eV/c² in kg
    
    # Dark photon mixing parameter (typical bounds: ε < 10⁻¹⁰ - 10⁻⁵)
    epsilon_default = 1e-8

class PhotonDarkPhotonSystem:
    """
    Coupled photon-dark-photon system with quantum entanglement
    Solves the von Neumann equation for density matrix evolution
    """
    
    def __init__(self, image_data, pixel_scale_arcsec=0.1, 
                 dark_photon_mass=None, mixing_parameter=None,
                 relative_velocity=None, redshift=0):
        """
        Initialize the coupled system
        
        Parameters:
        - image_data: 2D numpy array (photon field density |ψ_t|²)
        - pixel_scale_arcsec: angular resolution per pixel (arcseconds)
        - dark_photon_mass: dark photon mass in kg (default: FDM scale)
        - mixing_parameter: ε coupling strength (default: 1e-8)
        - relative_velocity: relative velocity between sectors (m/s)
        - redshift: cosmological redshift z
        """
        self.rho_photon = image_data  # Photon field density
        self.pixel_scale = pixel_scale_arcsec * (np.pi / (180 * 3600))  # Convert to radians
        
        # Physical parameters
        self.const = PhysicalConstants()
        self.m_d = dark_photon_mass or self.const.m_fdm_default
        self.epsilon = mixing_parameter or self.const.epsilon_default
        self.v_rel = relative_velocity or 1e5  # Default 100 km/s (typical galactic)
        self.z = redshift
        
        # Derived parameters
        self.de_broglie_wavelength = self.const.h / (self.m_d * self.v_rel)
        
        # Initialize dark photon field (coherent state approximation)
        self.psi_dark = self._initialize_dark_photon_field()
        
        # Density matrix (for entanglement quantification)
        self.rho_matrix = self._initialize_density_matrix()
        
    def _initialize_dark_photon_field(self):
        """
        Initialize dark photon wavefunction ψ_d
        Based on coherent state with de Broglie wavelength
        """
        ny, nx = self.rho_photon.shape
        
        # Create spatial coordinates
        x = np.arange(nx) * self.pixel_scale
        y = np.arange(ny) * self.pixel_scale
        X, Y = np.meshgrid(x, y)
        
        # De Broglie wavevector magnitude
        k_db = 2 * np.pi / self.de_broglie_wavelength
        
        # Random direction for dark photon propagation
        angle = np.random.uniform(0, 2*np.pi)
        kx = k_db * np.cos(angle)
        ky = k_db * np.sin(angle)
        
        # Plane wave with random phase
        phase = np.random.uniform(0, 2*np.pi)
        psi_d = np.exp(1j * (kx * X + ky * Y + phase))
        
        # Normalize to match photon field amplitude
        psi_d *= np.sqrt(np.mean(self.rho_photon))
        
        # Apply smoothing (dark matter is diffuse)
        psi_d = gaussian_filter(psi_d, sigma=5)
        
        return psi_d
    
    def _initialize_density_matrix(self):
        """
        Initialize 2x2 density matrix for photon-dark-photon system
        ρ = |ψ><ψ| in the {|γ>, |γ_d>} basis
        """
        # Coherent state: |ψ> = α|γ> + β|γ_d>
        # For weak mixing, β is small (∝ ε)
        alpha = np.sqrt(1 - self.epsilon**2)
        beta = self.epsilon
        
        # Density matrix ρ = |ψ><ψ|
        rho = np.array([
            [alpha**2, alpha*beta],
            [alpha*beta, beta**2]
        ])
        
        return rho
    
    def compute_interference(self):
        """
        Compute interference term from two-field superposition
        ρ_interference = 2ℜ(ψ_t* ψ_d e^{iΔϕ})
        
        Returns:
        - interference_pattern: 2D array of interference density
        - fringe_spacing: predicted fringe spacing in pixels
        """
        # ψ_t from photon field (square root of density)
        psi_photon = np.sqrt(self.rho_photon + 1e-10)  # Avoid division by zero
        
        # ψ_d from dark photon field
        psi_dark = self.psi_dark
        
        # Relative phase difference (Δϕ)
        # Could include cosmological expansion effects
        delta_phi = self._compute_relative_phase()
        
        # Interference term: 2ℜ(ψ_t* ψ_d e^{iΔϕ})
        interference = 2 * np.real(np.conj(psi_photon) * psi_dark * np.exp(1j * delta_phi))
        
        # Calculate expected fringe spacing in pixels
        fringe_spacing_px = self.de_broglie_wavelength / self.pixel_scale
        
        return interference, fringe_spacing_px
    
    def _compute_relative_phase(self):
        """
        Compute relative phase Δϕ including cosmological effects
        From von Neumann equation: dρ/dt = -i[H, ρ] + expansion terms
        """
        ny, nx = self.rho_photon.shape
        x = np.arange(nx) * self.pixel_scale
        y = np.arange(ny) * self.pixel_scale
        X, Y = np.meshgrid(x, y)
        
        # Baseline phase from momentum difference
        k_photon = 2 * np.pi / (500e-9)  # Typical optical wavelength 500 nm
        k_dark = 2 * np.pi / self.de_broglie_wavelength
        
        delta_k = k_photon - k_dark
        phase_momentum = delta_k * X  # Simple 1D gradient
        
        # Cosmological expansion effect (scales as (1+z) due to redshift)
        expansion_factor = 1 / (1 + self.z)
        phase_cosmology = expansion_factor * np.sqrt(X**2 + Y**2) / 1e6  # Simplified
        
        # Mixing-induced phase
        phase_mixing = self.epsilon * np.sin(2 * np.pi * X / self.de_broglie_wavelength)
        
        return phase_momentum + phase_cosmology + phase_mixing
    
    def evolve_von_neumann(self, dt=1e-9, steps=10):
        """
        Evolve density matrix using von Neumann equation:
        dρ/dt = -i[H, ρ] + Γ(ρ_thermal - ρ)  (including decoherence)
        
        Parameters:
        - dt: time step (seconds)
        - steps: number of evolution steps
        
        Returns:
        - evolved_density_matrix: updated ρ
        """
        # Hamiltonian in {|γ>, |γ_d>} basis
        # H = [[E_γ, ε], [ε, E_d]] with energy difference ΔE
        E_gamma = self.const.hbar * 3e15  # Optical photon energy ~3 eV
        E_dark = self.m_d * self.const.c**2  # Dark photon rest energy
        
        # Coupling term ε from Lagrangian
        H = np.array([
            [E_gamma, self.epsilon],
            [self.epsilon, E_dark]
        ])
        
        # Decoherence rate (due to environmental interactions)
        decoherence_rate = 1e6  # 1 MHz (placeholder)
        rho_thermal = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
        
        for _ in range(steps):
            # Commutator -i[H, ρ]
            commutator = -1j * (H @ self.rho_matrix - self.rho_matrix @ H)
            
            # Decoherence term
            decoherence = decoherence_rate * (rho_thermal - self.rho_matrix)
            
            # Update density matrix
            self.rho_matrix = self.rho_matrix + dt * (commutator + decoherence)
            
            # Ensure positive semidefinite and trace=1
            self.rho_matrix = (self.rho_matrix + self.rho_matrix.conj().T) / 2
            eigenvalues = np.linalg.eigvalsh(self.rho_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)
            self.rho_matrix = self.rho_matrix / np.trace(self.rho_matrix)
        
        return self.rho_matrix
    
    def compute_entanglement_measure(self):
        """
        Compute von Neumann entropy as entanglement measure:
        S = -Tr(ρ log ρ)
        
        Returns:
        - entanglement_entropy: scalar entropy value
        - concurrence: alternative entanglement measure (0-1)
        """
        # Von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(self.rho_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Avoid log(0)
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Concurrence for 2-qubit system
        # C = max(0, λ1 - λ2 - λ3 - λ4) where λi are sqrt(eigenvalues of ρσ_y⊗σ_y ρ* σ_y⊗σ_y)
        concurrence = max(0, 2 * np.abs(self.rho_matrix[0, 1]) - 
                          np.sqrt(1 - self.rho_matrix[0, 0]**2))
        
        return entropy, concurrence
    
    def generate_entanglement_map(self):
        """
        Generate full entanglement visualization map combining:
        - Photon field density
        - Dark photon field
        - Interference pattern
        - Quantum correlation measure
        
        Returns:
        - entanglement_map: 2D array with entangled density
        """
        # Compute interference term
        interference, fringe_spacing = self.compute_interference()
        
        # Evolve density matrix (simplified: use global evolution)
        self.evolve_von_neumann()
        
        # Get entanglement measure
        entropy, concurrence = self.compute_entanglement_measure()
        
        # Combine components with physics-based weights
        # ρ_total = |ψ_t|² + |ψ_d|² + 2ℜ(ψ_t*ψ_d e^{iΔϕ})
        total_density = self.rho_photon + np.abs(self.psi_dark)**2 + interference
        
        # Apply entanglement entropy as global scaling factor
        total_density *= (1 + entropy / 2)  # Enhance based on entanglement
        
        # Normalize
        total_density = (total_density - total_density.min()) / (total_density.max() - total_density.min() + 1e-10)
        
        # Add metadata for visualization
        self.metadata = {
            'entropy': entropy,
            'concurrence': concurrence,
            'fringe_spacing_px': fringe_spacing,
            'de_broglie_wavelength': self.de_broglie_wavelength,
            'mixing_parameter': self.epsilon,
            'dark_photon_mass': self.m_d
        }
        
        return total_density
    
    def get_predicted_fringe_spacing(self):
        """
        Return theoretical fringe spacing in pixels
        Based on de Broglie wavelength: λ = h/(m v)
        """
        return self.de_broglie_wavelength / self.pixel_scale

class CosmologicalEvolution:
    """
    Handle cosmological expansion effects on entanglement
    """
    
    def __init__(self, redshift, hubble_constant=70):
        self.z = redshift
        self.H0 = hubble_constant * 3.24078e-20  # km/s/Mpc to 1/s
        
    def expansion_factor(self):
        """Scale factor a(t) = 1/(1+z)"""
        return 1 / (1 + self.z)
    
    def hubble_parameter(self):
        """H(z) = H0 * sqrt(Ω_m (1+z)³ + Ω_Λ)"""
        Omega_m = 0.315  # Matter density parameter
        Omega_Lambda = 0.685  # Dark energy density parameter
        
        return self.H0 * np.sqrt(Omega_m * (1 + self.z)**3 + Omega_Lambda)
    
    def redshifting_wavelength(self, wavelength_rest):
        """Apply cosmological redshift to wavelength"""
        return wavelength_rest * (1 + self.z)
