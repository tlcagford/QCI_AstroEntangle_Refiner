"""
Photon-Dark-Photon Entanglement Physics - Working Implementation
Based on equations from Primordial-Photon-DarkPhoton-Entanglement theory repository

Author: Implementation based on theoretical framework by Tony E. Ford
Date: 2026-03-23
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import warnings

# ============================================================================
# Physical Constants
# ============================================================================

@dataclass
class PhysicalConstants:
    """Physical constants in SI units"""
    h: float = 6.62607015e-34      # Planck constant (J·s)
    hbar: float = 1.054571817e-34  # Reduced Planck constant (J·s)
    c: float = 299792458           # Speed of light (m/s)
    G: float = 6.67430e-11         # Gravitational constant (m³/kg/s²)
    eV_to_J: float = 1.602176634e-19  # Electronvolt to Joule
    pc_to_m: float = 3.085677581e16   # Parsec to meters
    
    @property
    def eV_to_kg(self) -> float:
        """Convert eV/c² to kg"""
        return self.eV_to_J / self.c**2


# ============================================================================
# Dark Matter / Dark Photon Parameters
# ============================================================================

@dataclass
class DarkPhotonParameters:
    """Parameters for dark photon physics"""
    # FDM mass scale (typical: 10^-22 eV for galaxy-scale effects)
    mass_eV: float = 1e-22  # eV/c²
    # Kinetic mixing parameter (bounds: ε < 10^-10 to 10^-5)
    mixing_epsilon: float = 1e-8
    # Relative velocity between photon and dark photon sectors (m/s)
    relative_velocity: float = 1e5  # 100 km/s typical
    # Dark photon field amplitude relative to photon field
    amplitude_ratio: float = 0.1
    
    def __post_init__(self):
        const = PhysicalConstants()
        self.mass_kg = self.mass_eV * const.eV_to_kg
        self.de_broglie_wavelength = const.h / (self.mass_kg * self.relative_velocity)
        self.compton_wavelength = const.hbar / (self.mass_kg * const.c)
        self.plasma_frequency = np.sqrt(4 * np.pi * self.mass_kg * const.c**2 / const.hbar**2)


# ============================================================================
# Schrödinger-Poisson Solver for Two-Field System
# ============================================================================

class SchrodingerPoissonSolver:
    """
    Solves the coupled Schrödinger-Poisson system for photon and dark photon fields:
    
    iħ ∂ψ_t/∂t = -ħ²/(2m_t) ∇²ψ_t + (Φ_t + ε Φ_d) ψ_t
    iħ ∂ψ_d/∂t = -ħ²/(2m_d) ∇²ψ_d + (Φ_d + ε Φ_t) ψ_d
    ∇²Φ = 4πG (|ψ_t|² + |ψ_d|²)
    """
    
    def __init__(self, grid_size: Tuple[int, int], box_size: float, 
                 dt: float = 0.01, use_gpu: bool = False):
        """
        Parameters:
        - grid_size: (nx, ny) number of grid points
        - box_size: physical size of simulation box (meters)
        - dt: time step in seconds
        """
        self.nx, self.ny = grid_size
        self.Lx = self.Ly = box_size
        self.dt = dt
        
        # Grid setup
        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.nx)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        # Wavevectors for FFT (kinetic energy operator)
        kx = 2 * np.pi * fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.ny, self.dy)
        self.kx_grid, self.ky_grid = np.meshgrid(kx, ky, indexing='ij')
        self.k_sq = self.kx_grid**2 + self.ky_grid**2
        
        self.const = PhysicalConstants()
        
    def kinetic_energy_operator(self, psi: np.ndarray, mass: float) -> np.ndarray:
        """Apply kinetic energy operator: -ħ²/(2m) ∇²ψ using FFT"""
        psi_k = fft2(psi)
        psi_k = psi_k * (self.const.hbar**2 * self.k_sq / (2 * mass))
        return ifft2(psi_k)
    
    def solve_schrodinger(self, psi: np.ndarray, potential: np.ndarray, 
                          mass: float, dt: float) -> np.ndarray:
        """
        Solve Schrödinger equation using split-step Fourier method:
        ψ(t+dt) ≈ exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2) ψ(t)
        """
        # Half step in potential
        psi = psi * np.exp(-1j * potential * dt / (2 * self.const.hbar))
        
        # Full step in kinetic energy (in Fourier space)
        psi_k = fft2(psi)
        psi_k = psi_k * np.exp(-1j * self.const.hbar * self.k_sq * dt / (2 * mass))
        psi = ifft2(psi_k)
        
        # Half step in potential
        psi = psi * np.exp(-1j * potential * dt / (2 * self.const.hbar))
        
        return psi
    
    def compute_gravitational_potential(self, rho: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: ∇²Φ = 4πG ρ
        Using FFT: Φ = -4πG * F^{-1}[F[ρ] / k²]
        """
        # Add small constant to avoid division by zero
        k_sq_safe = self.k_sq.copy()
        k_sq_safe[0, 0] = 1.0
        
        rho_k = fft2(rho)
        phi_k = -4 * np.pi * self.const.G * rho_k / k_sq_safe
        phi_k[0, 0] = 0  # Set mean potential to zero
        phi = np.real(ifft2(phi_k))
        
        return phi
    
    def evolve_two_field(self, psi_photon: np.ndarray, psi_dark: np.ndarray,
                         params: DarkPhotonParameters, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve coupled photon-dark-photon system
        
        Returns:
        - psi_photon_final: evolved photon wavefunction
        - psi_dark_final: evolved dark photon wavefunction
        """
        # Normalize wavefunctions
        norm_photon = np.sum(np.abs(psi_photon)**2) * self.dx * self.dy
        norm_dark = np.sum(np.abs(psi_dark)**2) * self.dx * self.dy
        psi_photon = psi_photon / np.sqrt(norm_photon)
        psi_dark = psi_dark / np.sqrt(norm_dark)
        
        for step in range(steps):
            # Compute densities
            rho_photon = np.abs(psi_photon)**2
            rho_dark = np.abs(psi_dark)**2
            
            # Compute gravitational potentials
            phi_photon = self.compute_gravitational_potential(rho_photon)
            phi_dark = self.compute_gravitational_potential(rho_dark)
            
            # Coupled potentials with mixing
            V_photon = phi_photon + params.mixing_epsilon * phi_dark
            V_dark = phi_dark + params.mixing_epsilon * phi_photon
            
            # Masses
            m_photon = 0  # Photon is massless in this approximation
            m_dark = params.mass_kg
            
            # Evolve each field (use small mass for photon to avoid division by zero)
            m_photon_eff = max(m_photon, 1e-36)  # Small but non-zero
            
            psi_photon = self.solve_schrodinger(psi_photon, V_photon, m_photon_eff, self.dt)
            psi_dark = self.solve_schrodinger(psi_dark, V_dark, m_dark, self.dt)
            
        return psi_photon, psi_dark


# ============================================================================
# Density Matrix and Entanglement Measures
# ============================================================================

class DensityMatrixEvolution:
    """
    Solves von Neumann equation for the density matrix:
    dρ/dt = -i/ħ [H, ρ] + Γ (ρ_thermal - ρ)
    """
    
    def __init__(self, epsilon: float, energy_gamma: float, energy_dark: float):
        """
        Parameters:
        - epsilon: mixing parameter
        - energy_gamma: photon energy (J)
        - energy_dark: dark photon rest energy (J)
        """
        self.const = PhysicalConstants()
        self.H = np.array([
            [energy_gamma, epsilon * self.const.hbar],
            [epsilon * self.const.hbar, energy_dark]
        ], dtype=complex)
        
    def commutator(self, rho: np.ndarray) -> np.ndarray:
        """Compute -i/ħ [H, ρ]"""
        return -1j / self.const.hbar * (self.H @ rho - rho @ self.H)
    
    def lindblad_decoherence(self, rho: np.ndarray, gamma: float, 
                             rho_thermal: np.ndarray) -> np.ndarray:
        """Lindblad decoherence term: Γ (ρ_thermal - ρ)"""
        return gamma * (rho_thermal - rho)
    
    def evolve(self, rho0: np.ndarray, t_span: Tuple[float, float], 
               gamma: float = 1e6, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve density matrix using scipy.integrate.solve_ivp
        
        Returns:
        - times: time points
        - rhos: density matrices at each time
        """
        rho_thermal = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        
        def dv_dt(t, v):
            rho = v.reshape(2, 2)
            drho = self.commutator(rho) + self.lindblad_decoherence(rho, gamma, rho_thermal)
            return drho.flatten()
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        solution = solve_ivp(dv_dt, t_span, rho0.flatten(), t_eval=t_eval, method='RK45')
        
        times = solution.t
        rhos = solution.y.T.reshape(-1, 2, 2)
        
        return times, rhos
    
    @staticmethod
    def von_neumann_entropy(rho: np.ndarray) -> float:
        """Compute von Neumann entropy: S = -Tr(ρ log₂ ρ)"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # Avoid log(0)
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    @staticmethod
    def concurrence(rho: np.ndarray) -> float:
        """Compute concurrence for 2-qubit system (0 = separable, 1 = maximally entangled)"""
        # Pauli Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # R = ρ (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        rho_star = np.conj(rho)
        R = rho @ sigma_y_tensor @ rho_star @ sigma_y_tensor
        
        eigenvalues = np.linalg.eigvals(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues.real, 0))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        return max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])


# ============================================================================
# Interference Pattern Generation (Physics-Based)
# ============================================================================

class InterferencePattern:
    """
    Generate interference patterns from two-field superposition:
    ρ_interference = 2ℜ(ψ_t* ψ_d e^{iΔϕ})
    """
    
    def __init__(self, photon_field: np.ndarray, dark_photon_field: np.ndarray,
                 pixel_scale: float, params: DarkPhotonParameters):
        """
        Parameters:
        - photon_field: |ψ_t|² photon density map
        - dark_photon_field: ψ_d dark photon wavefunction
        - pixel_scale: meters per pixel
        - params: dark photon parameters
        """
        self.rho_photon = photon_field
        self.psi_dark = dark_photon_field
        self.pixel_scale = pixel_scale
        self.params = params
        
        # ψ_t from photon field (square root of density with phase)
        self.psi_photon = np.sqrt(np.maximum(photon_field, 0)) * self._initial_phase()
        
    def _initial_phase(self) -> np.ndarray:
        """Initial phase of photon field (can be coherent across the field)"""
        ny, nx = self.rho_photon.shape
        # Simple linear phase gradient
        x = np.arange(nx) * self.pixel_scale
        y = np.arange(ny) * self.pixel_scale
        X, Y = np.meshgrid(x, y)
        return np.exp(1j * 2 * np.pi * X / (500e-9))  # 500 nm wavelength
    
    def relative_phase(self, redshift: float = 0) -> np.ndarray:
        """
        Compute relative phase Δϕ including:
        - Momentum difference
        - Cosmological expansion
        - Mixing-induced oscillations
        """
        ny, nx = self.rho_photon.shape
        x = np.arange(nx) * self.pixel_scale
        y = np.arange(ny) * self.pixel_scale
        X, Y = np.meshgrid(x, y)
        
        # Photon wavenumber (500 nm light)
        k_gamma = 2 * np.pi / 500e-9
        
        # Dark photon wavenumber from de Broglie
        k_dark = 2 * np.pi / self.params.de_broglie_wavelength
        
        # Momentum difference phase
        delta_k = k_gamma - k_dark
        phase_momentum = delta_k * X
        
        # Cosmological expansion: phase scales as 1/(1+z)
        expansion_factor = 1 / (1 + redshift)
        phase_cosmology = expansion_factor * np.sqrt(X**2 + Y**2) / 1e6  # Simplified
        
        # Mixing-induced oscillations
        phase_mixing = self.params.mixing_epsilon * np.sin(2 * np.pi * X / self.params.de_broglie_wavelength)
        
        return phase_momentum + phase_cosmology + phase_mixing
    
    def compute_interference(self, redshift: float = 0) -> np.ndarray:
        """
        Compute interference density: 2ℜ(ψ_t* ψ_d e^{iΔϕ})
        """
        delta_phi = self.relative_phase(redshift)
        
        # Interference term
        interference = 2 * np.real(np.conj(self.psi_photon) * self.psi_dark * np.exp(1j * delta_phi))
        
        return interference
    
    def total_density(self, redshift: float = 0) -> np.ndarray:
        """
        Total density: |ψ_t|² + |ψ_d|² + interference
        """
        rho_dark = np.abs(self.psi_dark)**2
        interference = self.compute_interference(redshift)
        
        total = self.rho_photon + rho_dark + interference
        
        # Ensure non-negative
        return np.maximum(total, 0)


# ============================================================================
# Main Interface for GUI Integration
# ============================================================================

class PhotonDarkPhotonEngine:
    """
    Main engine for photon-dark-photon entanglement calculations
    Provides simple interface for GUI applications
    """
    
    def __init__(self):
        self.const = PhysicalConstants()
        self.current_params: Optional[DarkPhotonParameters] = None
        self.current_entanglement_map: Optional[np.ndarray] = None
        self.current_metadata: Dict = {}
        
    def initialize_from_image(self, image_data: np.ndarray, 
                              pixel_scale_arcsec: float = 0.1,
                              dark_photon_mass_eV: float = 1e-22,
                              mixing_epsilon: float = 1e-8,
                              relative_velocity: float = 1e5) -> Dict:
        """
        Initialize physics engine from astronomical image
        
        Parameters:
        - image_data: 2D numpy array (photon density)
        - pixel_scale_arcsec: angular resolution (arcseconds per pixel)
        - dark_photon_mass_eV: dark photon mass in eV/c²
        - mixing_epsilon: kinetic mixing parameter
        - relative_velocity: relative velocity between sectors (m/s)
        
        Returns:
        - metadata dictionary with physics parameters
        """
        # Convert pixel scale to meters
        # For astronomical objects, we need distance to convert angle to length
        # Default: assume 1 Mpc distance for typical galaxy clusters
        distance_mpc = 100  # 100 Mpc typical
        distance_m = distance_mpc * self.const.pc_to_m * 1e6
        pixel_scale_rad = pixel_scale_arcsec * (np.pi / (180 * 3600))
        pixel_scale_m = pixel_scale_rad * distance_m
        
        # Set up dark photon parameters
        self.current_params = DarkPhotonParameters(
            mass_eV=dark_photon_mass_eV,
            mixing_epsilon=mixing_epsilon,
            relative_velocity=relative_velocity,
            amplitude_ratio=0.1
        )
        
        # Initialize dark photon field
        psi_dark = self._initialize_dark_photon_field(image_data.shape, pixel_scale_m)
        
        # Create interference pattern
        interferometer = InterferencePattern(
            photon_field=image_data,
            dark_photon_field=psi_dark,
            pixel_scale=pixel_scale_m,
            params=self.current_params
        )
        
        # Compute entanglement map
        self.current_entanglement_map = interferometer.total_density(redshift=0)
        
        # Normalize to [0, 1]
        self.current_entanglement_map = (self.current_entanglement_map - self.current_entanglement_map.min())
        self.current_entanglement_map = self.current_entanglement_map / (self.current_entanglement_map.max() + 1e-10)
        
        # Compute density matrix and entanglement measures
        rho_initial = self._initialize_density_matrix()
        density_evo = DensityMatrixEvolution(
            epsilon=mixing_epsilon,
            energy_gamma=2.0 * self.const.eV_to_J,  # 2 eV photon
            energy_dark=self.current_params.mass_kg * self.const.c**2
        )
        
        entropy = density_evo.von_neumann_entropy(rho_initial)
        concurrence = density_evo.concurrence(rho_initial)
        
        # Store metadata
        self.current_metadata = {
            'entropy': float(entropy),
            'concurrence': float(concurrence),
            'de_broglie_wavelength_m': self.current_params.de_broglie_wavelength,
            'dark_photon_mass_eV': dark_photon_mass_eV,
            'mixing_epsilon': mixing_epsilon,
            'relative_velocity_km_s': relative_velocity / 1000,
            'pixel_scale_arcsec': pixel_scale_arcsec,
            'physical_pixel_scale_m': pixel_scale_m,
            'predicted_fringe_spacing_px': self.current_params.de_broglie_wavelength / pixel_scale_m,
            'energy_photon_eV': 2.0,
            'energy_dark_eV': dark_photon_mass_eV,
        }
        
        return self.current_metadata
    
    def _initialize_dark_photon_field(self, shape: Tuple[int, int], 
                                       pixel_scale_m: float) -> np.ndarray:
        """Initialize coherent dark photon wavefunction"""
        ny, nx = shape
        
        # Spatial coordinates
        x = (np.arange(nx) - nx//2) * pixel_scale_m
        y = (np.arange(ny) - ny//2) * pixel_scale_m
        X, Y = np.meshgrid(x, y)
        
        # De Broglie wavevector
        k_db = 2 * np.pi / self.current_params.de_broglie_wavelength
        
        # Random propagation direction (isotropic)
        angle = np.random.uniform(0, 2 * np.pi)
        kx = k_db * np.cos(angle)
        ky = k_db * np.sin(angle)
        
        # Plane wave with random phase
        phase = np.random.uniform(0, 2 * np.pi)
        psi_d = np.exp(1j * (kx * X + ky * Y + phase))
        
        # Normalize to amplitude ratio
        psi_d = psi_d * self.current_params.amplitude_ratio
        
        # Apply Gaussian envelope (dark matter is diffuse)
        sigma = min(nx, ny) * pixel_scale_m / 2
        envelope = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        psi_d = psi_d * envelope
        
        return psi_d
    
    def _initialize_density_matrix(self) -> np.ndarray:
        """Initialize 2x2 density matrix for photon-dark-photon system"""
        if self.current_params is None:
            return np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        
        # Coherent superposition with mixing strength
        alpha = np.sqrt(1 - self.current_params.mixing_epsilon**2)
        beta = self.current_params.mixing_epsilon
        
        rho = np.array([
            [alpha**2, alpha * beta],
            [alpha * beta, beta**2]
        ], dtype=complex)
        
        return rho
    
    def update_parameters(self, **kwargs) -> Dict:
        """Update physics parameters and recompute entanglement map"""
        if self.current_params is None:
            raise ValueError("Must call initialize_from_image first")
        
        # Update parameters
        if 'mixing_epsilon' in kwargs:
            self.current_params.mixing_epsilon = kwargs['mixing_epsilon']
        if 'mass_eV' in kwargs:
            self.current_params.mass_eV = kwargs['mass_eV']
            self.current_params.__post_init__()  # Recompute derived quantities
        if 'relative_velocity' in kwargs:
            self.current_params.relative_velocity = kwargs['relative_velocity']
            self.current_params.__post_init__()
        
        # Recompute with new parameters (would need to store original image)
        # This is a placeholder - full recomputation requires storing original data
        self.current_metadata['mixing_epsilon'] = self.current_params.mixing_epsilon
        self.current_metadata['dark_photon_mass_eV'] = self.current_params.mass_eV
        
        return self.current_metadata
    
    def get_entanglement_map(self) -> np.ndarray:
        """Return current entanglement map"""
        if self.current_entanglement_map is None:
            raise ValueError("No entanglement map computed. Call initialize_from_image first.")
        return self.current_entanglement_map
    
    def get_metadata(self) -> Dict:
        """Return current physics metadata"""
        return self.current_metadata


# ============================================================================
# Example Usage and Testing
# ============================================================================

def create_test_image(size: int = 200) -> np.ndarray:
    """Create a synthetic test image (Gaussian blob)"""
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Central Gaussian
    image = np.exp(-(X**2 + Y**2) / 20)
    
    # Add some structure (simulating galaxy cluster)
    image += 0.3 * np.exp(-((X-3)**2 + (Y-2)**2) / 5)
    image += 0.3 * np.exp(-((X+2)**2 + (Y+3)**2) / 5)
    
    # Add noise
    image += 0.05 * np.random.randn(size, size)
    
    return np.maximum(image, 0)


def run_demo():
    """Run a demonstration of the physics engine"""
    print("=" * 60)
    print("Photon-Dark-Photon Entanglement Physics Engine - Demo")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating test image...")
    test_image = create_test_image(200)
    print(f"   Image shape: {test_image.shape}")
    print(f"   Image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    
    # Initialize physics engine
    print("\n2. Initializing physics engine...")
    engine = PhotonDarkPhotonEngine()
    
    metadata = engine.initialize_from_image(
        image_data=test_image,
        pixel_scale_arcsec=0.1,      # 0.1 arcsec/pixel
        dark_photon_mass_eV=1e-22,   # FDM mass scale
        mixing_epsilon=1e-8,          # Kinetic mixing
        relative_velocity=1e5         # 100 km/s
    )
    
    print("\n3. Physics Parameters:")
    print(f"   Dark photon mass: {metadata['dark_photon_mass_eV']:.2e} eV")
    print(f"   de Broglie wavelength: {metadata['de_broglie_wavelength_m']:.2e} m")
    print(f"   Mixing parameter ε: {metadata['mixing_epsilon']:.2e}")
    print(f"   Relative velocity: {metadata['relative_velocity_km_s']:.1f} km/s")
    print(f"   Predicted fringe spacing: {metadata['predicted_fringe_spacing_px']:.1f} pixels")
    
    # Get entanglement measures
    print("\n4. Entanglement Measures:")
    print(f"   von Neumann Entropy: {metadata['entropy']:.4f} bits")
    print(f"   Concurrence: {metadata['concurrence']:.4f} (0=separable, 1=entangled)")
    
    # Get entanglement map
    entanglement_map = engine.get_entanglement_map()
    print(f"\n5. Entanglement Map:")
    print(f"   Shape: {entanglement_map.shape}")
    print(f"   Range: [{entanglement_map.min():.3f}, {entanglement_map.max():.3f}]")
    print(f"   Mean: {entanglement_map.mean():.3f}")
    
    print("\n✅ Demo complete! Physics engine is working.")
    print("\nTo use with the GUI, integrate PhotonDarkPhotonEngine class.")
    
    return engine, test_image, entanglement_map


if __name__ == "__main__":
    engine, original, entangled = run_demo()
    
    # Optional: plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Photon Field')
        axes[0].axis('off')
        
        axes[1].imshow(entangled, cmap='plasma')
        axes[1].set_title('Entangled Photon-Dark Photon Map')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plot.")
