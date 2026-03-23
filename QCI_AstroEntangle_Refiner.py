from pdp_physics_working import PhotonDarkPhotonEngine

# And in __init__:
self.physics_engine = PhotonDarkPhotonEngine()

# And apply_pdp_entanglement_overlay should be replaced with:
def apply_pdp_entanglement_overlay(self, image):
    metadata = self.physics_engine.initialize_from_image(
        image_data=image,
        pixel_scale_arcsec=self.pixel_scale_arcsec,
        dark_photon_mass_eV=self._fringe_to_mass(),  # Convert slider
        mixing_epsilon=self._coupling_to_epsilon(),  # Convert slider
        relative_velocity=1e5
    )
    return self.physics_engine.get_entanglement_map()

# In the __init__ method of your main class
self.physics_engine = PhotonDarkPhotonEngine()
from pdp_physics_working import PhotonDarkPhotonEngine

        # Convert UI sliders to physics parameters
        pixel_scale = self.get_pixel_scale()  # Implement this
        epsilon = 10 ** (-8 + 6 * (self.entanglement_coupling - 0.05) / 0.45)
    """Apply actual photon-dark-photon entanglement physics"""
    
    # Get pixel scale from FITS header (implement this method)
    pixel_scale = self.get_pixel_scale_from_header()
    if pixel_scale is None:
        pixel_scale = 0.1  # Default arcsec/pixel
    
    # Map GUI sliders to physics parameters
    # entanglement_coupling (0.05-0.50) -> mixing_epsilon (log scale)
    # Use log scale because epsilon is typically tiny
    epsilon = 10 ** (-8 + 6 * (self.entanglement_coupling - 0.05) / 0.45)
    epsilon = np.clip(epsilon, 1e-12, 1e-5)
    
    # fringe_scale (pixels) -> dark photon mass via de Broglie
    # λ = h/(mv) -> m = h/(λv)
    fringe_meters = self.fringe_scale * pixel_scale * (np.pi / (180 * 3600))
    relative_velocity = 1e5  # 100 km/s
    const = PhysicalConstants()
    dark_photon_mass_eV = (const.h / (fringe_meters * relative_velocity)) / const.eV_to_kg
    
    # Initialize physics engine with image
    metadata = self.physics_engine.initialize_from_image(
        image_data=image,
        pixel_scale_arcsec=pixel_scale,
        dark_photon_mass_eV=dark_photon_mass_eV,
        mixing_epsilon=epsilon,
        relative_velocity=relative_velocity
    )
    
    # Get entanglement map
    entanglement_map = self.physics_engine.get_entanglement_map()
    
    # Store metadata for display
    self.current_physics_metadata = metadata
    self.physics_engine = PhotonDarkPhotonEngine()self.physics_engine = PhotonDarkPhotonEngine()
     # Convert UI sliders to physics parameters
        self.physics_engine = PhotonDarkPhotonEngine()apply_pdp_entanglement_overlayentanglement_map = self.physics_engine.initialize_from_image(...)
pixel_scale = self.get_pixel_scale()  # Implement this
        epsilon = 10 ** (-8 + 6 * (self.entanglement_coupling - 0.05) / 0.45)
        
        # Initialize physics engine
        metadata = self.physics_engine.initialize_from_image(
            image_data=image,
            pixel_scale_arcsec=pixel_scale,
            dark_photon_mass_eV=1e-22,  # Can map from fringe_scale
            mixing_epsilon=epsilon,
            relative_velocity=1e5
        )
        
        # Store metadata for physics info tab
        self.current_physics_metadata = metadata
        
        # Get entanglement map
        entanglement_map = self.physics_engine.get_entanglement_map()
        
        # Apply colormap
        colored_overlay = self._apply_colormap(entanglement_map)
        
        return colored_overlay
    except Exception as e:
        print(f"Physics error: {e}, falling back to decorative overlay")
        # Fallback to original decorative overlay
        return self._decorative_overlay(image)
# Apply colormap
    colored_overlay = self.apply_entanglement_colormap(entanglement_map)
    
    # Update physics info display
    self.update_physics_display()
    
    return colored_overlay
