# Add these imports at the top (replace the old file)
import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn
import torch.nn.functional as F
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import convolve2d
import cv2
import os
from datetime import datetime

# ==================== FULL PDP FORMULAS ====================

class PDPEntanglementModel:
    """Full Photon-Dark-Photon Entanglement Model from your visualizer"""
    def __init__(self, omega_pd=0.20, fringe_scale=45):
        self.omega_pd = omega_pd          # Entanglement observable Ω_PD
        self.fringe_scale = fringe_scale  # ~3.14 kpc scaled to pixels

    def combined_wavefunction(self, psi_t, psi_d, delta_phi=np.pi/2):
        """ψ = ψ_t + ψ_d * e^(i Δφ)"""
        return psi_t + psi_d * np.exp(1j * delta_phi)

    def interference_density(self, psi_t, psi_d, delta_phi=np.pi/2):
        """Full interference term: |ψ_t|² + |ψ_d|² + 2 Re(ψ_t* ψ_d e^(iΔφ))"""
        interference = 2 * np.real(psi_t.conj() * psi_d * np.exp(1j * delta_phi))
        return np.abs(psi_t)**2 + np.abs(psi_d)**2 + interference

    def solitonic_core(self, r, r_c=1.0, rho_c=1.9e7):
        """ρ(r) = ρ_c / [1 + (r/r_c)²]^8"""
        return rho_c / (1 + (r / r_c)**2)**8

    def apply_full_entanglement(self, image, r_c=74, omega_pd=0.20):
        """Apply full P-D-P model to image"""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2) / 50   # scaled radius

        # Solitonic core boost
        core = self.solitonic_core(r, r_c=r_c/10, rho_c=1.0)

        # Create two fields
        psi_t = image.astype(np.complex128)
        psi_d = image.astype(np.complex128) * 0.3   # dark photon amplitude

        # Interference
        density = self.interference_density(psi_t, psi_d, delta_phi=np.pi/2)

        # Final modulated image
        result = image * (1 + self.omega_pd * np.real(density) * core)
        return np.clip(result, 0, None)

# ==================== MAIN APP ====================

class QCI_AstroEntangle_Refiner(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QCI AstroEntangle Refiner v2.1 - Full PDP Model")
        self.geometry("1650x1050")

        # ... (keep your sidebar and sliders as before)

        self.pdp_model = PDPEntanglementModel(omega_pd=0.20, fringe_scale=45)

    def run_pipeline(self):
        if self.raw is None:
            return messagebox.showwarning("Error", "Load a FITS first")

        # 1. PSF + Neural (keep as before)
        # ...

        # 2. Full Photon-Dark-Photon Entanglement
        entangled = self.pdp_model.apply_full_entanglement(self.refined_data, 
                                                           r_c=74, 
                                                           omega_pd=self.slider_omega.get())

        self.entangled_data = entangled
        self.show_image(self.tab_entangled, entangled, "Full Photon–Dark-Photon Entanglement Overlay")

        self.show_comparison()
        messagebox.showinfo("Success", "Full PDP model applied with all formulas from the visualizer!")

# Rest of your GUI code remains the same...
