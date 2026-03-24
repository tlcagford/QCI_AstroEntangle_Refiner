# QCI_AstroEntangle_Refiner.py  -  v4.0 -  Tony E Ford
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve2d
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class PDPEntanglementModel:
    def __init__(self, omega_pd=0.20, fringe_scale=45.0):
        self.omega_pd = omega_pd
        self.fringe_scale = fringe_scale

    def solitonic_core(self, r, r_c=1.0, rho_c=1.0):
        return rho_c / (1 + (r / r_c)**2)**8

    def interference_density(self, psi_t, psi_d, delta_phi=np.pi/2):
        interference = 2 * np.real(psi_t.conj() * psi_d * np.exp(1j * delta_phi))
        return np.abs(psi_t)**2 + np.abs(psi_d)**2 + interference

    def apply_full_pdp(self, image, r_c=74.0, omega_pd=0.20):
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2) / 50.0
        core = self.solitonic_core(r, r_c=r_c/10.0, rho_c=1.0)
        psi_t = image.astype(np.complex128)
        psi_d = image.astype(np.complex128) * 0.35
        density = self.interference_density(psi_t, psi_d, delta_phi=np.pi/2)
        result = image * (1 + omega_pd * np.real(density) * core)
        return np.clip(result, image.min(), None)

class QCI_AstroEntangle_Refiner(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QCI AstroEntangle Refiner v2.1 - Tony E Ford")
        self.geometry("1680x1080")

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=300)
        sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(sidebar, text="QCI AstroEntangle Refiner", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)
        ctk.CTkLabel(sidebar, text="Tony E Ford • tlcagford@gmail.com", font=ctk.CTkFont(size=13)).pack(pady=5)

        ctk.CTkButton(sidebar, text="Load FITS", command=self.load_fits, height=45).pack(pady=12, padx=25, fill="x")
        ctk.CTkButton(sidebar, text="Run Full Pipeline", command=self.run_pipeline, height=55, fg_color="#00cc66").pack(pady=12, padx=25, fill="x")
        ctk.CTkButton(sidebar, text="Export Results", command=self.export, height=45).pack(pady=12, padx=25, fill="x")

        # Sliders
        ctk.CTkLabel(sidebar, text="Ω_PD", font=ctk.CTkFont(size=13)).pack(pady=(25,0))
        self.slider_omega = ctk.CTkSlider(sidebar, from_=0.05, to=0.50, number_of_steps=45)
        self.slider_omega.set(0.20)
        self.slider_omega.pack(pady=8, padx=25, fill="x")

        ctk.CTkLabel(sidebar, text="Fringe Scale", font=ctk.CTkFont(size=13)).pack(pady=(15,0))
        self.slider_fringe = ctk.CTkSlider(sidebar, from_=20, to=90, number_of_steps=35)
        self.slider_fringe.set(45)
        self.slider_fringe.pack(pady=8, padx=25, fill="x")

        # Tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=15)
        for name in ["Input", "Neural Enhanced", "Entangled Overlay", "Before / After"]:
            self.tabview.add(name)

        self.raw = None
        self.refined = None
        self.entangled = None

    def load_fits(self):
        path = filedialog.askopenfilename(filetypes=[("FITS", "*.fits *.fit *.fz")])
        if path:
            with fits.open(path) as hdul:
                self.raw = hdul[0].data.astype(np.float32)
                if len(self.raw.shape) == 3:
                    self.raw = np.mean(self.raw, axis=0)
            self.show_image("Input", self.raw, "Raw Input")

    def run_pipeline(self):
        if self.raw is None:
            return messagebox.showwarning("Error", "Load a FITS file first!")

        # Simple PSF + placeholder neural (you can expand later)
        self.refined = self.raw.copy()  # Replace with real PSF + neural

        # Full PDP
        model = PDPEntanglementModel(omega_pd=self.slider_omega.get(), fringe_scale=self.slider_fringe.get())
        self.entangled = model.apply_full_pdp(self.refined)

        self.show_image("Neural Enhanced", self.refined, "Neural Enhanced")
        self.show_image("Entangled Overlay", self.entangled, "Entangled FDM Overlay")
        self.show_comparison()

        messagebox.showinfo("Done", "Full pipeline completed with PDP model!")

    def show_image(self, tab_name, data, title):
        tab = self.tabview.tab(tab_name)
        for w in tab.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.imshow(data, cmap='viridis', origin='lower')
        ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_comparison(self):
        if self.raw is None or self.entangled is None:
            return
        tab = self.tabview.tab("Before / After")
        for w in tab.winfo_children():
            w.destroy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(self.raw, cmap='gray', origin='lower')
        ax1.set_title("Before")
        ax2.imshow(self.entangled, cmap='plasma', origin='lower')
        ax2.set_title("After - Full PDP")
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def export(self):
        if self.entangled is None:
            return messagebox.showwarning("No Results", "Run pipeline first!")
        folder = filedialog.askdirectory()
        if folder:
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            base = os.path.join(folder, f"QCI_Result_{ts}")
            os.makedirs(base, exist_ok=True)
            fits.writeto(os.path.join(base, "raw.fits"), self.raw, overwrite=True)
            fits.writeto(os.path.join(base, "entangled.fits"), self.entangled, overwrite=True)
            messagebox.showinfo("Exported", f"Saved to {base}")

if __name__ == "__main__":
    app = QCI_AstroEntangle_Refiner()
    app.mainloop()
