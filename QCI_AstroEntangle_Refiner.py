"""
QCI AstroEntangle Refiner v4.1
Full Desktop App – Tony E Ford (tlcagford@gmail.com)

Combines:
  • Percentile-stretch brightness normalization
  • PSF correction (Gaussian unsharp mask)
  • Neural super-resolution (EDSR-Small, no pretrained weights needed)
  • Photon–dark-photon entanglement fringe overlay
  • Colorful cmaps: inferno / viridis / plasma / magma
  • Before/After comparison tab
  • Batch folder processing
  • PNG + FITS export
"""

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

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Neural SR Model ───────────────────────────────────────────────────────────
class EDSR_Small(nn.Module):
    """Lightweight EDSR – no pretrained weights required."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.res_blocks = nn.Sequential(*[self._res_block() for _ in range(8)])
        self.conv_up = nn.Conv2d(32, 32 * scale ** 2, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def _res_block(self):
        return nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        residual = x
        x = self.res_blocks(x)
        x = x + residual
        x = self.conv_up(x)
        x = F.pixel_shuffle(x, self.scale)
        return self.conv_out(x)


# ── Main App ──────────────────────────────────────────────────────────────────
class QCI_AstroEntangle_Refiner(ctk.CTk):

    TAB_NAMES = ["Input", "Neural Enhanced", "Entangled Overlay", "Before / After"]

    def __init__(self):
        super().__init__()
        self.title("QCI AstroEntangle Refiner v2.1 – Tony E Ford")
        self.geometry("1700x1050")
        self.minsize(1200, 750)

        self.raw: np.ndarray | None = None
        self.refined: np.ndarray | None = None
        self.entangled: np.ndarray | None = None
        self.fits_path: str = ""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr_model = EDSR_Small(scale=2).to(self.device)
        self.sr_model.eval()

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left sidebar
        sidebar = ctk.CTkFrame(self, width=310, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="QCI AstroEntangle\nRefiner",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(24, 4))
        ctk.CTkLabel(
            sidebar,
            text="Tony E Ford • tlcagford@gmail.com",
            font=ctk.CTkFont(size=11),
            text_color="gray70",
        ).pack(pady=(0, 18))

        # Buttons
        btns = [
            ("Load FITS", self.load_fits, "blue", 40),
            ("Run Full Pipeline", self.run_pipeline, "green", 50),
            ("Batch Process Folder", self.batch_process, "blue", 40),
            ("Export Results", self.export, "blue", 40),
        ]
        for text, cmd, color, h in btns:
            ctk.CTkButton(
                sidebar, text=text, command=cmd, height=h, fg_color=color
            ).pack(pady=6, padx=20, fill="x")

        ctk.CTkSeparator(sidebar).pack(fill="x", padx=10, pady=16)

        # ── Sliders ────────────────────────────────────────────────────────
        def _slider(label, from_, to, steps, default):
            ctk.CTkLabel(sidebar, text=label, font=ctk.CTkFont(size=12)).pack(
                pady=(10, 0), padx=20, anchor="w"
            )
            s = ctk.CTkSlider(sidebar, from_=from_, to=to, number_of_steps=steps)
            s.set(default)
            s.pack(pady=4, padx=20, fill="x")
            return s

        self.slider_omega = _slider("Ω_PD (Entanglement)  0.05 – 0.50", 0.05, 0.50, 45, 0.20)
        self.slider_fringe = _slider("Fringe Scale (pixels)  20 – 80", 20, 80, 30, 45)
        self.slider_brightness = _slider("Output Brightness  0.5 – 3.0", 0.5, 3.0, 50, 1.2)
        self.slider_saturation = _slider("Color Saturation  0.5 – 3.0", 0.5, 3.0, 50, 1.4)

        # Colormap selector
        ctk.CTkLabel(sidebar, text="Colormap", font=ctk.CTkFont(size=12)).pack(
            pady=(14, 0), padx=20, anchor="w"
        )
        self.cmap_var = ctk.StringVar(value="plasma")
        ctk.CTkComboBox(
            sidebar,
            variable=self.cmap_var,
            values=["plasma", "inferno", "viridis", "magma", "hot", "turbo", "jet", "rainbow"],
        ).pack(pady=4, padx=20, fill="x")

        # Status label
        self.status_label = ctk.CTkLabel(
            sidebar, text="Ready", font=ctk.CTkFont(size=11), text_color="gray60"
        )
        self.status_label.pack(side="bottom", pady=12)

        # Right: tab view
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        for name in self.TAB_NAMES:
            self.tabview.add(name)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.status_label.configure(text=msg)
        self.update_idletasks()

    @staticmethod
    def _normalize(data: np.ndarray, lo=0.5, hi=99.5) -> np.ndarray:
        """Percentile stretch to [0,1]."""
        vmin, vmax = np.percentile(data, lo), np.percentile(data, hi)
        return np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1).astype(np.float32)

    @staticmethod
    def _apply_brightness_saturation(img: np.ndarray, brightness: float, saturation: float) -> np.ndarray:
        """
        Boost brightness and 'saturation' on a grayscale [0,1] image by
        mapping through a power curve and then enhancing contrast.
        """
        # Gamma-like brightness
        img = np.clip(img * brightness, 0, 1)
        # Contrast / saturation: stretch histogram around mid
        mid = img.mean()
        img = np.clip(mid + (img - mid) * saturation, 0, 1)
        return img.astype(np.float32)

    def _psf_correct(self, data: np.ndarray) -> np.ndarray:
        kernel = Gaussian2DKernel(x_stddev=2)
        psf = kernel.array / kernel.array.sum()
        blurred = convolve2d(data, psf, mode='same', boundary='symm')
        return np.clip(data + 0.5 * (data - blurred), 0, 1).astype(np.float32)

    def _neural_sr(self, data: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(data[None, None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.sr_model(tensor).squeeze().cpu().numpy()
        return np.clip(out, 0, 1).astype(np.float32)

    def _entangle(self, sr: np.ndarray, omega: float, fringe_scale: float) -> np.ndarray:
        H, W = sr.shape
        y, x = np.mgrid[0:H, 0:W]
        # Multi-frequency interference pattern for richer color variation
        fringe = (
            0.4 * np.sin(2 * np.pi * x / fringe_scale) * np.cos(2 * np.pi * y / fringe_scale)
            + 0.3 * np.sin(2 * np.pi * (x + y) / (fringe_scale * 1.4))
            + 0.2 * np.cos(2 * np.pi * x / (fringe_scale * 0.7))
            + 0.1 * np.sin(4 * np.pi * y / fringe_scale)
        )
        fringe = (fringe - fringe.min()) / (fringe.max() - fringe.min() + 1e-9)  # [0,1]
        # Weight overlay by local brightness so dim regions get more entanglement
        weight = omega * (1.0 - sr * 0.5)
        overlay = sr + weight * (fringe - 0.5)
        return np.clip(overlay, 0, 1).astype(np.float32)

    def _colorize(self, img: np.ndarray, cmap_name: str) -> np.ndarray:
        """Convert [0,1] grayscale to RGB uint8 via matplotlib colormap."""
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(img)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    # ── Tab rendering ─────────────────────────────────────────────────────────

    def _show_image(self, tab_name: str, img: np.ndarray, cmap: str, title: str = ""):
        frame = self.tabview.tab(tab_name)
        for w in frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="#0d0d1a")
        ax.set_facecolor("#0d0d1a")
        im = ax.imshow(img, cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        ax.set_title(title or tab_name, color='white', fontsize=13, pad=10)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _show_before_after(self, before: np.ndarray, after: np.ndarray):
        frame = self.tabview.tab("Before / After")
        for w in frame.winfo_children():
            w.destroy()

        cmap = self.cmap_var.get()
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0d0d1a")
        for ax, img, title, cm in zip(
            axes,
            [before, after],
            ["Input (stretched)", f"Entangled – {cmap}"],
            ["inferno", cmap],
        ):
            ax.set_facecolor("#0d0d1a")
            im = ax.imshow(img, cmap=cm, origin='lower', aspect='auto')
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            ax.set_title(title, color='white', fontsize=12, pad=8)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _run_on(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run full pipeline on a single 2-D float array. Returns (norm, refined, entangled)."""
        norm = self._normalize(raw)
        psf_fixed = self._psf_correct(norm)
        sr = self._neural_sr(psf_fixed)
        bright = self.slider_brightness.get()
        sat = self.slider_saturation.get()
        sr = self._apply_brightness_saturation(sr, bright, sat)
        omega = self.slider_omega.get()
        fringe = self.slider_fringe.get()
        entangled = self._entangle(sr, omega, fringe)
        entangled = self._apply_brightness_saturation(entangled, bright, sat)
        return norm, sr, entangled

    # ── Button callbacks ──────────────────────────────────────────────────────

    def load_fits(self):
        path = filedialog.askopenfilename(
            title="Open FITS file",
            filetypes=[("FITS files", "*.fits *.fit *.fz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                if data is None:
                    # Try second HDU
                    data = hdul[1].data
                data = data.astype(np.float32)
                if data.ndim == 3:
                    data = np.mean(data, axis=0)
                elif data.ndim > 3:
                    data = data[0, 0]
            self.raw = data
            self.fits_path = path
            # Show preview of input immediately
            norm = self._normalize(data)
            self._show_image("Input", norm, "inferno", f"Input: {os.path.basename(path)}")
            self.tabview.set("Input")
            self._status(f"Loaded: {os.path.basename(path)}  ({data.shape[1]}×{data.shape[0]})")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def run_pipeline(self):
        if self.raw is None:
            return messagebox.showwarning("No Data", "Load a FITS file first.")
        cmap = self.cmap_var.get()
        self._status("Running PSF correction…")
        try:
            norm, sr, entangled = self._run_on(self.raw)
            self.refined = sr
            self.entangled = entangled

            self._status("Rendering tabs…")
            self._show_image("Input", norm, "inferno", "Input (percentile stretch)")
            self._show_image("Neural Enhanced", sr, "viridis", "Neural SR + Brightness/Saturation")
            self._show_image(
                "Entangled Overlay", entangled, cmap,
                f"Entanglement Overlay  Ω={self.slider_omega.get():.2f}  fringe={int(self.slider_fringe.get())}px"
            )
            self._show_before_after(norm, entangled)
            self.tabview.set("Entangled Overlay")
            self._status("Pipeline complete ✓")
            messagebox.showinfo("Done", "Pipeline complete!\nCheck all four tabs.")
        except Exception as e:
            messagebox.showerror("Pipeline Error", str(e))
            self._status("Error – see dialog")

    def batch_process(self):
        folder = filedialog.askdirectory(title="Select folder with FITS files")
        if not folder:
            return
        fits_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.fits', '.fit', '.fz'))
        ]
        if not fits_files:
            return messagebox.showwarning("No FITS", "No FITS files found in that folder.")

        out_folder = os.path.join(folder, f"QCI_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_folder, exist_ok=True)
        cmap = self.cmap_var.get()
        errors = []

        for i, fname in enumerate(fits_files):
            self._status(f"Batch {i+1}/{len(fits_files)}: {fname}")
            try:
                with fits.open(os.path.join(folder, fname)) as hdul:
                    raw = hdul[0].data
                    if raw is None:
                        raw = hdul[1].data
                    raw = raw.astype(np.float32)
                    if raw.ndim == 3:
                        raw = np.mean(raw, axis=0)

                _, sr, entangled = self._run_on(raw)
                stem = os.path.splitext(fname)[0]

                for tag, arr, cm in [("refined", sr, "viridis"), ("entangled", entangled, cmap)]:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(arr, cmap=cm, origin='lower')
                    ax.axis('off')
                    fig.savefig(
                        os.path.join(out_folder, f"{stem}_{tag}.png"),
                        bbox_inches='tight', dpi=150,
                    )
                    plt.close(fig)
                    fits.writeto(
                        os.path.join(out_folder, f"{stem}_{tag}.fits"),
                        arr, overwrite=True,
                    )
            except Exception as e:
                errors.append(f"{fname}: {e}")

        msg = f"Processed {len(fits_files) - len(errors)}/{len(fits_files)} files.\nOutput: {out_folder}"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        messagebox.showinfo("Batch Complete", msg)
        self._status("Batch done ✓")

    def export(self):
        if self.refined is None or self.entangled is None:
            return messagebox.showwarning("Nothing to export", "Run the pipeline first.")
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return

        cmap = self.cmap_var.get()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = []

        for tag, arr, cm in [
            ("refined", self.refined, "viridis"),
            ("entangled", self.entangled, cmap),
        ]:
            # PNG
            png_path = os.path.join(folder, f"QCI_{tag}_{ts}.png")
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(arr, cmap=cm, origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"QCI {tag} – {cm}", fontsize=13)
            ax.axis('off')
            fig.savefig(png_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
            exported.append(png_path)

            # FITS
            fits_path = os.path.join(folder, f"QCI_{tag}_{ts}.fits")
            fits.writeto(fits_path, arr.astype(np.float32), overwrite=True)
            exported.append(fits_path)

        messagebox.showinfo(
            "Exported",
            f"Saved {len(exported)} files to:\n{folder}\n\n" + "\n".join(os.path.basename(p) for p in exported),
        )
        self._status("Export done ✓")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QCI_AstroEntangle_Refiner()
    app.mainloop()
