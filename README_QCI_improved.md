# QCI AstroEntangle Refiner

**Full desktop application for astronomical FITS image processing**  
PSF correction · Neural super-resolution · Photon–dark-photon entanglement visualisation

**Author:** Tony E. Ford — [tlcagford@gmail.com](mailto:tlcagford@gmail.com)  
**Version:** 2.1 (patched) · **Python:** 3.9+

---

## What it does

QCI AstroEntangle Refiner runs a three-stage pipeline on any FITS image from HST, JWST, MAST, or any other observatory:

| Stage | What happens |
|-------|-------------|
| **1 · PSF correction** | Wiener deconvolution removes point-spread-function blurring from the telescope optics |
| **2 · Neural super-resolution** | Lightweight EDSR network (×2 upscale) sharpens fine structure |
| **3 · PDP entanglement overlay** | Photon–dark-photon fringe colormap applied; before/after comparison generated |

Each stage is displayed in its own tab. Results can be exported as PNG or FITS.

---

## Quick start

### 1 — Install dependencies

```bash
pip install customtkinter matplotlib numpy astropy scipy opencv-python torch torchvision
```

> **GPU acceleration (optional):** Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for faster neural SR on large images. The app falls back to CPU automatically.

### 2 — Run the app

```bash
python QCI_AstroEntangle_Refiner.py
```

### 3 — Use the pipeline

1. Click **Load FITS** — select any `.fits`, `.fit`, `.fz`, or `.fits.gz` file
2. Adjust the three sliders if desired (defaults work well for most images)
3. Click **Run Full Pipeline** — all four output tabs populate automatically
4. Click **Export Results** to save the PDP overlay as PNG or FITS

---

## Sliders

| Slider | Range | Default | What it controls |
|--------|-------|---------|-----------------|
| **Ω_PD (entanglement coupling)** | 0.05 – 0.50 | 0.20 | Amplitude of the PDP fringe overlay. Higher = stronger colour effect |
| **Fringe scale (pixels)** | 20 – 80 | 45 | Spatial wavelength of the fringe pattern in pixels |
| **PSF sigma (deconvolution)** | 0.5 – 4.0 | 1.5 | Width of the Wiener filter kernel. Match to your telescope's PSF FWHM |

---

## Build a standalone executable

To distribute the app without requiring a Python environment:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "QCI_AstroEntangle_Refiner" QCI_AstroEntangle_Refiner.py
```

The executable will be in the `dist/` folder.

> **macOS note:** You may need `--target-architecture x86_64` on Apple Silicon if PyTorch is not yet arm64-compatible in your environment.

---

## Batch processing

Click **Batch Process Folder** to process every FITS file in a directory automatically. Converted images are saved as PNGs in a `qci_output/` subfolder alongside the originals.

---

## Supported FITS formats

- Single-extension FITS (most common — HST, JWST Level-2 calibrated)
- Multi-extension FITS (MEF) — automatically finds the first image HDU
- Compressed FITS (`.fz`, `.fits.gz`)
- 2D and 3D data cubes (cubes are averaged along axis 0)
- NaN / Inf values are replaced with zero before processing

---

## Output tabs

| Tab | Contents |
|-----|---------|
| **Input** | Raw FITS data, normalised to [0, 1] |
| **PSF Corrected** | After Wiener deconvolution |
| **Neural Enhanced** | After EDSR ×2 super-resolution |
| **Entangled Overlay** | PDP fringe colormap (blue → magenta → amber) |
| **Before / After** | Side-by-side input vs final output |

---

## Requirements

```
customtkinter>=5.2.0
matplotlib>=3.7.0
numpy>=1.24.0
astropy>=5.3.0
scipy>=1.11.0
opencv-python>=4.8.0
torch>=2.0.0
```

Save as `requirements.txt` and install with `pip install -r requirements.txt`.

---

## Troubleshooting

**`ModuleNotFoundError` on startup**  
The app checks dependencies at launch and prints the exact `pip install` command needed.

**`No image data found in FITS file`**  
Your FITS may have all image data in a non-primary HDU (common with Spitzer, XMM-Newton). Try opening it with `astropy.io.fits.info(your_file.fits)` to see the HDU structure.

**App window is too large / too small**  
Resize the window freely — the layout is dynamic. Minimum size is 1100×700.

**Neural SR is very slow**  
Install CUDA PyTorch for GPU acceleration, or reduce image size before loading. The model runs in ~0.5s on GPU and ~3–8s on CPU for a typical 2048×2048 FITS.

**Exported PNG looks different from the tab preview**  
Matplotlib's colour rendering can vary slightly by platform. The exported file uses the raw NumPy array directly via OpenCV, so it is more faithful to the actual data values.

---

## Changelog

| Version | Date | Notes |
|---------|------|-------|
| 2.1 | 2026-03-22 | Bug fixes: pipeline stubs wired, FITS loader fixed, EDSR residual corrected, PSF normalisation fixed, PDP amplitude guard added |
| 2.0 | 2026-01-xx | Initial public release |

---

## Licence

© 2025–2026 Tony E. Ford.  
Released for research and educational use.  
Commercial use requires written permission — contact [tlcagford@gmail.com](mailto:tlcagford@gmail.com).

---

## Related projects

- [Primordial PDP Entanglement](https://sourceforge.net/projects/primordial-pdp-entanglement/) — core PDP physics framework
- [PDP Astronomical Image Framework](https://sourceforge.net/projects/astronomical-image-refiner/) — SourceForge image pipeline
- [OmniSim / PDPBioGen](https://sourceforge.net/projects/pdpbiogen/) — multi-scale biological integration framework
