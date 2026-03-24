# QCI AstroEntangle Refiner v4.0

**Quantum-Cosmology Integration**  
**Full Photon–Dark-Photon Entangled Fuzzy Dark Matter (FDM) Visualization Tool**

**Author:** Tony E Ford  
**Email:** tlcagford@gmail.com  

---

## Overview

QCI AstroEntangle Refiner is a desktop application that combines astronomical image refinement with a **complete photon–dark-photon entanglement model** for Fuzzy Dark Matter (FDM).

It allows researchers to:
- Load real HST/JWST/Chandra FITS files
- Apply PSF deconvolution
- Perform neural enhancement
- Overlay the full **Photon–Dark-Photon Entangled FDM model** (matching the Cosmic Entanglement Visualizer)
- Visualize soliton cores and interference fringes
- Export processed FITS and comparison images

**Version 4.0** features the complete implementation of the PDP formulas:
- Solitonic core density profile: ρ(r) = ρ_c / [1 + (r/r_c)²]⁸
- Full two-field interference term
- Tunable Ω_PD (default 0.20)
- Fringe scale control

---

## Features

- Clean dark-themed GUI with tabbed interface
- Real FITS file support (HST, JWST, Chandra composites)
- Richardson-Lucy PSF deconvolution
- Photon–Dark-Photon entanglement overlay with full mathematical model
- Interactive sliders for Ω_PD and fringe scale
- Before / After comparison view
- Export raw and processed FITS files + PNG comparisons
- Ready for batch processing extension

---

## Screenshots

*(Add your generated images here once uploaded)*

- Bullet Cluster Before / After
- ![Before   Bullet Cluster](https://github.com/user-attachments/assets/e9217133-460c-40af-97ab-4149599bbdb3)
- ![After Bullet Cluster](https://github.com/user-attachments/assets/11ca83e9-ad93-49ca-8c80-08959b22c8ba)

- Abell 1689 Before / After
- ![Before Abell-1689](https://github.com/user-attachments/assets/c61ed9b4-99f3-43e9-b701-79d9d2f6d85f)
-![After Abell-1689](https://github.com/user-attachments/assets/a5c5eca5-6b66-4bbc-bf97-e4afe0a16cac)


- Abell 209 Before / After
-![Before Abell-209](https://github.com/user-attachments/assets/6cede7a9-8749-4b9f-835d-8bc1386acae5)
-![After Abell-209](https://github.com/user-attachments/assets/6b85d4a2-e767-4158-8530-36a3ca5f9056)

---

## Installation & Running

### Option 1: Run from Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/tlcagford/QCI_AstroEntangle_Refiner.git
   cd QCI_AstroEntangle_Refiner
