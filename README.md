# 🌌 QCI AstroEntangle Refiner v4.5

**Photon-Dark Photon Quantum Entanglement + FDM Two-Field Physics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qci-astroentangle-refiner.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2403.12345-red.svg)](https://arxiv.org/abs/2403.12345)

---

## 📖 Overview

**QCI AstroEntangle Refiner** is a powerful interactive web application that simulates **photon-dark photon quantum entanglement** and **Fuzzy Dark Matter (FDM) two-field physics** on astronomical images. Built with Streamlit, it provides a comprehensive framework for visualizing quantum interference patterns, gravitational lensing effects, and dark matter distributions in galaxy clusters.

### Key Scientific Framework

The app implements the complete theoretical framework from the **Cosmic Entanglement Visualizer**, including:

- **Klein-Gordon Equation**: Relativistic wave equation for scalar field dark matter
  \[
  \Box\phi + m^2\phi = 0
  \]

- **Schrödinger-Poisson System**: Non-relativistic limit for galactic-scale structure
  \[
  i\partial_t\psi = -\frac{\nabla^2\psi}{2m} + \Phi\psi,\quad \nabla^2\Phi = 4\pi G|\psi|^2
  \]

- **Two-Field FDM Interference**: Photon-dark photon duality
  \[
  \psi_{\text{total}} = \psi_{\text{light}} + \psi_{\text{dark}} e^{i\Delta\phi}
  \]

- **Fringe Spacing**: de Broglie wavelength of FDM particles
  \[
  \lambda = \frac{h}{m\Delta v}
  \]

- **Solitonic Core Profile**: Solution to the cusp-core problem
  \[
  \rho(r) = \frac{\rho_c}{[1 + (r/r_c)^2]^8}
  \]

---

## ✨ Features

### 🔬 Physics Engine
- **Full FDM Two-Field Implementation** with Klein-Gordon and Schrödinger-Poisson equations
- **Photon-Dark Photon Oscillation** probability with quantum interference
- **Gravitational Lensing Simulation** with adjustable strength
- **Dark Matter Convergence Maps** from lensing reconstruction
- **Solitonic Core Profiles** for FDM halos

### 🎨 Visualization
- **Before/After Comparison** views (Original → PDP Conversion → PDP + Lensing)
- **Color Overlays**: Red = PDP conversion probability, Blue = Dark matter distribution
- **Two-Field Interference Pattern** visualization
- **Interactive Metrics Dashboard** with real-time updates
- **Solitonic Core Profile** explorer

### 🔧 Advanced Corrections
- **PSF Correction**: Wiener deconvolution for telescope beam smearing
- **Neural Enhancement**: CLAHE/Retinex/Unsharp contrast optimization
- **Customizable FWHM** for PSF modeling
- **Multi-method enhancement** options

### 🤖 AI Data Download
- **MAST Database Integration** (HST/JWST)
- **NED Database Queries**
- **Quick Select** buttons for common clusters
- **Natural language query** support

### 📊 Physics Metrics
- **Entropy**: Information content of the image
- **Concurrence**: Quantum entanglement measure (Ω = 2ε)
- **Purity**: Quantum state purity (1 - C²)
- **Fringe Spacing**: λ = h/(mΔv) in kpc
- **Conversion-Mass Correlation**: How well conversion traces dark matter

### 📥 Export Options
- Download original image
- Download PDP conversion result
- Download PDP + lensing composite
- High-resolution PNG export

---

## 🚀 Quick Start

### Live Demo
Access the live app: [https://qci-astroentangle-refiner.streamlit.app](https://qci-astroentangle-refiner.streamlit.app)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/tlcagford/QCI_AstroEntangle_Refiner.git
cd QCI_AstroEntangle_Refiner

# Install dependencies
pip install -r requirements_v4.txt

# Run the app
streamlit run streamlit_app_v4.py
