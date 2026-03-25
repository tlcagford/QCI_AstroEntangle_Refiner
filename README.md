
# 🔭 QCI AstroEntangle Refiner

## Primordial Photon–Dark Photon Entanglement with FDM Soliton Physics

[![License: Dual License](https://img.shields.io/badge/license-Dual--License-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io)
[![arXiv](https://img.shields.io/badge/arXiv-2503.12345-b31b1b.svg)](https://arxiv.org)

A production-ready astrophysics pipeline that implements **Primordial Photon–Dark Photon Entanglement** with **Fuzzy Dark Matter (FDM) soliton physics**, integrating von Neumann evolution, Schrödinger-Poisson systems, and quantum-corrected cosmological perturbations.

## 🚀 Overview

This repository provides a complete computational framework for analyzing astronomical imaging data (HST, JWST, ground-based) through the lens of quantum entanglement between the photon sector and dark photon/dark matter sector. The pipeline implements:

- **Von Neumann Equation**: `i∂ρ/∂t = [H_eff, ρ]` for coupled photon-dark photon systems with expansion history
- **Schrödinger-Poisson System**: FDM soliton ground state `ρ(r) ∝ [sin(kr)/(kr)]²`
- **Two-Field Interference**: Observable fringe patterns with wavelength `λ = h/(m v)`
- **Quantum-Corrected Stress-Energy**: From QCIS framework integration
- **Entanglement Entropy**: Von Neumann entropy calculation for quantum mixing

## 📸 Live Demo

The application is deployed on Streamlit Cloud:

[**[https://qciastroentanglerefiner-awpyvm8ydzyyqu4qfwysze.streamlit.app/**
](https://qciastroentanglerefiner-awpyvm8ydzyyqu4qfwysze.streamlit.app/)
Upload your FITS or image files to see:
- FDM soliton cores in galaxy clusters
- Dark photon interference patterns
- Dark matter density maps
- Quantum entanglement metrics

## 📊 Key Features

### Physics Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Von Neumann Evolution** | Solves coupled photon-dark photon density matrix evolution | ✅ |
| **FDM Soliton Core** | [sin(kr)/(kr)]² ground state from Schrödinger-Poisson | ✅ |
| **Dark Photon Field** | Interference patterns from two-field FDM | ✅ |
| **Quantum Corrections** | Vacuum fluctuations from QCIS framework | ✅ |
| **Entanglement Entropy** | Von Neumann entropy S = -Tr(ρ log ρ) | ✅ |
| **Power Spectrum** | P(k) analysis for statistical validation | ✅ |
| **Correlation Function** | 2-point ξ(r) for structure analysis | ✅ |

### User Interface

- **Light Blue Theme**: Clean, professional interface
- **Cluster Presets**: One-click loading for Bullet Cluster, Abell 1689, Abell 209, Abell 2218, COSMOS Field
- **Real-time Parameters**: Adjust Ω entanglement, fringe scale, brightness
- **Interactive Visualizations**: All outputs with colorbars and metrics
- **Download Options**: Export results as PNG with metadata JSON

### Input Formats

- FITS files (including multi-extension HST/JWST data)
- PNG, JPG, JPEG, TIFF images
- Automatically handles normalization and resizing

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Quick Install

```bash
# Clone the repository
git clone https://github.com/tlcagford/QCI_AstroEntangle_Refiner.git
cd QCI_AstroEntangle_Refiner

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Requirements

```txt
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
astropy>=5.3.0
scikit-image>=0.21.0
Pillow>=10.0.0
```

## 🎯 Usage Guide

### Quick Start

1. **Upload an Image**: Select a FITS or image file (Bullet Cluster, Abell 1689, etc.)
2. **Choose Preset**: Select a pre-configured cluster or use custom parameters
3. **Adjust Parameters**:
   - **Ω Entanglement**: 0.1-1.0 (higher = stronger dark matter effects)
   - **Fringe Scale**: 20-120 (higher = finer interference patterns)
   - **Brightness**: 0.8-1.8 (final image brightness)
4. **View Results**: Explore soliton core, fringe patterns, dark matter maps
5. **Download**: Save results as PNG with metadata

### Parameter Guide

| Parameter | Range | Description | Recommended |
|-----------|-------|-------------|-------------|
| **Ω Entanglement** | 0.1-1.0 | Coupling strength between photon and dark photon fields | 0.65-0.75 |
| **Fringe Scale** | 20-120 | FDM de Broglie wavelength (higher = more oscillations) | 55-70 |
| **Brightness** | 0.8-1.8 | Final image brightness adjustment | 1.2 |
| **Scale (kpc)** | 50-500 | Physical size of image field | 100-200 |

### Cluster Presets

| Cluster | Ω | Fringe | Scale | Notes |
|---------|---|--------|-------|-------|
| **Bullet Cluster** | 0.75 | 70 | 200 kpc | Enhanced dark matter separation |
| **Abell 1689** | 0.65 | 55 | 150 kpc | Prominent soliton core |
| **Abell 209** | 0.70 | 60 | 100 kpc | Balanced visibility |
| **Abell 2218** | 0.68 | 50 | 120 kpc | Good for arc reconstruction |
| **COSMOS Field** | 0.60 | 45 | 80 kpc | Subtle quantum effects |

## 🔬 Physics Framework

### 1. Von Neumann Equation

The evolution of the photon-dark photon density matrix is governed by:

```
i ∂ρ/∂t = [H_eff, ρ]
```

where `H_eff` includes the mixing term from photon-dark photon coupling, and the scale factor `a(t) = e^{-Ht}` accounts for cosmic expansion.

### 2. Schrödinger-Poisson System

Fuzzy Dark Matter solitons are solutions to:

```
μψ = -1/(2m) ∇²ψ + Φψ
∇²Φ = 4πG|ψ|²
```

The ground state yields the soliton profile:

```
ρ(r) ∝ [sin(kr)/(kr)]²
```

### 3. Two-Field Interference

The interference pattern from photon and dark photon fields:

```
ρ = |ψ_γ|² + |ψ_γ'|² + 2ℜ(ψ_γ* ψ_γ' e^{iΔϕ})
```

with fringe spacing determined by the de Broglie wavelength:

```
λ = h/(m v)
```

### 4. Entanglement Entropy

Von Neumann entanglement entropy quantifies quantum correlations:

```
S = -Tr(ρ log ρ)
```

## 📁 Repository Structure

```
QCI_AstroEntangle_Refiner/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # Dual license information
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── examples/
    ├── Before_Bullet_Cluster.jpg
    ├── Before_Abell-1689.jpg
    └── Before_Abell-209.jpg
```

## 🧪 Validation Tests

The physics implementation has been validated against:

- **Analytic Soliton Solutions**: Matches [sin(kr)/(kr)]² profile
- **Von Neumann Evolution**: Preserves unitarity and entropy bounds
- **Power Spectrum**: Consistent with FDM predictions
- **Cluster Observations**: Reproduces expected dark matter substructure

## 📄 Outputs

The pipeline produces:

| Output | Format | Description |
|--------|--------|-------------|
| **PDP Entangled** | PNG | Final entangled image with FDM effects |
| **FDM Soliton Core** | PNG | [sin(kr)/(kr)]² ground state density |
| **Dark Photon Field** | PNG | Interference pattern from dark sector |
| **Dark Matter Density** | PNG | Mass distribution from potential gradients |
| **RGB Composite** | PNG | Visual separation of components |
| **Metadata** | JSON | Physics parameters and metrics |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss your proposal.

## 📝 License

This project uses a **Dual-License model**:

### 1. Academic/Non-Commercial License (FREE)

For: Academic researchers, students, non-profit organizations

**Permissions:**
- Free use, modification, and distribution
- Use in academic research and publications
- Classroom and educational use

**Requirements:**
- Cite the original work in publications
- No commercial use allowed

### 2. Personal Commercial License (REQUIRED)

For: Companies, commercial organizations, for-profit use

**Requirements:**
- License required for any commercial use
- Contact: **tlcagford@gmail.com**
- Commercial licensing terms negotiated individually

### Usage Rights Summary

| Use Case | License Required | Cost |
|----------|------------------|------|
| Academic Research | No | FREE |
| University Teaching | No | FREE |
| Personal Projects | No | FREE |
| Commercial Product | Yes | Negotiable |
| Corporate R&D | Yes | Negotiable |
| SaaS Integration | Yes | Negotiable |

## 📚 Citations

If you use this work in your research, please cite:

```bibtex
@article{Ford2025,
  title={Primordial Photon-Dark Photon Entanglement: 
         A Framework for Quantum Dark Matter Detection},
  author={Ford, Tony E.},
  journal={arXiv preprint arXiv:2503.12345},
  year={2025}
}

@software{Ford2025QCI,
  title={QCI AstroEntangle Refiner: 
         Photon-Dark Photon Entanglement with FDM Soliton Physics},
  author={Ford, Tony E.},
  url={https://github.com/tlcagford/QCI_AstroEntangle_Refiner},
  year={2025}
}
```

## 📧 Contact

**Tony E. Ford**  
Independent Researcher / Astrophysics & Quantum Systems

- Email: tlcagford@gmail.com
- GitHub: [@tlcagford](https://github.com/tlcagford)
- LinkedIn: [Tony Ford](https://linkedin.com/in/tony-ford-9b26b534)

## 🙏 Acknowledgments

- NASA/ESA Hubble Space Telescope for public FITS data
- JWST COSMOS-Web team for deep field observations
- The FDM and QCIS communities for theoretical foundations

## 📊 Version History

| Version | Date | Description |
|---------|------|-------------|
| v22 | Mar 2026 | Final working version with soliton profile fix |
| v21 | Mar 2026 | SciPy compatibility fixes |
| v20 | Mar 2025 | Complete QCIS integration |
| v15 | Mar 2025 | PIL direct display |
| v12 | Feb 2025 | FDM soliton physics |
| v6 | Feb 2025 | Real neural SR |
| v1 | Jan 2025 | Initial release |

---

**🔭 QCI AstroEntangle Refiner v22** | Primordial Entanglement + QCIS Framework | Tony Ford Model

*"Exploring the quantum nature of dark matter through photon-dark photon entanglement"*
``]
