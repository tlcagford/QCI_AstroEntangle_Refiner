#!/bin/bash
echo "========================================"
echo "Building QCI AstroEntangle Refiner for macOS"
echo "========================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found! Please install Python 3.9+"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install pyinstaller customtkinter matplotlib numpy astropy scipy opencv-python

# Install PyTorch (CPU version for compatibility)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run build
echo "Building executable..."
python3 build_standalone.py

echo ""
echo "Build complete! Check the 'dist' folder."
echo ""
echo "To run: open dist/QCI_AstroEntangle_Refiner.app"
