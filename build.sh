#!/bin/bash
cd /workspaces/QCI_AstroEntangle_Refiner
echo "Cleaning up previous builds..."
rm -rf build dist __pycache__
echo "Starting PyInstaller build..."
echo "Using: /usr/bin/python3 -m PyInstaller"
/usr/bin/python3 -m PyInstaller --onedir --windowed --name "QCI_AstroEntangle_Refiner" QCI_AstroEntangle_Refiner.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build completed successfully!"
    echo "✓ Executable location: dist/QCI_AstroEntangle_Refiner/"
    echo "✓ To run: ./dist/QCI_AstroEntangle_Refiner/QCI_AstroEntangle_Refiner"
else
    echo "✗ Build failed"
    exit 1
fi
