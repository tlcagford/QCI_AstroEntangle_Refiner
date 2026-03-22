#!/bin/bash
cd /workspaces/QCI_AstroEntangle_Refiner
echo "Starting PyInstaller build..."
echo "Working directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "---"
python3 -m PyInstaller --onefile --windowed --name "QCI_AstroEntangle_Refiner" QCI_AstroEntangle_Refiner.py
echo ""
echo "Build exit code: $?"
if [ -f "dist/QCI_AstroEntangle_Refiner" ]; then
  echo "✓ Executable created successfully"
  echo "Size: $(du -h dist/QCI_AstroEntangle_Refiner | cut -f1)"
else
  echo "✗ Executable not found"
fi
