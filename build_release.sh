#!/bin/bash
set -e

cd /workspaces/QCI_AstroEntangle_Refiner

echo "=========================================="
echo "Building QCI_AstroEntangle_Refiner"
echo "=========================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist __pycache__ *.pyc

# Check for logo files, skip if not found
HAS_ICON=0
HAS_DATA=0

if [ -f "logo.ico" ]; then
    echo "✓ Found logo.ico"
    HAS_ICON=1
else
    echo "⚠ logo.ico not found - building without icon"
fi

if [ -f "logo.png" ]; then
    echo "✓ Found logo.png"
    HAS_DATA=1
else
    echo "⚠ logo.png not found - building without data files"
fi

# Build PyInstaller command
CMD="/usr/bin/python3 -m PyInstaller --onefile --windowed --name QCI_AstroEntangle_Refiner"

if [ $HAS_ICON -eq 1 ]; then
    CMD="$CMD --icon=logo.ico"
fi

if [ $HAS_DATA -eq 1 ]; then
    CMD="$CMD --add-data logo.png:."
fi

CMD="$CMD QCI_AstroEntangle_Refiner.py"

echo ""
echo "Running: $CMD"
echo "=========================================="

if eval $CMD; then
    echo "=========================================="
    echo "✓ Build completed successfully!"
    echo "✓ Executable: dist/QCI_AstroEntangle_Refiner"
    echo "✓ File size: $(du -h dist/QCI_AstroEntangle_Refiner | cut -f1)"
    echo "=========================================="
else
    echo "✗ Build failed"
    exit 1
fi
