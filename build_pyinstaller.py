#!/usr/bin/env python3
"""Build executable with PyInstaller"""
import subprocess
import sys
import os

os.chdir("/workspaces/QCI_AstroEntangle_Refiner")

# Run PyInstaller
result = subprocess.run([
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "QCI_AstroEntangle_Refiner",
    "QCI_AstroEntangle_Refiner.py"
])

sys.exit(result.returncode)
