#!/usr/bin/env python3
"""Quick build with onedir option."""
import subprocess, os, shutil
os.chdir('/workspaces/QCI_AstroEntangle_Refiner')
for p in ['build', 'dist']: 
    shutil.rmtree(p, ignore_errors=True)
print("Building with --onedir (faster than --onefile)...")
subprocess.run(['/usr/bin/python3', '-m', 'PyInstaller', '--onedir', '--windowed', '--name', 'QCI_AstroEntangle_Refiner', 'QCI_AstroEntangle_Refiner.py'])
if os.path.exists('dist/QCI_AstroEntangle_Refiner'):
    print("✓ Build complete! App in: dist/QCI_AstroEntangle_Refiner/")
