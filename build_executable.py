#!/usr/bin/env python3
"""Build executable with PyInstaller using system Python 3."""

import subprocess
import os
import sys
import shutil

def main():
    os.chdir('/workspaces/QCI_AstroEntangle_Refiner')
    
    # Clean up previous builds
    print("Cleaning up previous builds...")
    for path in ['build', 'dist', '__pycache__']:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"  Removed {path}/")
    
    print("\nStarting PyInstaller build...")
    print("Command: /usr/bin/python3 -m PyInstaller --onefile --windowed --name QCI_AstroEntangle_Refiner QCI_AstroEntangle_Refiner.py")
    print("-" * 80)
    
    result = subprocess.run([
        '/usr/bin/python3',
        '-m', 'PyInstaller',
        '--onefile',
        '--windowed',
        '--name', 'QCI_AstroEntangle_Refiner',
        'QCI_AstroEntangle_Refiner.py'
    ])
    
    print("-" * 80)
    
    if result.returncode == 0:
        print("\n✓ Build completed successfully!")
        
        # Check if executable was created
        exe_path = 'dist/QCI_AstroEntangle_Refiner'
        if os.path.exists(exe_path):
            size = os.path.getsize(exe_path) / (1024*1024)
            print(f"✓ Executable created: {exe_path}")
            print(f"  Size: {size:.1f} MB")
            print("\nYou can run it with: ./dist/QCI_AstroEntangle_Refiner")
        else:
            print("⚠ Warning: Executable not found at expected location")
    else:
        print(f"\n✗ Build failed with return code: {result.returncode}")
        sys.exit(result.returncode)

if __name__ == '__main__':
    main()
