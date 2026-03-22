#!/usr/bin/env python3
"""Build QCI_AstroEntangle_Refiner executable."""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def main():
    os.chdir('/workspaces/QCI_AstroEntangle_Refiner')
    
    print("=" * 60)
    print("Building QCI_AstroEntangle_Refiner with PyInstaller")
    print("=" * 60)
    
    # Clean previous builds
    print("\n[1/4] Cleaning previous builds...")
    for path in ['build', 'dist', '__pycache__']:
        if Path(path).exists():
            shutil.rmtree(path)
            print(f"  ✓ Removed {path}/")
    
    # Check for icon and data files
    print("\n[2/4] Checking for optional files...")
    has_icon = Path('logo.ico').exists()
    has_data = Path('logo.png').exists()
    
    if has_icon:
        print("  ✓ logo.ico found")
    else:
        print("  ⚠ logo.ico not found - will build without icon")
    
    if has_data:
        print("  ✓ logo.png found")
    else:
        print("  ⚠ logo.png not found - will build without data files")
    
    # Build command
    print("\n[3/4] Preparing PyInstaller command...")
    cmd = [
        '/usr/bin/python3', '-m', 'PyInstaller',
        '--onefile',
        '--windowed',
        '--name', 'QCI_AstroEntangle_Refiner',
    ]
    
    if has_icon:
        cmd.extend(['--icon', 'logo.ico'])
    
    if has_data:
        cmd.extend(['--add-data', 'logo.png:.'])
    
    cmd.append('QCI_AstroEntangle_Refiner.py')
    
    print(f"  Command: {' '.join(cmd)}")
    
    # Run build
    print("\n[4/4] Running PyInstaller build...")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd='/workspaces/QCI_AstroEntangle_Refiner')
    
    print("-" * 60)
    
    if result.returncode == 0:
        exe_path = Path('dist/QCI_AstroEntangle_Refiner')
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("\n" + "=" * 60)
            print("✓ BUILD SUCCESSFUL!")
            print("=" * 60)
            print(f"Executable: dist/QCI_AstroEntangle_Refiner")
            print(f"Size: {size_mb:.1f} MB")
            print("\nTo run the application:")
            print("  ./dist/QCI_AstroEntangle_Refiner")
            print("=" * 60)
        else:
            print("⚠ Executable not found at expected location")
            return 1
    else:
        print("\n✗ Build failed!")
        return result.returncode
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
