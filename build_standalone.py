# build_standalone.py
"""
Build standalone executables for QCI AstroEntangle Refiner
Supports Windows (.exe) and macOS (.app)
"""

import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path

def clean_build_dirs():
    """Remove previous build artifacts"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    # Also remove .spec files
    for spec_file in Path('.').glob('*.spec'):
        spec_file.unlink()
        print(f"  Removed {spec_file}")

def install_pyinstaller():
    """Ensure PyInstaller is installed"""
    try:
        import PyInstaller
        print("✓ PyInstaller already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed")

def get_platform_config():
    """Get platform-specific PyInstaller configuration"""
    system = platform.system()
    
    if system == "Windows":
        return {
            "name": "QCI_AstroEntangle_Refiner.exe",
            "icon": "icon.ico",
            "windowed": True,
            "target_arch": None,
            "extra_args": []
        }
    elif system == "Darwin":  # macOS
        return {
            "name": "QCI_AstroEntangle_Refiner.app",
            "icon": "icon.icns",
            "windowed": True,
            "target_arch": ["x86_64", "arm64"] if platform.machine() == "arm64" else ["x86_64"],
            "extra_args": ["--osx-bundle-identifier", "com.qci.astroentangle"]
        }
    else:  # Linux
        return {
            "name": "QCI_AstroEntangle_Refiner",
            "icon": "icon.png",
            "windowed": False,
            "target_arch": None,
            "extra_args": []
        }

def create_icon_files():
    """Create icon files for each platform if they don't exist"""
    import base64
    
    # Simple base64 encoded icon (you can replace with your own)
    # This creates a simple placeholder icon - replace with your actual icon file
    
    # Windows .ico file (minimal)
    ico_data = b""  # Replace with actual icon data
    if not os.path.exists("icon.ico"):
        print("  Note: icon.ico not found, building without custom icon")
    
    # macOS .icns file
    if not os.path.exists("icon.icns"):
        print("  Note: icon.icns not found, building without custom icon")
    
    # Linux .png file
    if not os.path.exists("icon.png"):
        print("  Note: icon.png not found, building without custom icon")

def build_executable():
    """Build the standalone executable using PyInstaller"""
    
    print("\n" + "="*60)
    print("QCI AstroEntangle Refiner - Standalone Builder")
    print("="*60)
    
    system = platform.system()
    print(f"\nPlatform: {system}")
    print(f"Python: {sys.version}")
    
    # Clean previous builds
    print("\n[1] Cleaning previous builds...")
    clean_build_dirs()
    
    # Install PyInstaller
    print("\n[2] Checking PyInstaller...")
    install_pyinstaller()
    
    # Get platform config
    config = get_platform_config()
    
    # Create icons if needed
    print("\n[3] Preparing icons...")
    create_icon_files()
    
    # Build the command
    print("\n[4] Building executable...")
    
    # Base command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "QCI_AstroEntangle_Refiner",
        "--add-data", "pdp_physics_working.py:.",
    ]
    
    # Add windowed mode if specified
    if config["windowed"]:
        cmd.append("--windowed")
    
    # Add icon if exists
    if config["icon"] and os.path.exists(config["icon"]):
        cmd.extend(["--icon", config["icon"]])
    
    # Add target architecture for macOS
    if config["target_arch"]:
        for arch in config["target_arch"]:
            cmd.extend(["--target-architecture", arch])
    
    # Add hidden imports for dependencies
    hidden_imports = [
        "customtkinter",
        "matplotlib",
        "numpy",
        "astropy",
        "scipy",
        "cv2",
        "torch",
        "torchvision",
        "PIL",
        "pkg_resources.py2_warn",
    ]
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # Add main file
    cmd.append("QCI_AstroEntangle_Refiner.py")
    
    # Print command for debugging
    print(f"\nCommand: {' '.join(cmd)}")
    
    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
        print("\n✓ Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Build failed: {e}")
        return False
    
    # Show output location
    print("\n[5] Build results:")
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            file_path = os.path.join("dist", file)
            size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file} ({size:.1f} MB)")
    
    return True

def create_launcher_scripts():
    """Create launcher scripts for easy execution"""
    system = platform.system()
    
    if system == "Windows":
        # Create .bat file
        bat_content = """@echo off
echo Starting QCI AstroEntangle Refiner...
cd /d "%~dp0"
start "" "dist\\QCI_AstroEntangle_Refiner.exe"
"""
        with open("run_app.bat", "w") as f:
            f.write(bat_content)
        print("  Created: run_app.bat")
        
    elif system == "Darwin":
        # Create .command file for macOS
        command_content = """#!/bin/bash
cd "$(dirname "$0")"
open "dist/QCI_AstroEntangle_Refiner.app"
"""
        with open("run_app.command", "w") as f:
            f.write(command_content)
        os.chmod("run_app.command", 0o755)
        print("  Created: run_app.command")
        
    else:  # Linux
        # Create .sh file
        sh_content = """#!/bin/bash
cd "$(dirname "$0")"
./dist/QCI_AstroEntangle_Refiner
"""
        with open("run_app.sh", "w") as f:
            f.write(sh_content)
        os.chmod("run_app.sh", 0o755)
        print("  Created: run_app.sh")

def create_zip_package():
    """Create a zip archive of the standalone application"""
    import zipfile
    from datetime import datetime
    
    system = platform.system()
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if system == "Windows":
        zip_name = f"QCI_AstroEntangle_Refiner_Win_{timestamp}.zip"
        app_path = "dist/QCI_AstroEntangle_Refiner.exe"
    elif system == "Darwin":
        zip_name = f"QCI_AstroEntangle_Refiner_Mac_{timestamp}.zip"
        app_path = "dist/QCI_AstroEntangle_Refiner.app"
    else:
        zip_name = f"QCI_AstroEntangle_Refiner_Linux_{timestamp}.zip"
        app_path = "dist/QCI_AstroEntangle_Refiner"
    
    if not os.path.exists(app_path):
        print(f"  No executable found at {app_path}")
        return
    
    print(f"\n[6] Creating zip package: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.isfile(app_path):
            zipf.write(app_path, os.path.basename(app_path))
        elif os.path.isdir(app_path):
            for root, dirs, files in os.walk(app_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "dist")
                    zipf.write(file_path, arcname)
    
    print(f"  Created: {zip_name} ({os.path.getsize(zip_name) / (1024*1024):.1f} MB)")

def print_instructions():
    """Print usage instructions"""
    system = platform.system()
    
    print("\n" + "="*60)
    print("✅ BUILD COMPLETE!")
    print("="*60)
    
    if system == "Windows":
        print("\nTo run the application:")
        print("  1. Navigate to the 'dist' folder")
        print("  2. Double-click 'QCI_AstroEntangle_Refiner.exe'")
        print("\nTo distribute:")
        print("  - Share the entire 'dist' folder")
        print("  - Or share the zip file created")
        
    elif system == "Darwin":
        print("\nTo run the application:")
        print("  1. Navigate to the 'dist' folder")
        print("  2. Double-click 'QCI_AstroEntangle_Refiner.app'")
        print("\nNote: On first run, you may need to:")
        print("  - Right-click and select 'Open'")
        print("  - Click 'Open' in the security dialog")
        
    else:  # Linux
        print("\nTo run the application:")
        print("  ./dist/QCI_AstroEntangle_Refiner")
        
    print("\nThe executable includes all dependencies and")
    print("does not require Python to be installed.")
    print("="*60)

if __name__ == "__main__":
    # Build the executable
    success = build_executable()
    
    if success:
        # Create launcher scripts
        create_launcher_scripts()
        
        # Create zip package
        create_zip_package()
        
        # Print instructions
        print_instructions()
    else:
        print("\n❌ Build failed. Please check the errors above.")
        sys.exit(1)
