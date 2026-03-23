@echo off
echo ========================================
echo Building QCI AstroEntangle Refiner
echo ========================================

echo.
echo [1] Checking Python...
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

echo.
echo [2] Installing/updating PyInstaller...
pip install --upgrade pyinstaller

echo.
echo [3] Installing dependencies...
pip install customtkinter matplotlib numpy astropy scipy opencv-python torch torchvision

echo.
echo [4] Building executable...
pyinstaller --onefile --windowed --name "QCI_AstroEntangle_Refiner" --add-data "pdp_physics_working.py;." --hidden-import customtkinter --hidden-import matplotlib --hidden-import numpy --hidden-import astropy --hidden-import scipy --hidden-import cv2 --hidden-import torch --hidden-import torchvision QCI_AstroEntangle_Refiner.py

if errorlevel 1 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Executable created at: dist\QCI_AstroEntangle_Refiner.exe
echo.
echo To run: double-click the .exe file
echo.

pause
