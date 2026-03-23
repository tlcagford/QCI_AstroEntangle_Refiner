@echo off
echo ========================================
echo Building QCI AstroEntangle Refiner for Windows
echo ========================================

REM Check Python
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.9+
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install pyinstaller customtkinter matplotlib numpy astropy scipy opencv-python torch torchvision

REM Run build
echo Building executable...
python build_standalone.py

echo.
echo Build complete! Check the 'dist' folder.
pause
