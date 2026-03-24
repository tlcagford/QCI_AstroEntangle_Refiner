@echo off
echo ========================================================
echo    QCI AstroEntangle Refiner v4.0 - Build Script
echo    Author: Tony E Ford (tlcagford@gmail.com)
echo ========================================================
echo.

:: Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    echo.
)

echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist *.spec del /q *.spec

echo.
echo Building standalone executable (v4.0)...
echo This may take a few minutes...

pyinstaller --onefile ^
            --windowed ^
            --name "QCI_AstroEntangle_Refiner_v4.0" ^
            --icon=logo.ico ^
            --add-data "logo.png;." ^
            --clean ^
            QCI_AstroEntangle_Refiner.py

echo.
echo ========================================================
echo Build completed!
echo.
if exist dist\QCI_AstroEntangle_Refiner_v4.0.exe (
    echo Executable created successfully:
    echo dist\QCI_AstroEntangle_Refiner_v4.0.exe
    echo.
    echo You can now distribute this .exe file.
) else (
    echo ERROR: Build failed. Check the output above for errors.
)

echo.
pause
