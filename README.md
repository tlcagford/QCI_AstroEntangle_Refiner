# QCI AstroEntangle Refiner
Full desktop app by Tony E Ford (tlcagford@gmail.com)

Combines PSF correction + neural super-resolution + photon–dark-photon entanglement visualization.

## How to use
1. pip install -r requirements.txt
2. python QCI_AstroEntangle_Refiner.py
3. Click "Load FITS" → "Run Full Pipeline"

## Build standalone executable (recommended)
pyinstaller --onefile --windowed --name "QCI_AstroEntangle_Refiner" QCI_AstroEntangle_Refiner.py