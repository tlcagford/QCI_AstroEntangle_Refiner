# Create a virtual environment (recommended)
python -m venv qci_env
source qci_env/bin/activate  # On Windows: qci_env\Scripts\activate

# Install required packages
pip install numpy astropy matplotlib astroquery scipy
pip install customtkinter opencv-python torch torchvision

# If you want to use the full GUI, also install:
pip install customtkinter
