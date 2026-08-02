# GalSim_Training_Data
This repository contains a high-performance, multi-core astrophotography simulation pipeline. It uses European Space Agency (ESA) Gaia DR3 data to synthesize mathematically perfect starfields tailored to specific optical hardware (e.g., ASI294MM sensor, 200mm lens).

🛠️ Installation & Setup
This project requires a Conda environment (Miniforge/Miniconda) to properly handle the complex C++ astronomical dependencies required by GalSim.

Step 1: Create a Dedicated Environment
It is highly recommended to isolate these packages to prevent dependency conflicts.

# Install conda for linux x86
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

# Accept Terms of Service
This step can be skipped if it doesn't go through, otherwise try using AI if the conda environment can't be created successfuly.
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
# Create a new environment named 'galsim_env' running Python 3.10
```
conda create -n galsim_env -c conda-forge python=3.10 galsim astropy astroquery pandas matplotlib numpy -y
```
# Activate the environment
```
conda activate galsim_env
```
Step 2: Install Required Packages
Run the following command to install the entire pipeline in one go from the conda-forge channel. This ensures all C++ binaries are correctly compiled for the Raspberry Pi's ARM architecture.

# Building Star Catalog
Bash
```
 python src/utils/cache_manager.py
```
# Running V3 Star Generator
```
python run_pipeline.py
```
To change the camera specification or catalog number, open config.py and adjust there. Make sure to save the config.py before running run_pipeline.py again.

# Using the csv_comparer.py
This is a CSV comparer tool to check if all the stars in a given CSV file exist in a star catalog file.
Make sure to install PyQt5, this is meant to be run on any machine, however windows is recommended for now.
```
pip install PyQt5 pandas
python catalog_validator_gui.py
```

Technical Specs of Hardware Simulated:
Sensor: ZWO ASI585MM Pro - https://www.zwoastro.com/product/asi585mc-mm-pro/
![alt text](image-1.png)
Lens: ZWO FF65 APO - https://www.zwoastro.com/product/zwo-ff65-apo/
![alt text](image.png)

# For devs
Install pyinstaller.
```
pip install pyinstaller
```
Compile executable by:
```
pyinstaller --name "ERAU_Catalog_Validator" --onefile --windowed catalog_validator.py
```
Compile executable for fits extractor.
```
pyinstaller --name "Fits Extractor" --onefile --windowed fits_extractor_gui.py
```
Compile star RA and DEC extractor
```
pyinstaller --name "Build Massive Star Ledge" --onefile --windowed build_massive_star_ledger.py
```