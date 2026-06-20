# config.py

DATASET_CONFIG = {
    'global_seed': 50,
    'mode': "opticalPSF_",
    'total_images': 10,
    'exposure_time': 1.0,        # seconds
    'focal_length_mm': 200,    
    'roll': 0.0,                 # degrees
    'pixel_size_um': 2.9,
    'image_size_x': 1024,
    'image_size_y': 1024,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_11.csv",
    'additional comments': "Extra background level added 250"
}