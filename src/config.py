# config.py

DATASET_CONFIG = {
    'global_seed': 50,
    'mode': "opticalPSF_",
    'total_images': 250,
    'exposure_time': 1.0,        # seconds
    'focal_length_mm': 400.0,    
    'roll': 0.0,                 # degrees
    'pixel_size_um': 2.9,
    'image_size_x': 1024,
    'image_size_y': 1024,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_12.csv"
}