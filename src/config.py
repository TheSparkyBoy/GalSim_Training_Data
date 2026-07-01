# config.py

DATASET_CONFIG = {
    'global_seed': 53,           # Global random seed for reproducibility
    'mode': "opticalPSF_",
    'total_images': 64,          
    'exposure_time': 1.0,        # seconds
    'focal_length_mm': 150,      # millimeters
    'roll': 0.0,                 # degrees
    'pixel_size_um': 2.9,        # physical size of a pixel in micrometers
    'image_size_x': 1024,        # horizontal pixels
    'image_size_y': 1024,        # vertical pixels
    'anom_lens_distortion': False,
    'anom_false_stars': False,
    'anom_drop_stars': False,
    'anom_pos_variation': False,
    'anom_mag_variation': False,
    'anom_motion_smear': False,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_12.csv",
    'additional comments': ""
}