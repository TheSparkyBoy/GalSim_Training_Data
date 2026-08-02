# config.py

DATASET_CONFIG = {
    'global_seed': 55,           # Global random seed for reproducibility
    'mode': "opticalPSF_",
    'total_images': 10000,          
    'exposure_time': 0.5,        # seconds
    'focal_length_mm': 100,      # millimeters
    'roll': 0.0,                 # degrees
    'pixel_size_um': 2.9,        # physical size of a pixel in micrometers
    'image_size_x': 1024,        # horizontal pixels
    'image_size_y': 1024,        # vertical pixels
    
    # --- Anomaly Toggles ---
    'anom_lens_distortion': False,
    'anom_false_stars': False,
    'anom_drop_stars': False,
    'anom_pos_variation': False,
    'anom_mag_variation': False,
    'anom_motion_smear': False,
    'anom_dead_pixels': False,
    'anom_defocus': False,
    'anom_hot_pixels': False,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_12.csv",
    'additional comments': ""
}