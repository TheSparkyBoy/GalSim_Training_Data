# config.py

DATASET_CONFIG = {
    'global_seed': 50,           # Global random seed for reproducibility
    'mode': "opticalPSF_",
    'total_images': 10,          
    'exposure_time': 1.0,        # seconds
    'focal_length_mm': 200,      # millimeters
    'roll': 0.0,                 # degrees
    'pixel_size_um': 2.9,        # physical size of a pixel in micrometers
    'image_size_x': 1024,        # horizontal pixels
    'image_size_y': 1024,        # vertical pixels
    'anom_lens_distortion': True,
    'anom_false_stars': True,
    'anom_drop_stars': True,
    'anom_pos_variation': True,
    'anom_mag_variation': True,
    'anom_motion_smear': True,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_11.csv",
    'additional comments': "Testing anomolies on"
}