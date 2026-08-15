# config.py

DATASET_CONFIG = {
    'global_seed': 53,           
    'mode': "opticalPSF_",
    'total_images': 10,          
    
    # --- Telescope & Sensor Hardware ---
    'exposure_time': 0.5,        # seconds
    'focal_length_mm': 416.0,    # millimeters
    'aperture_mm': 65.0,         # millimeters
    'pixel_size_um': 2.9,        # micrometers
    'image_size_x': 3840,        # horizontal pixels
    'image_size_y': 2160,        # vertical pixels
    'roll': 0.0,                 # degrees
    
    # --- Environmental & Location Conditions ---
    'delta_T_celsius': 2.0,                  # Drop in ambient temperature
    'obs_lat': 33.3062,                      # Observatory Latitude (Degrees)
    'obs_lon': -111.8413,                    # Observatory Longitude (Degrees)
    'obs_alt': 370.0,                        # Observatory Altitude above sea level (Meters)
    'obs_time_utc': '2026-08-15T08:17:00',   # UTC Time of observation
    
    # --- Anomaly Toggles ---
    'anom_lens_distortion': False,
    'anom_false_stars': False,
    'anom_drop_stars': False,
    'anom_pos_variation': False,
    'anom_mag_variation': False,
    'anom_motion_smear': False,
    'anom_defocus': False,
    'anom_dead_pixels': False,
    'anom_hot_pixels': False,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_12.csv",
    'additional comments': ""
}