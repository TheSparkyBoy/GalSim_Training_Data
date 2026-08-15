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
    'delta_T_celsius': 20.0,                  # Drop in ambient temperature
    'obs_lat': 33.3062,                      # Observatory Latitude (Degrees)
    'obs_lon': -111.8413,                    # Observatory Longitude (Degrees)
    'obs_alt': 370.0,                        # Observatory Altitude above sea level (Meters)
    'obs_time_utc': '2026-08-15T08:17:00',   # UTC Time of observation
    
    # --- Anomaly Toggles ---
    'anom_lens_distortion': True,
    'anom_false_stars': True,
    'anom_drop_stars': True,
    'anom_pos_variation': True,
    'anom_mag_variation': True,
    'anom_motion_smear': True,
    'anom_defocus': True,
    'anom_dead_pixels': True,
    'anom_hot_pixels': True,
    
    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_12.csv",
    'additional comments': ""
}