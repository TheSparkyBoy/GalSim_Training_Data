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

    # --- Sensor & Background Noise Profile ---
    'full_well_capacity_e': 40000.0, # Physical max limit of the silicon well (electrons)
    'native_gain_e_adu': 9.77,       # Hardware conversion rate at Gain 0 (e-/ADU)
    'nina_gain': 100,                # NINA/ASIStudio Gain slider value (0 to 252+)
    'bias_pedestal': 500.0,          # Fixed electrical offset (ADU)
    'read_noise': 0.7,               # Amplifier read noise (electrons)
    'dark_current_rate': 0.002,      # Thermal noise rate (e-/pixel/second)
    'sky_background_rate': 25.0,     # Light pollution flux (e-/pixel/second)
    
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