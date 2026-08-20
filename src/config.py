# config.py
import os

# =========================================================================
# 1. SELECT VALIDATION SET & ANOMALY PACKAGE
# =========================================================================
# Optical Set: 'Set_6mag_17.5deg' (Beginning), 'Set_7mag_11deg', 'Set_8mag_7deg' (Mid), 
#              'Set_10mag_3deg', 'Set_12mag_1.5deg' (End)
ACTIVE_SET = 'Set_6mag_17.5deg'

# Anomaly Package: 'PKG_0_Clean', 'PKG_1_False_Mild', 'PKG_2_False_Heavy',
#                  'PKG_3_Drop_Mild', 'PKG_4_Drop_Heavy', 'PKG_5_Pos_Mild', 'PKG_6_Pos_Heavy'
ACTIVE_PACKAGE = 'PKG_0_Clean'

# =========================================================================
# 2. OPTICAL REGIME DEFINITIONS
# =========================================================================
OPTICAL_PRESETS = {
    'Set_6mag_17.5deg':  {'focal_length_mm': 36.45,  'aperture_mm': 18.0, 'exposure_time': 0.2, 'target_mag_limit': 6.0},
    'Set_7mag_11deg':    {'focal_length_mm': 58.00,  'aperture_mm': 29.0, 'exposure_time': 0.3, 'target_mag_limit': 7.0},
    'Set_8mag_7deg':     {'focal_length_mm': 91.13,  'aperture_mm': 32.5, 'exposure_time': 0.5, 'target_mag_limit': 8.0},
    'Set_10mag_3deg':    {'focal_length_mm': 212.63, 'aperture_mm': 50.0, 'exposure_time': 0.5, 'target_mag_limit': 10.0},
    'Set_12mag_1.5deg':  {'focal_length_mm': 416.00, 'aperture_mm': 65.0, 'exposure_time': 0.5, 'target_mag_limit': 12.0}
}

# =========================================================================
# 3. ISOLATED ANOMALY PACKAGES (VALIDATION SPECIFIC)
# =========================================================================
PACKAGE_PRESETS = {
    'PKG_0_Clean':       {'anom_false': False, 'anom_drop': False, 'anom_pos': False},
    'PKG_1_False_Mild':  {'anom_false': True,  'false_star_min': 1,  'false_star_max': 5,   'anom_drop': False, 'anom_pos': False},
    'PKG_2_False_Heavy': {'anom_false': True,  'false_star_min': 10, 'false_star_max': 25,  'anom_drop': False, 'anom_pos': False},
    'PKG_3_Drop_Mild':   {'anom_false': False, 'anom_drop': True,  'star_drop_rate': 0.10, 'anom_pos': False},
    'PKG_4_Drop_Heavy':  {'anom_false': False, 'anom_drop': True,  'star_drop_rate': 0.25, 'anom_pos': False},
    'PKG_5_Pos_Mild':    {'anom_false': False, 'anom_drop': False, 'anom_pos': True,  'pos_jitter_sigma_px': 1.0},
    'PKG_6_Pos_Heavy':   {'anom_false': False, 'anom_drop': False, 'anom_pos': True,  'pos_jitter_sigma_px': 4.0},
}

DATASET_NAME = f"VAL_{ACTIVE_SET}_{ACTIVE_PACKAGE}"

# =========================================================================
# 4. MASTER DATASET CONFIGURATION
# =========================================================================
DATASET_CONFIG = {
    'dataset_name': DATASET_NAME,
    'cache_filename': 'GAIADR3_master_star_cache_12.csv',
    'mode': '',
    'global_seed': 53,               # Fixed seed guarantees identical pointings across packages
    'total_images': 1000,            # 1,000 validation images per package
    'pixel_size_um': 2.9,
    'image_size_x': 3840,
    'image_size_y': 2160,
    'roll': 0.0,
    
    # NINA Sensor & Electronics
    'nina_gain': 100,
    'native_gain_e_adu': 9.77,
    'bias_pedestal': 500.0,
    'read_noise': 0.7,
    'dark_current_rate': 0.002,
    'sky_background_rate': 25.0,
    'full_well_capacity_e': 40000.0,
    
    # Observatory Environment
    'obs_lat': 33.3062,
    'obs_lon': -111.8413,
    'obs_alt': 370.0,
    'obs_time_utc': '2026-08-15T08:17:00',
    'delta_T_celsius': 2.0,
    
    # Inactive Anomalies
    'anom_lens_distortion': False,
    'anom_motion_smear': False,
    'anom_defocus': False,
    'anom_dead_pixels': False,
    'anom_hot_pixels': False,
    'anom_mag_variation': False,
    
    # Dynamic Presets
    **OPTICAL_PRESETS[ACTIVE_SET],
    **PACKAGE_PRESETS[ACTIVE_PACKAGE],
    'anom_false_stars': PACKAGE_PRESETS[ACTIVE_PACKAGE]['anom_false'],
    'anom_drop_stars': PACKAGE_PRESETS[ACTIVE_PACKAGE]['anom_drop'],
    'anom_pos_variation': PACKAGE_PRESETS[ACTIVE_PACKAGE]['anom_pos']
}