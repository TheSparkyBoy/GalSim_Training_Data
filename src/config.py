# config.py

# --- ZWO ASI585MM Pro (Sony IMX585, mono) ---
_ADC_BITS = 12

DATASET_CONFIG = {
    'global_seed': 53,           
    'mode': "opticalPSF_",
    'total_images': 15,          
    
    # --- Telescope & Sensor Hardware ---
    'exposure_time': 0.5,        # seconds
    'focal_length_mm': 416.0,    # millimeters
    'aperture_mm': 65.0,         # millimeters
    'pixel_size_um': 2.9,        # micrometers
    'image_size_x': 3840,        # horizontal pixels
    'image_size_y': 2160,        # vertical pixels
    'roll': 0.0,                 # degrees
    
    # This is a deterministic target FWHM, including seeing/focus blur.  It is
    # deliberately wider than the diffraction-only PSF at this plate scale.
    'psf_fwhm_pixels': 2.0,

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

    # --- SENSOR MODEL: ZWO ASI585MM Pro @ ZWO gain 60 ---
    # Calibrate these four values from a flat-field/photon-transfer measurement
    # for the exact camera mode.  Do not derive one from another: physical full
    # well and the digitizer limit are distinct parts of the camera model.
    'sky_e_per_pix_s': 2.5,          # ~21.3 mag/arcsec^2 at this plate scale
    'adc_bits': _ADC_BITS,
    'gain_e_per_adu': 4.8956,
    'full_well_e': 20047.5,
    'read_noise_e': 3.0,             # ZWO spec: 3.8 e- at gain 0 -> 0.8 e- at max
    'bias_adu': 31.0,
    # Target median background after digitization.  This is converted to
    # electrons before Poisson sampling, preserving natural scattered grain.
    # Set to None to use the physical sky_e_per_pix_s estimate instead.
    'target_background_adu': 1500.0,

    # --- PHOTOMETRIC MODEL ---
    # This provisional zero point preserves the previous flux scale.  Replace
    # it after matching an unsaturated Gaia reference star in a real exposure.
    'zero_point_photons_cm2_s_mag0': 8.47e5,
    'aperture_diameter_mm': 65.0,
    'system_throughput': 0.91 * 0.9,

    # Specific catalog to pull from your cache directory
    'cache_filename': "GAIADR3_master_star_cache_7_to_10.csv",
    'additional comments': ""
}