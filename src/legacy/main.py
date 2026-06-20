import galsim
import os
import numpy as np
import csv
import warnings
import time
import random
import multiprocessing as mp
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u

warnings.filterwarnings('ignore')

# --- Main Thread ---
if __name__ == '__main__':
    base_dir = os.path.expanduser('~/GalSim_Training_Data')
    
    # ==========================================
    # --- DATASET CONFIGURATION (CHANGE THESE!) ---
    # ==========================================
    GLOBAL_SEED = 42
    mode = "opticalPSF_"
    total_images_to_generate = 12
    exposure_time = 0.3 # seconds
    focal_length_mm = 150 #416
    roll = 1 # degrees 
    pixel_size_um = 2.9
    image_size_x = 1024
    image_size_y = 1024
    fov = round(206.264806247096355 * (pixel_size_um / focal_length_mm) * image_size_x / 3600.0, 2) # degrees
    dataset_name = mode + "gaiadr3_" + "global_seed_" + str(GLOBAL_SEED)+ "_fov_" + str(fov) + "size_x_" + str(image_size_x) + "size_y_" + str(image_size_y) + "_pxlsz_" + str(pixel_size_um) + "um_" + str(focal_length_mm) + "mm_" + str(exposure_time) + \
    "s_" + "mag11_" + "roll" + str(roll) + "deg" + "_5seconds"
    # ==========================================

    # --- Seed the main thread ---
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    
    # --- Build the Isolated Dataset Folders ---
    dataset_dir = os.path.join(base_dir, 'training_data', dataset_name)
    fits_dir = os.path.join(dataset_dir, 'fits')
    png_dir = os.path.join(dataset_dir, 'png')
    csv_dir = os.path.join(dataset_dir, 'csv')
    
    os.makedirs(fits_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # --- 1. Load Local Cache ---
    print("Loading Master Star Catalog from local solid-state drive...")
    cache_file = os.path.join(base_dir, "master_star_caches", "GAIADR3_master_star_cache_11.csv")
    
    if not os.path.exists(cache_file):
        print(f"ERROR: Cannot find {cache_file}. Run build_cache.py first!")
        exit()
        
    master_df = pd.read_csv(cache_file)
    print(f"--> Loaded {len(master_df)} stars into RAM instantly.\n")

    ### Well-Known Positions ###
    well_known_targets = [
        (83.82, -5.0),   # 1. Orion's Belt
        (101.28, -16.7), # 2. Sirius (Canis Major)
        (201.36, -11.1), # 3. Spica (Virgo)
        (279.23, 38.78), # 4. Vega (Lyra)
        (15.00, 60.00),  # 5. Cassiopeia region
        (180.00, 0.0),   # 6. Celestial Equator
        (45.00, 89.0),   # 7. Polaris (North Celestial Pole)
        (250.00, -60.0)  # 8. Southern Hemisphere deep sky
    ]

    # ==========================================
    # --- THE GLOBAL ODOMETER ---
    # ==========================================
    manifest_path = os.path.join(base_dir, 'training_data', 'dataset_manifest.csv')
    global_img_id = 1
    
    if os.path.exists(manifest_path):
        try:
            existing_manifest = pd.read_csv(manifest_path)
            if not existing_manifest.empty:
                max_id = int(existing_manifest['image_id'].max())
                global_img_id = max_id + 1
                print(f"--> Memory loaded: Resuming at Universal ID {global_img_id:07d}...")
        except Exception as e:
            print(f"Warning: Could not read manifest. ({e})")
    # ==========================================

    tasks = []
    images_queued = 0
    
    # Stage 1: Queue the well-known positions first
    for t_ra, t_dec in well_known_targets:
        if images_queued >= total_images_to_generate:
            break

        tasks.append((global_img_id, t_ra, t_dec, roll, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_df, fits_dir, png_dir, csv_dir))
        global_img_id += 1
        images_queued += 1

    # Stage 2: Queue random positions to fill the rest of the quota
    if images_queued < total_images_to_generate:
        num_random_needed = total_images_to_generate - images_queued
        print(f"Queued 8 well-known targets. Calculating {num_random_needed} additional random coordinates...")
        
        while images_queued < total_images_to_generate:
            t_ra, t_dec = get_random_sky_coord()
            tasks.append((global_img_id, t_ra, t_dec, roll, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_df, fits_dir, png_dir, csv_dir))
            global_img_id += 1
            images_queued += 1
        
    num_cores = mp.cpu_count()
    print(f"\nFiring up {num_cores} autonomous cores for dataset: '{dataset_name}'...")
    
    generation_start = time.time()
    manifest_data = []
    
    # --- Parallel Dispatch ---
    with mp.Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(generate_single_image, tasks):
            # Print the success log to the terminal
            print(f"Image {result['image_id']} | Roll: {result['roll']:6.2f} | Stars: {result['stars_drawn']:4d} | Med. SNR: {result['median_snr']} | Time: {result['time_s']}s")
            
            # Append the global metadata to our Manifest list
            manifest_data.append({
                'image_id': result['image_id'],
                'dataset_group': dataset_name,
                'ra': result['ra'],
                'dec': result['dec'],
                'camera_roll': result['roll'],
                'focal_length_mm': focal_length_mm,
                'exposure_time_s': exposure_time,
                'total_stars': result['stars_drawn'],
                'median_image_snr': result['median_snr']
            })
            
    # --- Save the Master Manifest ---
    if len(manifest_data) > 0:
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df = manifest_df.sort_values(by='image_id') # Keep it neatly ordered
        
        manifest_path = os.path.join(base_dir, 'training_data', 'dataset_manifest.csv')
        
        # If the manifest already exists, we append to it without writing headers again
        if os.path.exists(manifest_path):
            manifest_df.to_csv(manifest_path, mode='a', header=False, index=False)
        else:
            manifest_df.to_csv(manifest_path, index=False)
                
        print(f"\nUniversal Dataset complete! Total Time: {time.time() - generation_start:.2f} seconds.")
        print(f"Global metadata appended to: {manifest_path}")
    else:
        print(f"\nZero tasks were queued! (Check your 'total_images_to_generate' variable).")
        print("Exiting safely without modifying the manifest.")
