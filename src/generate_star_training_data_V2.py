import galsim
import os
import numpy as np
import csv
import warnings
import time
import random
import multiprocessing as mp
import pandas as pd

# CRITICAL FOR MULTIPROCESSING: Forces matplotlib to run in the background 
# without trying to open GUI windows, preventing memory leaks and crashes.
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u

warnings.filterwarnings('ignore')

# --- 1. The Worker Function (Independent Telescope Node) ---
def generate_single_image(args):
    """Each core downloads its own data and generates an image from start to finish."""
    image_id, target_ra, target_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_table, fits_dir, png_dir, csv_dir = args
    
    # [ANTI-BAN MEASURE]: Stagger the network requests so Vizier doesn't block the Pi
    time.sleep(random.uniform(0.1, 3.0)) 
    
    process_start = time.time()
    
    # 2. Calculate the true optical pixel scale (arcseconds per pixel)
    pixel_scale = 206.264806247096355 * (pixel_size_um / focal_length_mm)
    fov_x = pixel_scale * image_size_x / 3600.0 # Convert arcseconds to degrees
    fov_y = pixel_scale * image_size_y / 3600.0 # Convert arcseconds to degrees
    
    fits_filename = os.path.join(fits_dir, f'{image_id:07d}.fits')
    png_filename = os.path.join(png_dir, f'{image_id:07d}.png')
    csv_filename = os.path.join(csv_dir, f'{image_id:07d}.csv')
    
    image = galsim.ImageF(image_size_x, image_size_y)
    
    # -pixel_scale on X-axis so RA increases to the LEFT
    affine = galsim.AffineTransform(pixel_scale, 0, 0, pixel_scale, origin=image.true_center)
    world_origin = galsim.CelestialCoord(target_ra * galsim.degrees, target_dec * galsim.degrees)
    
    wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
    image.wcs = wcs
    
    # --- C. Draw Stars ---
    universal_baseline = 1e6 # 1 million photons per second for a 0 mag star
    aperature_area_cm2 = np.pi * (6.5/2)**2 # cm^2 for a 65mm diameter lens
    quantum_efficiency = 0.91*0.9 # 91% for ASI585MM sensor, 90% for APO glass transmission
    F0 = universal_baseline * aperature_area_cm2 * exposure_time * quantum_efficiency # Approximately Aperature Area (33.18cm^2) * Exposure Time (30s) * Quantum Efficiency (10,000photons/s)
    label_data = []
    stars_drawn = 0

    # 1. Global Focus: The lens has one focus position for the entire image
    global_defocus = random.uniform(-0.04, 0.04)

    # 2. Find the optical center of your 3840x2160 sensor
    center_x = image_size_x / 2.0
    center_y = image_size_y / 2.0
    max_radius = np.sqrt(center_x**2 + center_y**2) # Distance from center to the extreme corner
    
    for i, row in master_table.iterrows():
        real_star_id = int(row['source_id'])
        ra_val = row['RA_ICRS'] * galsim.degrees
        dec_val = row['DE_ICRS'] * galsim.degrees
        mag = row['Gmag']
        
        if pd.isna(mag): continue
            
        flux = F0 * 10 ** ((-mag) / 2.5)
        # star = galsim.Gaussian(flux=flux, sigma=1.8) # Simple Gaussian PSF for testing
        
        world_pos = galsim.CelestialCoord(ra_val, dec_val)
        pixel_pos = wcs.toImage(world_pos)
        
        if image.bounds.includes(pixel_pos):
            # star.drawImage(image=image, center=pixel_pos, add_to_image=True, method='real_space')
            # label_data.append([real_star_id, round(pixel_pos.x, 2), round(pixel_pos.y, 2), round(flux, 2), mag])
            # stars_drawn += 1

            # --- Spatial Variance Math ---
            # 1. How far is this specific star from the center of the lens?
            dx = pixel_pos.x - center_x
            dy = pixel_pos.y - center_y
            r = np.sqrt(dx**2 + dy**2)
            
            # 2. Normalize the distance (0.0 is dead center, 1.0 is the extreme corner)
            r_norm = r / max_radius
            
            # 3. Scale the aberrations. We square r_norm so it degrades faster at the edges!
            # The FF65 APO is excellent, so max corner aberration is kept small (0.04)
            edge_coma = 0.04 * (r_norm ** 2)
            edge_astig = 0.03 * (r_norm ** 2)
            
            # 4. Calculate the angle so the "comet tail" points radially outward from the center
            angle = np.arctan2(dy, dx)
            
            # Force GalSim to draw much larger bounding boxes for bright stars
            # Default folding_threshold is 1e-3. We lower it to 1e-5.
            high_accuracy_params = galsim.GSParams(
                folding_threshold=1e-3,
                maximum_fft_size=32768 # Prevents memory errors when the box gets really big
            )
            # 5. Build the physically accurate PSF for this exact pixel location
            optical_psf = galsim.OpticalPSF(
                lam=500.0,              #nm Wavelength of light (green)  
                diam=0.065,             #meter 65mm aperature        
                defocus=global_defocus, # Focus is the same everywhere
                spher=0.01,             # Spherical aberration is inherent to the glass
                astig1=edge_astig * np.cos(2*angle),
                astig2=edge_astig * np.sin(2*angle),
                coma1=edge_coma * np.cos(angle),
                coma2=edge_coma * np.sin(angle),
                gsparams=high_accuracy_params
            )
            
            star = optical_psf.withFlux(flux)
            # star = galsim.Gaussian(flux=flux, sigma=0.85)
            star.drawImage(image=image, center=pixel_pos, add_to_image=True, method='phot')
                        
            label_data.append([
                round(pixel_pos.x, 2), 
                round(pixel_pos.y, 2), 
                round(mag, 3), 
                focal_length_mm, 
                exposure_time
            ])            
            stars_drawn += 1

    # --- D. Sensor Noise ---
    image += 10.0 # background level
    rng = galsim.BaseDeviate(image_id) 
    
    # 1. Physics of Light (Shot Noise based on background level)
    poisson_noise = galsim.PoissonNoise(rng)
    image.addNoise(poisson_noise)
    
    # 2. Camera Electronics (Read Noise of the ASI585MM)
    read_noise = galsim.GaussianNoise(rng, sigma=0.7)
    image.addNoise(read_noise)

    # 3. Analog-to-Digital Conversion
    # The ASI585 uses a 12-bit ADC. We quantize the continuous electron 
    # decimals into discrete integer ADU steps, capping at absolute white (4095).
    image.quantize()
    image.array[image.array > 4095] = 4095
    
    # --- E. Export Files ---
    image.write(fits_filename)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x_image', 'y_image', 'flux_mag', 'focal_length', 'exposure_time'])
        writer.writerows(label_data)
        
    fig = plt.figure(dpi=300, figsize=(16, 9), facecolor='black')
    img_array = image.array
    plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=10, vmax=np.percentile(img_array, 99.9)))
    plt.title(f'Image {image_id:04d} (RA:{target_ra:.2f}, DEC:{target_dec:.2f})', color='white')
    plt.axis('off')
    plt.savefig(png_filename, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    process_duration = time.time() - process_start
    return {
        'image_id': f'{image_id:07d}',
        'ra': target_ra,
        'dec': target_dec,
        'fov_x': round(fov_x, 2),
        'fov_y': round(fov_y, 2),
        'stars_drawn': stars_drawn,
        'time_s': round(process_duration, 2)
    }

def get_random_sky_coord():
    """Generates a random point on the sky with uniform spherical distribution."""
    ra = random.uniform(0.0, 360.0)
    
    # Dec requires a sine-distribution to avoid 'clustering' at the poles
    z = random.uniform(-1.0, 1.0)
    dec = np.degrees(np.arcsin(z))
    
    return round(ra, 4), round(dec, 4)

# --- Main Thread ---
if __name__ == '__main__':
    base_dir = os.path.expanduser('~/GalSim_Training_Data')
    
    # ==========================================
    # --- DATASET CONFIGURATION (CHANGE THESE!) ---
    # ==========================================
    dataset_name = "optical_gaiadr3_416mm_15s_mag10"  # <--- Change this name for different experiments!
    total_images_to_generate = 1000       
    exposure_time = 15 # seconds
    focal_length_mm = 416 #416
    pixel_size_um = 2.9 
    image_size_x = 3840   
    image_size_y = 2160
    # ==========================================
    
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
    cache_file = os.path.join(base_dir, "master_star_caches", "GAIADR3_master_star_cache_12.csv")
    
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
            
        tasks.append((global_img_id, t_ra, t_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_df, fits_dir, png_dir, csv_dir))
        global_img_id += 1
        images_queued += 1

    # Stage 2: Queue random positions to fill the rest of the quota
    if images_queued < total_images_to_generate:
        num_random_needed = total_images_to_generate - images_queued
        print(f"Queued 8 well-known targets. Calculating {num_random_needed} additional random coordinates...")
        
        while images_queued < total_images_to_generate:
            t_ra, t_dec = get_random_sky_coord()
            tasks.append((global_img_id, t_ra, t_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_df, fits_dir, png_dir, csv_dir))
            global_img_id += 1
            images_queued += 1
        
    num_cores = 10#mp.cpu_count()
    print(f"\nFiring up {num_cores} autonomous cores for dataset: '{dataset_name}'...")
    
    generation_start = time.time()
    manifest_data = []
    
    # --- Parallel Dispatch ---
    with mp.Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(generate_single_image, tasks):
            # Print the success log to the terminal
            print(f"Image {result['image_id']} | FOV: {result['fov_x']}x{result['fov_y']} deg | RA:{result['ra']:6.2f}, DEC:{result['dec']:6.2f} | Stars: {result['stars_drawn']:4d} | Time: {result['time_s']}s")
            
            # Append the global metadata to our Manifest list
            manifest_data.append({
                'image_id': result['image_id'],
                'dataset_group': dataset_name,
                'ra': result['ra'],
                'dec': result['dec'],
                'focal_length_mm': focal_length_mm,
                'exposure_time_s': exposure_time,
                'total_stars': result['stars_drawn']
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
