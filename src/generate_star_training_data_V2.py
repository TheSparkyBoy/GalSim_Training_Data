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
    image_id, target_ra, target_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, master_table, output_dir = args
    
    # [ANTI-BAN MEASURE]: Stagger the network requests so Vizier doesn't block the Pi
    time.sleep(random.uniform(0.1, 3.0)) 
    
    process_start = time.time()
    
    # 2. Calculate the true optical pixel scale (arcseconds per pixel)
    pixel_scale = 206.264806247096355 * (pixel_size_um / focal_length_mm)
    fov_x = pixel_scale * image_size_x / 3600.0 # Convert arcseconds to degrees
    fov_y = pixel_scale * image_size_y / 3600.0 # Convert arcseconds to degrees
    
    fits_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.fits')
    csv_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.csv')
    png_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.png')
    
    image = galsim.ImageF(image_size_x, image_size_y)
    
    # -pixel_scale on X-axis so RA increases to the LEFT
    affine = galsim.AffineTransform(pixel_scale, 0, 0, pixel_scale, origin=image.true_center)
    world_origin = galsim.CelestialCoord(target_ra * galsim.degrees, target_dec * galsim.degrees)
    
    wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
    image.wcs = wcs
    
    # --- C. Draw Stars ---
    universal_baseline = 1e6 # 1 million photons per second for a 0 mag star
    exposure_time = 15 # seconds
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
                folding_threshold=1.e-3,
                maximum_fft_size=32768 # Prevents memory errors when the box gets really big
            )
            # 5. Build the physically accurate PSF for this exact pixel location
            optical_psf = galsim.OpticalPSF(
                lam=500.0,                
                diam=0.065,               
                defocus=global_defocus, # Focus is the same everywhere
                spher=0.01,             # Spherical aberration is inherent to the glass
                astig1=edge_astig * np.cos(2*angle),
                astig2=edge_astig * np.sin(2*angle),
                coma1=edge_coma * np.cos(angle),
                coma2=edge_coma * np.sin(angle),
                gsparams=high_accuracy_params
            )
            
            star = optical_psf.withFlux(flux)
            star.drawImage(image=image, center=pixel_pos, add_to_image=True, method='phot')
            
            label_data.append([real_star_id, round(pixel_pos.x, 2), round(pixel_pos.y, 2), round(flux, 2), mag])
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
    
    # --- E. Export Files ---
    image.write(fits_filename)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['star_id', 'x_pixel', 'y_pixel', 'flux', 'magnitude'])
        writer.writerows(label_data)
        
    fig = plt.figure(dpi=300, figsize=(16, 9), facecolor='black')
    img_array = image.array
    plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=10, vmax=np.percentile(img_array, 99.9)))
    plt.title(f'Image {image_id:04d} (RA:{target_ra:.2f}, DEC:{target_dec:.2f})', color='white')
    plt.axis('off')
    plt.savefig(png_filename, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    process_duration = time.time() - process_start
    return f"Image {image_id:04d} | FOV: {fov_x:.2f} x {fov_y:.2f} deg | RA:{target_ra:6.2f}, DEC:{target_dec:6.2f} | Stars: {stars_drawn:4d} | Time: {process_duration:5.2f}s"

def get_random_sky_coord():
    """Generates a random point on the sky with uniform spherical distribution."""
    ra = random.uniform(0.0, 360.0)
    
    # Dec requires a sine-distribution to avoid 'clustering' at the poles
    z = random.uniform(-1.0, 1.0)
    dec = np.degrees(np.arcsin(z))
    
    return round(ra, 4), round(dec, 4)

# --- Main Thread ---
if __name__ == '__main__':
    # Force the script to use the absolute base directory
    base_dir = os.path.expanduser('~/GalSim_Training_Data')
    
    # Create the dedicated output folder inside ~/GalSim
    output_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load Local Cache ---
    print("Loading Master Star Catalog from local solid-state drive...")
    # Point explicitly to the catalog_cache folder where the ESA data lives
    cache_file = os.path.join(base_dir, "GAIADR3_master_star_cache_11.csv")
    
    if not os.path.exists(cache_file):
        print(f"ERROR: Cannot find {cache_file}. Run build_cache.py first!")
        exit()
        
    master_df = pd.read_csv(cache_file)
    print(f"--> Loaded {len(master_df)} stars into RAM instantly.\n")

    # --- 2. Define Random Targets ---
    num_images_to_generate = 10  # <--- CHANGE THIS NUMBER TO GENERATE MORE!
    # --- Image Setup (Real-World Optics) ---
    image_size_x = 3840   
    image_size_y = 2160
    # Your Physical Hardware Specs
    pixel_size_um = 2.9 #microns (e.g., IMX482 has 5.8µm pixels)
    focal_length_mm = 416
    
    ### For Generating Well Know Positions ###
    image_targets = [
        (1, 83.82, -5.0),   # Orion's Belt
        (2, 101.28, -16.7), # Sirius (Canis Major)
        (3, 201.36, -11.1), # Spica (Virgo)
        (4, 279.23, 38.78), # Vega (Lyra)
        (5, 15.00, 60.00),  # Cassiopeia region
        (6, 180.00, 0.0),   # Celestial Equator
        (7, 45.00, 89.0),   # Polaris (North Celestial Pole)
        (8, 250.00, -60.0)  # Southern Hemisphere deep sky
    ]

    tasks = []
    for (img_id, t_ra, t_dec) in image_targets:
        tasks.append((img_id, t_ra, t_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, master_df, output_dir))

    ### For Generating Random Positions ###
    # # Package the arguments for the workers using our new random math
    # tasks = []
    # print(f"Calculating {num_images_to_generate} random spherical coordinates...")
    
    # for img_id in range(1, num_images_to_generate + 1):
    #     t_ra, t_dec = get_random_sky_coord()
    #     tasks.append((img_id, t_ra, t_dec, image_size_x, image_size_y, pixel_size_um, focal_length_mm, master_df, output_dir))
        
    num_cores = mp.cpu_count()
    print(f"Firing up {num_cores} autonomous cores to generate {len(tasks)} whole-sky images...")
    
    generation_start = time.time()
    
    # --- Parallel Dispatch ---
    with mp.Pool(processes=num_cores) as pool:
        for log_message in pool.imap_unordered(generate_single_image, tasks):
            print(log_message)
            
    print(f"\nUniversal Dataset complete! Total Time: {time.time() - generation_start:.2f} seconds.")
    print(f"Check the '{output_dir}' folder.")