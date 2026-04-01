import galsim
import os
import numpy as np
import csv
import warnings
import time
import random
import multiprocessing as mp

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
    image_id, target_ra, target_dec, fov, output_dir = args
    
    # [ANTI-BAN MEASURE]: Stagger the network requests so Vizier doesn't block the Pi
    time.sleep(random.uniform(0.1, 3.0)) 
    
    process_start = time.time()
    
    # --- A. Fetch Localized Data ---
    print(f"[Core] Image {image_id:04d} | Fetching stars around RA:{target_ra}, DEC:{target_dec}...")
    v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag'], column_filters={'Gmag': '<5.0'}) 
    v.TIMEOUT = 6000 
    v.ROW_LIMIT = -1 
    
    try:
        result = v.query_region(coord.SkyCoord(ra=target_ra, dec=target_dec, unit=(u.deg, u.deg)),
                                radius=(fov * 0.75) * u.deg, catalog='I/355/gaiadr3')
        
        if len(result) == 0:
            return f"[Core] Image {image_id:04d} FAILED: No stars found at RA:{target_ra}, DEC:{target_dec}"
        star_table = result[0]
    except Exception as e:
        return f"[Core] Image {image_id:04d} FAILED: Network Error - {e}"

    # --- B. Image Setup ---
    image_size = 1024
    pixel_scale = (fov * 3600) / image_size 
    
    fits_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.fits')
    csv_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.csv')
    png_filename = os.path.join(output_dir, f'starfield_{image_id:04d}.png')
    
    image = galsim.ImageF(image_size, image_size)
    
    # -pixel_scale on X-axis so RA increases to the LEFT
    affine = galsim.AffineTransform(-pixel_scale, 0, 0, pixel_scale, origin=image.true_center)
    world_origin = galsim.CelestialCoord(target_ra * galsim.degrees, target_dec * galsim.degrees)
    
    wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
    image.wcs = wcs
    
    # --- C. Draw Stars ---
    F0 = 1e6
    label_data = []
    stars_drawn = 0
    
    for i, row in enumerate(star_table):
        ra_val = row['RA_ICRS'] * galsim.degrees
        dec_val = row['DE_ICRS'] * galsim.degrees
        mag = row['Gmag']
        
        if np.ma.is_masked(mag): continue
            
        flux = F0 * 10 ** ((-mag) / 2.5)
        star = galsim.Gaussian(flux=flux, sigma=100)
        
        world_pos = galsim.CelestialCoord(ra_val, dec_val)
        pixel_pos = wcs.toImage(world_pos)
        
        if image.bounds.includes(pixel_pos):
            star.drawImage(image=image, center=pixel_pos, add_to_image=True)
            label_data.append([i, round(pixel_pos.x, 2), round(pixel_pos.y, 2), round(flux, 2), mag])
            stars_drawn += 1

    # --- D. Sensor Noise ---
    image += 10.0 
    rng = galsim.BaseDeviate(image_id) # Unique static pattern per image
    noise = galsim.GaussianNoise(rng, sigma=2.0)
    image.addNoise(noise)
    
    # --- E. Export Files ---
    image.write(fits_filename)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['star_id', 'x_pixel', 'y_pixel', 'flux', 'magnitude'])
        writer.writerows(label_data)
        
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    img_array = image.array
    plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=10, vmax=np.percentile(img_array, 99.9)))
    plt.title(f'Image {image_id:04d} (RA:{target_ra:.2f}, DEC:{target_dec:.2f})', color='white')
    plt.axis('off')
    plt.savefig(png_filename, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    process_duration = time.time() - process_start
    return f"Image {image_id:04d} | RA:{target_ra:6.2f}, DEC:{target_dec:6.2f} | Stars: {stars_drawn:4d} | Time: {process_duration:5.2f}s"


# --- Main Thread ---
if __name__ == '__main__':
    output_dir = 'dataset_universal'
    os.makedirs(output_dir, exist_ok=True)
    
    fov = 30.0 # Global Field of View for the camera
    
    # --- Define Universal Targets ---
    # We are now pulling from entirely different parts of the celestial sphere
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
    
    # Package the arguments for the workers
    tasks = []
    for (img_id, t_ra, t_dec) in image_targets:
        tasks.append((img_id, t_ra, t_dec, fov, output_dir))
        
    num_cores = mp.cpu_count()
    print(f"Firing up {num_cores} autonomous cores to generate {len(image_targets)} whole-sky images...")
    
    generation_start = time.time()
    
    # --- Parallel Dispatch ---
    with mp.Pool(processes=num_cores) as pool:
        # pool.imap_unordered prints the log the exact second a core finishes its specific image
        for log_message in pool.imap_unordered(generate_single_image, tasks):
            print(log_message)
            
    print(f"\nUniversal Dataset complete! Total Time: {time.time() - generation_start:.2f} seconds.")
    print(f"Check the '{output_dir}' folder.")