import galsim
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u
import csv
import warnings
import time

# Suppress warnings for a clean terminal
warnings.filterwarnings('ignore')

# --- 1. Data Acquisition (Orion) ---
print("Querying Gaia DR3 for Orion constellation stars...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag'], 
           column_filters={'Gmag': '<5.0'}) 
v.TIMEOUT = 6000 #seconds
v.ROW_LIMIT = -1 #Critical, otherwise it only returns 50 rows by default. We want all stars in the region, even if it's a lot!

# Fixed Coordinates: RA 83.8, Dec -5.0 is near the celestial equator in Orion
center_ra, center_dec, fov = 83.82, -5.0, 30

# Searching a 30 degree radius pulls in stars far outside the square canvas.
query_radius = fov

fetch_start_time = time.time()

result = v.query_region(coord.SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg)),
                        radius=query_radius * u.deg, catalog='I/355/gaiadr3')
fetch_end_time = time.time()

star_table = result[0]
print(f"Downloaded {len(star_table)} stars within the canvas geometry.")
print(star_table)

# --- 2. Configuration & Setup ---
image_size = 1024
pixel_scale = (fov * 3600) / image_size  # Roughly 52.7 arcseconds per pixel

# Ensure output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
fits_filename = os.path.join(output_dir, 'orion_constellation.fits')
csv_filename = os.path.join(output_dir, 'orion_labels.csv')
png_filename = os.path.join(output_dir, 'orion_visual.png')

# Initialize the blank sensor
full_image = galsim.ImageF(image_size, image_size)

# WCS Setup: Map the affine transform to the TRUE center of the image grid
affine = galsim.AffineTransform(pixel_scale, 0, 0, pixel_scale, origin=full_image.true_center)
world_origin = galsim.CelestialCoord(center_ra * galsim.degrees, center_dec * galsim.degrees)

# Explicitly tell TanWCS that our affine transform is in arcseconds
wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
full_image.wcs = wcs
label_data = []

# --- 3. The Generation Loop ---
print(f"Drawing {len(star_table)} real stars onto the FITS canvas...")
stars_drawn = 0

F0 = 1e7
for i, row in enumerate(star_table):
    ra_val = row['RA_ICRS'] * galsim.degrees
    dec_val = row['DE_ICRS'] * galsim.degrees
    mag = row['Gmag']
    
    # Safely skip stars with missing magnitude data in Gaia
    if np.ma.is_masked(mag):
        continue
    
    # Convert magnitude to flux 
    flux = F0 * 10 ** ((-mag) / 2.5)
    
    # Create the star profile (Sigma=100 arcsec ensures it spans ~2 pixels)
    star = galsim.Gaussian(flux=flux, sigma=100)
    
    world_pos = galsim.CelestialCoord(ra_val, dec_val)
    pixel_pos = wcs.toImage(world_pos)
    
    # Safely check if the center of the star lands on our 1024x1024 sensor
    if full_image.bounds.includes(pixel_pos):
        # Draw the star
        star.drawImage(image=full_image, center=pixel_pos, add_to_image=True)
        
        # Save exact sub-pixel data for AI labels
        label_data.append([i, round(pixel_pos.x, 2), round(pixel_pos.y, 2), round(flux, 2), mag])
        stars_drawn += 1
        print(f"Star {i}: RA={ra_val}, Dec={dec_val}, Mag={mag} -> Pixel=({pixel_pos.x:.2f}, {pixel_pos.y:.2f}), Flux={flux:.2f}\n")
    else:
        print(f"Skipped Star {i}: RA={ra_val}, Dec={dec_val}, Mag={mag} -> Pixel=({pixel_pos.x:.2f}, {pixel_pos.y:.2f})\n")

print(f"\nSUCCESS: {stars_drawn} stars successfully landed inside the camera frame!\n")
print(f"Data fetch completed in {fetch_end_time - fetch_start_time:.2f} seconds.")


# --- 4. Sensor Effects ---
print("Applying sensor noise...")
# Add a base sky background so negative noise values don't break logarithms
full_image += 10.0 
noise = galsim.GaussianNoise(sigma=2.0)
full_image.addNoise(noise)

# --- 5. Export ---
print(f"Saving files to '{output_dir}/' folder...")
full_image.write(fits_filename)

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['star_id', 'x_pixel', 'y_pixel', 'flux', 'magnitude'])
    writer.writerows(label_data)

# Generate a visual PNG using Logarithmic scaling to handle the extreme contrast
plt.figure(figsize=(10, 10), facecolor='black')
img_array = full_image.array
plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=10, vmax=np.percentile(img_array, 99.9)))
plt.title(f'Orion Constellation (RA:{center_ra}, DEC:{center_dec})', color='white')
plt.axis('off')
plt.savefig(png_filename, bbox_inches='tight', facecolor='black')

print(f"Scientific FITS saved: {fits_filename}")
print(f"AI Labels saved: {csv_filename}")
print(f"Visual PNG saved: {png_filename}")