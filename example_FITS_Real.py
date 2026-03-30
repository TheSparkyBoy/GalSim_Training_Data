import galsim
import os
import numpy as np
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u

# --- PART 1: The Fast Data Pull ---
print("Querying Orion stars...")
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'phot_g_mean_mag'], 
           column_filters={'phot_g_mean_mag': '2.0..7.0'})
v.TIMEOUT = 200
center_ra, center_dec = 83.82, -5.0
result = v.query_region(coord.SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg)),
                        radius=2.0 * u.deg, catalog='I/355/gaiadr3')
star_table = result[0]
print(star_table.colnames)

# --- PART 2: The GalSim Config Dictionary ---
# This is the Python equivalent of the 'imsim-user.yaml'
config = {
    'image': {
        'type': 'Single',
        'size': 1024,
        'pixel_scale': 10.0,
        'world_orientation': {
            'type': 'Tan',
            'ra': f'{center_ra} deg',
            'dec': f'{center_dec} deg',
        },
        'noise': {'type': 'Gaussian', 'sigma': 15.0},
        'nproc': 4 
    },
    'output': {
        'file_name': 'orion_full_constellation.fits',
        'dir': 'output_images' # Optional: folder name
    },
    'objects': {'type': 'List', 'items': []}
}

# --- PART 3: Injecting the Stars into the Config ---
# --- Updated Mapping Section ---
# Use the exact names printed in your output: ['RA_ICRS', 'DE_ICRS', 'Gmag']

for row in star_table:
    mag = row['Gmag']  # Changed from 'phot_g_mean_mag'
    
    # Standard flux conversion
    flux = 10**((10 - mag) / 2.5) * 5000
    
    star_obj = {
        'type': 'Gaussian',
        'sigma': 1.2,
        'flux': flux,
        'world_pos': {
            'type': 'Celestial',
            'ra': f"{row['RA_ICRS']} deg",  # Matching your printed colnames
            'dec': f"{row['DE_ICRS']} deg"
        }
    }
    config['objects']['items'].append(star_obj)

# --- PART 4: Run the Simulation ---
print(f"Generating image with {len(star_table)} stars using 4 cores...")
# Create the directory if it doesn't exist
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# This one line handles the multi-core forking, the drawing, 
# the noise, and the saving to the filename in the config.
galsim.config.Process(config)

print("Success! Check the output_images folder.")

# Note: In a 'Single' image config, GalSim writes to the filename 
# specified in an 'output' block, or you can manually save:
final_image = galsim.config.BuildImage(config)
final_image.write('orion_config_example.fits')
print("Done! View orion_config_example.fits")