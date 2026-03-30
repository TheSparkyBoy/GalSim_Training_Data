import galsim
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u

# 1. Define the "Wide-Angle" Pin (Center of Orion's Body)
# This point is roughly between the Belt and the Sword
center_ra = 83.82    # ~05h 35m
center_dec = -5.0    # Lowered to capture Rigel
fov_radius = 12.0    # 12-degree radius = 24-degree total width

# 2. Query Gaia for the Iconic Stars
print(f"Querying Gaia for the entire Orion constellation...")
# We filter for stars brighter than Mag 7.5 to keep the iconic shape clear
v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'phot_g_mean_mag'], 
           column_filters={'phot_g_mean_mag': '<7.5'}) 
v.ROW_LIMIT = 5000

result = v.query_region(coord.SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg), frame='icrs'),
                        radius=fov_radius * u.deg, 
                        catalog='I/355/gaiadr3')

star_table = result[0]
print(f"Captured {len(star_table)} major stars in the Orion region.")

# 3. Setup GalSim "Wide-Angle" Camera
image_size = 1024
# Calculation: (24 degrees * 3600") / 1024 pixels = ~84 arcsec/pixel
pixel_scale = 84.0   
affine = galsim.AffineTransform(pixel_scale, 0, 0, pixel_scale)
wcs = galsim.TanWCS(affine, world_origin=galsim.CelestialCoord(center_ra * galsim.degrees, center_dec * galsim.degrees))
full_image = galsim.ImageF(image_size, image_size, wcs=wcs)

# 4. Render the Constellation
for row in star_table:
    ra_val = row['RA_ICRS'] * galsim.degrees
    dec_val = row['DE_ICRS'] * galsim.degrees
    mag = row['phot_g_mean_mag']
    
    # Adjusting flux for the wide-angle view
    flux = 10**((8 - mag) / 2.5) * 5000
    
    # We use a slightly larger sigma because we are zoomed way out
    star_model = galsim.Gaussian(flux=flux, sigma=1.0)
    star_coord = galsim.CelestialCoord(ra_val, dec_val)
    
    try:
        star_model.drawImage(image=full_image, center=star_coord, add_to_image=True)
    except galsim.GalSimError:
        continue 

# 5. Save
full_image.write('full_orion_constellation.fits')
print("Success! 'full_orion_constellation.fits' generated.")