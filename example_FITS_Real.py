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
center_ra, center_dec, fov = 83.82, -5.0, 15.0
result = v.query_region(coord.SkyCoord(ra=center_ra, dec=center_dec, unit=(u.deg, u.deg)),
                        radius=fov * u.deg, catalog='I/355/gaiadr3')
star_table = result[0]
print(star_table)

