import pandas as pd
import time
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
import warnings

warnings.filterwarnings('ignore')

def build_local_cache(filename="master_star_cache.csv", max_magnitude=8.0):
    print(f"Starting Whole-Sky Harvester (Mag < {max_magnitude})...")
    
    v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag'], column_filters={'Gmag': f'<{max_magnitude}'})
    v.TIMEOUT = 6000
    v.ROW_LIMIT = -1 
    
    all_stars = []
    
    # Sweep the sky from the South Pole (-90) to the North Pole (+90) in 15-degree slices
    dec_bands = range(-90, 90, 15)
    
    for dec in dec_bands:
        print(f"Scanning Declination band: {dec}° to {dec+15}°...")
        
        # Define a rectangular box that wraps all the way around the equator (360 degrees of RA)
        center = coord.SkyCoord(ra=180, dec=dec + 7.5, unit=(u.deg, u.deg))
        
        try:
            # width=360 grabs the whole horizontal slice, height=15 grabs the vertical band
            result = v.query_region(center, width=360 * u.deg, height=15 * u.deg, catalog='I/355/gaiadr3')
            
            if len(result) > 0:
                table = result[0].to_pandas()
                all_stars.append(table)
                print(f"   -> Found {len(table)} stars.")
            
        except Exception as e:
            print(f"   -> Error pulling band {dec}: {e}")
            
        time.sleep(1) # Be polite to the server
        
    # Combine all the slices into one massive database
    print("\nCompiling master database...")
    master_df = pd.concat(all_stars, ignore_index=True)
    
    # Drop any duplicates (just in case the bands overlapped slightly)
    master_df = master_df.drop_duplicates(subset=['RA_ICRS', 'DE_ICRS'])
    
    # Save to your hard drive
    master_df.to_csv(filename, index=False)
    print(f"SUCCESS! Saved {len(master_df)} stars to '{filename}'.")

if __name__ == '__main__':
    build_local_cache()