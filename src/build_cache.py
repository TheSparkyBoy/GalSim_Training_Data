import os
import pandas as pd
import time
import random
import multiprocessing as mp
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
import warnings

warnings.filterwarnings('ignore')

# --- 1. The Worker Function (Runs independently on each core) ---
def fetch_band(args):
    """Fetches a single 15-degree declination band from Vizier."""
    dec, max_magnitude = args
    
    # [ANTI-BAN JITTER]: Force the core to wait a random amount of time (0.5s to 3.5s)
    # This staggers the 4 cores so they don't hit the Vizier server at the exact same millisecond.
    time.sleep(random.uniform(0.5, 3.5))
    
    # Instantiate a fresh Vizier object for this specific core
    v = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Gmag'], column_filters={'Gmag': f'<{max_magnitude}'})
    v.TIMEOUT = 6000
    v.ROW_LIMIT = -1 
    
    # Define a rectangular box that wraps all the way around the equator
    center = coord.SkyCoord(ra=180, dec=dec + 7.5, unit=(u.deg, u.deg))
    
    try:
        start_time = time.time()
        result = v.query_region(center, width=360 * u.deg, height=15 * u.deg, catalog='I/355/gaiadr3')
        end_time = time.time()
        
        if len(result) > 0:
            table = result[0].to_pandas()
            return f"   -> Band {dec:3d}° to {dec+15:3d}° | Found {len(table):6d} stars | Time: {end_time - start_time:.2f}s", table
        else:
            return f"   -> Band {dec:3d}° to {dec+15:3d}° | No stars found.", None
            
    except Exception as e:
        return f"   -> [ERROR] Pulling band {dec}°: {e}", None


# --- 2. Main Thread ---
def build_local_cache(filename="GAIADR3_master_star_cache.csv", max_magnitude=8.0):
    print(f"Starting Quad-Core Whole-Sky Harvester (Mag < {max_magnitude})...")
    
    # Create the tasks (the slices of the sky)
    dec_bands = range(-90, 90, 15)
    tasks = [(dec, max_magnitude) for dec in dec_bands]
    
    all_stars = []
    num_cores = mp.cpu_count()
    print(f"Firing up {num_cores} cores. Staggering network requests to protect Vizier limits...\n")
    
    total_start = time.time()
    
    # Launch the multi-core processing pool
    with mp.Pool(processes=num_cores) as pool:
        # pool.imap_unordered lets us print the results the exact second a core finishes its band
        for log_message, df in pool.imap_unordered(fetch_band, tasks):
            print(log_message)
            if df is not None:
                all_stars.append(df)
                
    if not all_stars:
        print("\n[ERROR] No data was downloaded. Check your internet connection or Vizier status.")
        return

    # Combine all the slices into one massive database
    print("\nCompiling master database...")
    master_df = pd.concat(all_stars, ignore_index=True)
    
    # Drop any duplicates (just in case the bands overlapped slightly)
    master_df = master_df.drop_duplicates(subset=['RA_ICRS', 'DE_ICRS'])
    
    # Save to your hard drive
    master_df.to_csv(f'../{filename}', index=False)
    
    total_time = time.time() - total_start
    print(f"SUCCESS! Saved {len(master_df)} unique stars to '{filename}'.")
    print(f"Total Harvest Time: {total_time:.2f} seconds.")

if __name__ == '__main__':
    build_local_cache()