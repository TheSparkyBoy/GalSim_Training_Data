import os
import pandas as pd
import time
from astroquery.gaia import Gaia
import warnings

warnings.filterwarnings('ignore')

def build_local_cache(filename="GAIADR3_master_star_cache.csv", max_magnitude=5.0):
    print(f"Connecting to ESA Gaia Supercomputer (Mag < {max_magnitude})...")
    
    # ADQL (Astronomical Data Query Language)
    # We use 'AS' to perfectly match the column names your generator script expects
    query = f"""
    SELECT ra AS "RA_ICRS", 
           dec AS "DE_ICRS", 
           phot_g_mean_mag AS "Gmag"
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < {max_magnitude}
    """
    
    print("Executing full-sky ADQL query. Waiting for ESA servers...")
    start_time = time.time()
    
    try:
        # Send the job to the ESA archive
        job = Gaia.launch_job_async(query)
        result_table = job.get_results()
        
        # Convert directly to a Pandas DataFrame
        master_df = result_table.to_pandas()
        
        # Clean out any stars that might be missing magnitude data
        master_df = master_df.dropna(subset=['Gmag'])
        
    except Exception as e:
        print(f"\n[ERROR] Failed to query Gaia database: {e}")
        return

    # --- Save to Hard Drive ---
    base_dir = os.path.expanduser('~/GalSim')
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename)
    
    master_df.to_csv(filepath, index=False)
    
    total_time = time.time() - start_time
    print(f"\nSUCCESS! Downloaded {len(master_df)} flawless stars.")
    print(f"Total Time: {total_time:.2f} seconds.")
    print(f"Saved to: {filepath}")

if __name__ == '__main__':
    # You can safely push this to 9.0 or 10.0 if you want a massive dataset!
    build_local_cache(max_magnitude=8.0)