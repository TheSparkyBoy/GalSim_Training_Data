import os
import pandas as pd
import time
from astroquery.gaia import Gaia
import warnings

warnings.filterwarnings('ignore')

def build_local_cache(filename="GAIADR3_master_star_cache.csv", max_magnitude=7.0, chunk_size=15):
    print(f"Connecting to ESA Gaia Supercomputer (Mag < {max_magnitude})...")
    
    # Remove the global row limit so Gaia doesn't truncate our chunks
    Gaia.ROW_LIMIT = -1
    
    all_chunks = []
    start_time = time.time()

    # Slice the sky from -90 degrees to +90 degrees
    for dec_start in range(-90, 90, chunk_size):
        dec_end = dec_start + chunk_size
        print(f"--> Querying Declination {dec_start:3d}° to {dec_end:3d}°...")

        # We inject the Dec limits directly into the ADQL WHERE clause
        query = f"""
        SELECT source_id,
               ra AS "RA_ICRS", 
               dec AS "DE_ICRS", 
               phot_g_mean_mag AS "Gmag"
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag < {max_magnitude}
        AND dec >= {dec_start} AND dec < {dec_end}
        """
        
        try:
            # Send the job to the ESA archive
            job = Gaia.launch_job_async(query)
            result_table = job.get_results()
            
            # Convert to Pandas and clean missing magnitudes
            chunk_df = result_table.to_pandas()
            chunk_df = chunk_df.dropna(subset=['Gmag'])
            
            all_chunks.append(chunk_df)
            print(f"    Found {len(chunk_df):,} stars in this band.")
            
        except Exception as e:
            print(f"    [ERROR] Failed on band {dec_start} to {dec_end}: {e}")
        
        # [ANTI-BAN MEASURE]: Let the ESA server cool down for 3 seconds
        time.sleep(3.0)

    # --- Stitch the chunks together ---
    print("\nStitching universe together...")
    if all_chunks:
        master_df = pd.concat(all_chunks, ignore_index=True)
    else:
        print("Error: No data was downloaded.")
        return

    # --- Save to Hard Drive ---
    base_dir = os.path.expanduser('~/GalSim_Training_Data')
    os.makedirs(base_dir, exist_ok=True)
    filepath = os.path.join(base_dir, filename)
    
    master_df.to_csv(filepath, index=False)
    
    total_time = time.time() - start_time
    print(f"\nSUCCESS! Downloaded {len(master_df):,} flawless stars.")
    print(f"Total Time: {total_time:.2f} seconds.")
    print(f"Saved to: {filepath}")

if __name__ == '__main__':
    # A max_magnitude of 12.0 with a chunk size of 15 degrees is the sweet spot
    build_local_cache(filename="GAIADR3_master_star_cache_12.csv", max_magnitude=12.0, chunk_size=15)