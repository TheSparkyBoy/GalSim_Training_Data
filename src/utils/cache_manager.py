# utils/cache_manager.py
import os
import time
import pandas as pd
import warnings
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

warnings.filterwarnings('ignore')

class CacheManager:
    """Handles downloading and formatting stellar catalogs from global databases."""
    
    def __init__(self, base_dir='~/GalSim_Training_Data'):
        self.base_dir = os.path.expanduser(base_dir)
        self.cache_dir = os.path.join(self.base_dir, 'master_star_caches')
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_gaiadr3_cache(self, max_magnitude=12.0, chunk_size=5):
        """Connects to the ESA Supercomputer to pull Gaia DR3 data."""
        filename = f"GAIADR3_master_star_cache_{int(max_magnitude)}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        
        print(f"Connecting to ESA Gaia Supercomputer (Mag < {max_magnitude})...")
        Gaia.ROW_LIMIT = -1
        all_chunks = []
        start_time = time.time()

        for dec_start in range(-90, 90, chunk_size):
            dec_end = dec_start + chunk_size
            print(f"--> Querying Declination {dec_start:3d}° to {dec_end:3d}°...")

            query = f"""
            SELECT source_id, ra AS "RA_ICRS", dec AS "DE_ICRS", phot_g_mean_mag AS "Gmag"
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag < {max_magnitude}
            AND dec >= {dec_start} AND dec < {dec_end}
            """
            
            # Automatic Retry Loop for ESA Timeouts
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    job = Gaia.launch_job_async(query)
                    result_table = job.get_results()
                    chunk_df = result_table.to_pandas()
                    
                    # Protect the 64-bit ID from Pandas float corruption
                    chunk_df['source_id'] = chunk_df['source_id'].astype(str)
                    chunk_df = chunk_df.dropna(subset=['Gmag'])
                    
                    all_chunks.append(chunk_df)
                    print(f"    Found {len(chunk_df):,} stars.")
                    break 
                except Exception as e:
                    print(f"    [WARNING] Timeout on band {dec_start}. Attempt {attempt+1}/{max_retries}. Retrying in 10s... ({e})")
                    time.sleep(10.0)
                    if attempt == max_retries - 1:
                        print(f"    [CRITICAL ERROR] Failed to download {dec_start} to {dec_end}. Dataset will have a hole.")
            
            time.sleep(2.0)

        self._stitch_and_save(all_chunks, filepath, "Gaia DR3", start_time)

    def build_tycho2_cache(self, max_magnitude=12.0, chunk_size=30):
        """Connects to VizieR to pull Tycho-2 data and formats it to match Gaia."""
        filename = f"TYCHO2_master_star_cache_{int(max_magnitude)}.csv"
        filepath = os.path.join(self.cache_dir, filename)
        
        print(f"Connecting to VizieR for Tycho-2 Catalog (Mag < {max_magnitude})...")
        v = Vizier(columns=['TYC1', 'TYC2', 'TYC3', 'RAmdeg', 'DEmdeg', 'VTmag'])
        v.ROW_LIMIT = -1
        
        all_chunks = []
        start_time = time.time()

        for dec_start in range(-90, 90, chunk_size):
            dec_end = dec_start + chunk_size
            print(f"--> Querying Declination {dec_start:3d}° to {dec_end:3d}°...")
            
            try:
                result = v.query_constraints(catalog='I/259/tyc2', VTmag=f'<{max_magnitude}', DEmdeg=f'>={dec_start} & <{dec_end}')
                if len(result) > 0:
                    chunk_df = result[0].to_pandas()
                    chunk_df = chunk_df.dropna(subset=['VTmag'])
                    all_chunks.append(chunk_df)
                    print(f"    Found {len(chunk_df):,} stars.")
                else:
                    print("    No stars found in this band.")
            except Exception as e:
                print(f"    [ERROR] Failed on band {dec_start} to {dec_end}: {e}")
            
            time.sleep(2.0)

        print("\nStitching Tycho-2 together...")
        if all_chunks:
            master_df = pd.concat(all_chunks, ignore_index=True)
            
            # Create the Fused Integer ID and string ID
            master_df['source_id'] = (master_df['TYC1'].astype('int64') * 1000000) + \
                                     (master_df['TYC2'].astype('int64') * 10) + \
                                     master_df['TYC3'].astype('int64')
            master_df['source_id'] = master_df['source_id'].astype(str) # Protect as string
            
            master_df['TYC_ID'] = 'TYC ' + master_df['TYC1'].astype(str) + '-' + \
                                           master_df['TYC2'].astype(str) + '-' + \
                                           master_df['TYC3'].astype(str)
            
            # Standardize columns to match the Gaia format
            master_df = master_df.rename(columns={'RAmdeg': 'RA_ICRS', 'DEmdeg': 'DE_ICRS', 'VTmag': 'Gmag'})
            master_df = master_df[['source_id', 'TYC_ID', 'RA_ICRS', 'DE_ICRS', 'Gmag']]
            
            master_df.to_csv(filepath, index=False)
            print(f"\nSUCCESS! Downloaded {len(master_df):,} Tycho-2 stars.")
            print(f"Total Time: {time.time() - start_time:.2f} seconds.")
            print(f"Saved to: {filepath}")
        else:
            print("Error: No data was downloaded.")

    def _stitch_and_save(self, chunks, filepath, catalog_name, start_time):
        """Internal helper to concatenate and save chunked data."""
        print(f"\nStitching {catalog_name} universe together...")
        if chunks:
            master_df = pd.concat(chunks, ignore_index=True)
            master_df.to_csv(filepath, index=False)
            print(f"\nSUCCESS! Downloaded {len(master_df):,} {catalog_name} stars.")
            print(f"Total Time: {time.time() - start_time:.2f} seconds.")
            print(f"Saved to: {filepath}")
        else:
            print("Error: No data was downloaded.")


# =====================================================================
# STANDALONE EXECUTION
# =====================================================================
if __name__ == '__main__':
    # You can run this file directly from the terminal to manually download catalogs!
    # e.g., `python utils/cache_manager.py`
    
    manager = CacheManager()
    magnitude = 0
    while magnitude < 13.0:
        manager.build_gaiadr3_cache(max_magnitude=magnitude)
        manager.build_tycho2_cache(max_magnitude=magnitude)
        magnitude += 1.0
