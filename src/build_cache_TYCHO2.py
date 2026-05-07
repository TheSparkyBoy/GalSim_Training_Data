import os
import pandas as pd
import time
from astroquery.vizier import Vizier
import warnings

warnings.filterwarnings('ignore')

def build_tycho2_cache(max_magnitude=8.0, chunk_size=30):
    filename=("TYCHO2_master_star_cache_" + str(max_magnitude) + ".csv")
    print(f"Connecting to VizieR for Tycho-2 Catalog (Mag < {max_magnitude})...")
    
    # VizieR ID for Tycho-2
    catalog_id = 'I/259/tyc2'
    
    # Set up VizieR to fetch unlimited rows per query
    v = Vizier(columns=['TYC1', 'TYC2', 'TYC3', 'RAmdeg', 'DEmdeg', 'VTmag'])
    v.ROW_LIMIT = -1
    
    all_chunks = []
    start_time = time.time()

    # Slice the sky from -90 to +90 to avoid database timeouts
    for dec_start in range(-90, 90, chunk_size):
        dec_end = dec_start + chunk_size
        print(f"--> Querying Declination {dec_start:3d}° to {dec_end:3d}°...")
        
        try:
            # We constrain both Magnitude and Declination
            result = v.query_constraints(
                catalog=catalog_id, 
                VTmag=f'<{max_magnitude}',
                DEmdeg=f'>={dec_start} & <{dec_end}'
            )
            
            if len(result) > 0:
                chunk_df = result[0].to_pandas()
                
                # Clean missing magnitudes
                chunk_df = chunk_df.dropna(subset=['VTmag'])
                
                all_chunks.append(chunk_df)
                print(f"    Found {len(chunk_df):,} stars.")
            else:
                print("    No stars found in this band.")
                
        except Exception as e:
            print(f"    [ERROR] Failed on band {dec_start} to {dec_end}: {e}")
        
        # Give the VizieR server a tiny breather
        time.sleep(2.0)

    print("\nStitching Tycho-2 together...")
    if all_chunks:
        master_df = pd.concat(all_chunks, ignore_index=True)
        
        # --- The TYC to source_id Conversion ---
        # We mathematically fuse the 3 columns into a single unique integer.
        # Formula: (TYC1 * 1,000,000) + (TYC2 * 10) + TYC3
        master_df['source_id'] = (master_df['TYC1'].astype('int64') * 1000000) + \
                                 (master_df['TYC2'].astype('int64') * 10) + \
                                 master_df['TYC3'].astype('int64')
        
        # Rename columns to match Gaia DR3 format exactly
        master_df = master_df.rename(columns={
            'RAmdeg': 'RA_ICRS', 
            'DEmdeg': 'DE_ICRS', 
            'VTmag': 'Gmag'
        })
        
        # Drop the old TYC columns to keep the CSV clean and small
        master_df = master_df[['source_id', 'RA_ICRS', 'DE_ICRS', 'Gmag']]
        
        # --- Save to Hard Drive ---
        base_dir = os.path.expanduser('~/GalSim_Training_Data')
        filepath = os.path.join(base_dir,'master_star_caches', filename)
        
        master_df.to_csv(filepath, index=False)
        
        total_time = time.time() - start_time
        print(f"\nSUCCESS! Downloaded {len(master_df):,} Tycho-2 stars.")
        print(f"Total Time: {total_time:.2f} seconds.")
        print(f"Saved to: {filepath}")
    else:
        print("Error: No data was downloaded.")

if __name__ == '__main__':
    # Magnitude 12.0 pulls roughly the ~40,000 brightest stars in the sky
    build_tycho2_cache(max_magnitude=6, chunk_size=30)