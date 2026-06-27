import os
import pandas as pd

def sync_manifest():
    print("==================================================")
    print("  AEROSPACE DATASET MANIFEST SYNCHRONIZER ")
    print("==================================================")
    
    base_dir = os.path.expanduser('~/GalSim_Training_Data/training_data')
    manifest_path = os.path.join(base_dir, 'dataset_manifest.csv')
    
    if not os.path.exists(manifest_path):
        print(f"\n[ERROR] Master manifest not found at:\n{manifest_path}")
        print("There is nothing to synchronize.")
        return

    # 1. Load the current manifest
    print("\n[1/3] Loading Master Ledger into memory...")
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"[ERROR] Could not read manifest: {e}")
        return
        
    original_count = len(df)
    print(f"      Found {original_count} total entries.")

    # 2. Deduplicate
    print("\n[2/3] Scanning for duplicate database entries...")
    # Drop rows that have the exact same image_id and dataset_group
    df = df.drop_duplicates(subset=['image_id', 'dataset_group'], keep='last')
    dedup_count = len(df)
    duplicates_removed = original_count - dedup_count
    if duplicates_removed > 0:
        print(f"      [FIXED] Removed {duplicates_removed} duplicate row(s).")
    else:
        print("      No duplicates found. Ledger is clean.")

    # 3. Verify physical files on SSD
    print("\n[3/3] Verifying physical file integrity on solid-state drive...")
    valid_rows = []
    missing_count = 0
    
    for index, row in df.iterrows():
        dataset_group = str(row['dataset_group'])
        # Simulator pads IDs to 7 digits (e.g., '0000001')
        img_id_str = str(row['image_id']).zfill(7)
        
        # Construct expected physical paths
        csv_path = os.path.join(base_dir, dataset_group, 'csv', f"{img_id_str}.csv")
        fits_path = os.path.join(base_dir, dataset_group, 'fits', f"{img_id_str}.fits")
        png_path = os.path.join(base_dir, dataset_group, 'png', f"{img_id_str}.png")
        
        # Check if ALL three critical files exist for this entry
        if os.path.exists(csv_path) and os.path.exists(fits_path) and os.path.exists(png_path):
            valid_rows.append(row)
        else:
            missing_count += 1

    clean_df = pd.DataFrame(valid_rows)
    
    if missing_count > 0:
        print(f"      [FIXED] Found {missing_count} 'ghost' entries in the ledger where files were missing.")
        print("      Purging missing entries from the manifest...")
    else:
        print("      100% physical file integrity verified. No ghost files found.")

    # 4. Save the synchronized manifest
    print("\n==================================================")
    if len(clean_df) < original_count:
        clean_df.to_csv(manifest_path, index=False)
        print(f"SUCCESS: Synchronized manifest saved with {len(clean_df)} verified entries.")
    else:
        print("SUCCESS: Ledger is perfectly synced. No changes were necessary.")
    print("==================================================\n")

if __name__ == "__main__":
    sync_manifest()