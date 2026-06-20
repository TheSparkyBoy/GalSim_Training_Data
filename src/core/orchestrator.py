# core/orchestrator.py
import os
import random
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from core.simulator import TelescopeSimulator

# Global bridge function required for mp.Pool
def worker_bridge(task_config):
    simulator = TelescopeSimulator(task_config)
    return simulator.run_pipeline()

class DatasetOrchestrator:
    def __init__(self, config):
        self.cfg = config
        self.base_dir = os.path.expanduser('~/GalSim_Training_Data')
        
        # Calculate FOV
        fov = round(206.264806247096355 * (self.cfg['pixel_size_um'] / self.cfg['focal_length_mm']) * self.cfg['image_size_x'] / 3600.0, 2)
        
        # Strip the '.csv' extension from the catalog name for a cleaner folder
        catalog_name = self.cfg['cache_filename'].replace('.csv', '')

        # --- UPDATED FOLDER NAMING LOGIC ---
        # Check if ANY anomalies are currently active in the config
        any_anomalies = any([
            self.cfg.get('anom_lens_distortion', False), 
            self.cfg.get('anom_false_stars', False),
            self.cfg.get('anom_drop_stars', False), 
            self.cfg.get('anom_pos_variation', False),
            self.cfg.get('anom_mag_variation', False), 
            self.cfg.get('anom_motion_smear', False)
        ])
        anom_status = "mixed_anomalies" if any_anomalies else "perfect_optics"
        
        # Build the dynamic dataset name
        self.dataset_name = (f"{self.cfg['mode']}{catalog_name}_seed_{self.cfg['global_seed']}_"
                             f"fov_{fov}_x_{self.cfg['image_size_x']}_y_{self.cfg['image_size_y']}_"
                             f"pxlsz_{self.cfg['pixel_size_um']}um_{self.cfg['focal_length_mm']}mm_"
                             f"{self.cfg['exposure_time']}s_mag11_roll{self.cfg['roll']}deg_{anom_status}")
        
        self.dirs = {
            'fits': os.path.join(self.base_dir, 'training_data', self.dataset_name, 'fits'),
            'png':  os.path.join(self.base_dir, 'training_data', self.dataset_name, 'png'),
            'csv':  os.path.join(self.base_dir, 'training_data', self.dataset_name, 'csv')
        }
        self.manifest_path = os.path.join(self.base_dir, 'training_data', 'dataset_manifest.csv')
        self.master_table = None

    def setup_directories(self):
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    def load_cache(self):
        print("Loading Master Star Catalog from local solid-state drive...")
        cache_file = os.path.join(self.base_dir, "master_star_caches", self.cfg['cache_filename'])
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cannot find {cache_file}. Run build_cache.py first!")
        self.master_table = pd.read_csv(cache_file, dtype={'source_id': str, 'star_id': str})
        
        print(f"--> Loaded {len(self.master_table)} stars into RAM instantly.\n")

    def get_starting_id(self):
        if os.path.exists(self.manifest_path):
            try:
                df = pd.read_csv(self.manifest_path)
                if not df.empty:
                    next_id = int(df['image_id'].max()) + 1
                    print(f"--> Memory loaded: Resuming at Universal ID {next_id:07d}...")
                    return next_id
            except Exception as e:
                print(f"Warning: Could not read manifest. ({e})")
        return 1

    def _get_random_sky_coord(self):
        ra = random.uniform(0.0, 360.0)
        z = random.uniform(-1.0, 1.0)
        dec = np.degrees(np.arcsin(z))
        return round(ra, 4), round(dec, 4)

    def build_task_list(self, starting_id):
        tasks = []
        global_img_id = starting_id
        
        well_known_targets = [
            (83.82, -5.0), (101.28, -16.7), (201.36, -11.1), (279.23, 38.78),
            (15.00, 60.00), (180.00, 0.0), (45.00, 89.0), (250.00, -60.0)
        ]

        def create_task(img_id, ra, dec):
            task = self.cfg.copy()
            task.update({
                'image_id': img_id,
                'ra': ra,
                'dec': dec,
                'dirs': self.dirs,
                'master_table': self.master_table
            })
            return task

        for t_ra, t_dec in well_known_targets:
            if len(tasks) >= self.cfg['total_images']: break
            tasks.append(create_task(global_img_id, t_ra, t_dec))
            global_img_id += 1

        if len(tasks) < self.cfg['total_images']:
            num_needed = self.cfg['total_images'] - len(tasks)
            print(f"Queued 8 well-known targets. Calculating {num_needed} additional random coordinates...")
            while len(tasks) < self.cfg['total_images']:
                t_ra, t_dec = self._get_random_sky_coord()
                tasks.append(create_task(global_img_id, t_ra, t_dec))
                global_img_id += 1
                
        return tasks

    def execute(self):
        self.setup_directories()
        self.load_cache()
        starting_id = self.get_starting_id()
        tasks = self.build_task_list(starting_id)
        
        num_cores = mp.cpu_count()
        print(f"\nFiring up {num_cores} autonomous cores for dataset: '{self.dataset_name}'...")
        
        generation_start = time.time()
        manifest_data = []
        
        with mp.Pool(processes=num_cores) as pool:
            for result in pool.imap_unordered(worker_bridge, tasks):
                print(f"Image {result['image_id']} | Roll: {result['roll']:6.2f} | Stars: {result['stars_drawn']:4d} | Med. SNR: {result['median_snr']} | Time: {result['time_s']}s")
                
                manifest_data.append({
                    'image_id': result['image_id'],
                    'dataset_group': self.dataset_name,
                    'ra': result['ra'],
                    'dec': result['dec'],
                    'camera_roll': result['roll'],
                    'focal_length_mm': self.cfg['focal_length_mm'],
                    'exposure_time_s': self.cfg['exposure_time'],
                    'total_stars': result['stars_drawn'],
                    'median_image_snr': result['median_snr'],
                    'bg_mean_e': result['bg_mean_e'],
                    'bg_std_e': result['bg_std_e'],
                    # --- QUANTITATIVE METRICS ---
                    'distorted_stars': result['distorted_stars'],
                    'dropped_stars': result['dropped_stars'],
                    'false_stars': result['false_stars'],
                    'smear_px': result['smear_px'],
                    
                    # --- CONFIGURATION TOGGLE STATES ---
                    'anom_lens_on': self.cfg.get('anom_lens_distortion', False),
                    'anom_false_on': self.cfg.get('anom_false_stars', False),
                    'anom_drop_on': self.cfg.get('anom_drop_stars', False),
                    'anom_pos_on': self.cfg.get('anom_pos_variation', False),
                    'anom_mag_on': self.cfg.get('anom_mag_variation', False),
                    'anom_smear_on': self.cfg.get('anom_motion_smear', False)
                })
                
        if manifest_data:
            df = pd.DataFrame(manifest_data).sort_values(by='image_id')
            if os.path.exists(self.manifest_path):
                df.to_csv(self.manifest_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.manifest_path, index=False)
                
            print(f"\nUniversal Dataset complete! Total Time: {time.time() - generation_start:.2f} seconds.")
            print(f"Global metadata appended to: {self.manifest_path}")
        else:
            print("\nZero tasks were queued!")