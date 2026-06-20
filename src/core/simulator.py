# core/simulator.py
import galsim
import os
import numpy as np
import csv
import time
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.stats import sigma_clipped_stats

class TelescopeSimulator:
    """Handles the physical simulation and rendering of a single astronomical image."""
    
    def __init__(self, task_config):
        self.cfg = task_config
        self.image_id = self.cfg['image_id']
        self.target_ra = self.cfg['ra']
        self.target_dec = self.cfg['dec']
        
        # Deterministic seeding for this specific worker/image
        self.worker_seed = self.cfg['global_seed'] + int(self.image_id)
        random.seed(self.worker_seed)
        np.random.seed(self.worker_seed)
        
        # State variables
        self.image = galsim.ImageF(self.cfg['image_size_x'], self.cfg['image_size_y'])
        self.wcs = None
        self.pixel_scale = None
        self.stars_drawn = 0
        
        # Data storage
        self.temp_star_coords = [] # Holds coordinates temporarily until noise is applied
        self.label_data = []
        self.star_snr_list = []
        
        # Photometry Constants
        self.universal_baseline = 1e6 
        self.aperature_area_cm2 = np.pi * (6.5/2)**2 
        self.quantum_efficiency = 0.91 * 0.9 
        self.F0 = self.universal_baseline * self.aperature_area_cm2 * self.cfg['exposure_time'] * self.quantum_efficiency
        
    def setup_optics_and_wcs(self):
        self.pixel_scale = 206.264806247096355 * (self.cfg['pixel_size_um'] / self.cfg['focal_length_mm'])
        
        theta = np.radians(self.cfg['roll'])
        dudx = self.pixel_scale * np.cos(theta)
        dudy = -self.pixel_scale * np.sin(theta)
        dvdx = self.pixel_scale * np.sin(theta)
        dvdy = self.pixel_scale * np.cos(theta)
        
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=self.image.true_center)
        world_origin = galsim.CelestialCoord(self.target_ra * galsim.degrees, self.target_dec * galsim.degrees)
        
        self.wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
        self.image.wcs = self.wcs

    def draw_stars(self):
        """Draws the optical PSFs and saves their coordinates for later measurement."""
        global_defocus = random.uniform(-0.04, 0.04)
        center_x, center_y = self.cfg['image_size_x'] / 2.0, self.cfg['image_size_y'] / 2.0
        max_radius = np.sqrt(center_x**2 + center_y**2) 
        
        for _, row in self.cfg['master_table'].iterrows():
            mag = row['Gmag']
            if pd.isna(mag): continue
                
            world_pos = galsim.CelestialCoord(row['RA_ICRS'] * galsim.degrees, row['DE_ICRS'] * galsim.degrees)
            pixel_pos = self.wcs.toImage(world_pos)
            
            if self.image.bounds.includes(pixel_pos):
                flux = self.F0 * 10 ** ((-mag) / 2.5)
                
                dx, dy = pixel_pos.x - center_x, pixel_pos.y - center_y
                r_norm = np.sqrt(dx**2 + dy**2) / max_radius
                edge_coma = 0.04 * (r_norm ** 2)
                edge_astig = 0.03 * (r_norm ** 2)
                angle = np.arctan2(dy, dx)
                
                optical_psf = galsim.OpticalPSF(
                    lam=500.0, diam=0.065, defocus=global_defocus, spher=0.01,             
                    astig1=edge_astig * np.cos(2*angle), astig2=edge_astig * np.sin(2*angle),
                    coma1=edge_coma * np.cos(angle), coma2=edge_coma * np.sin(angle),
                    gsparams=galsim.GSParams(folding_threshold=1e-3, maximum_fft_size=32768)
                )
                
                star = optical_psf.withFlux(flux)
                star.drawImage(image=self.image, center=pixel_pos, add_to_image=True, method='phot')
                
                # Save the star data temporarily, DO NOT calculate SNR yet
                self.temp_star_coords.append({
                    'id': int(row['source_id']),
                    'x': pixel_pos.x,
                    'y': pixel_pos.y,
                    'mag': round(mag, 3)
                })
                self.stars_drawn += 1

    def apply_sensor_noise(self):
        self.image += 500.0 # background level
        rng = galsim.BaseDeviate(int(self.image_id))
        
        self.image.addNoise(galsim.PoissonNoise(rng))
        self.image.addNoise(galsim.GaussianNoise(rng, sigma=0.7))
        self.image.quantize()
        self.image.array[self.image.array > 4095] = 4095

    def measure_snr(self, pixel_x, pixel_y):
        """Performs true aperture photometry on a specific coordinate to measure SNR."""
        r_inner = 2.5  
        r_bg_in = 4.0  
        r_bg_out = 7.0 
        
        box_size = int(np.ceil(r_bg_out)) + 2
        x_min, x_max = int(pixel_x) - box_size, int(pixel_x) + box_size
        y_min, y_max = int(pixel_y) - box_size, int(pixel_y) + box_size
        
        if x_min < 0 or y_min < 0 or x_max >= self.cfg['image_size_x'] or y_max >= self.cfg['image_size_y']:
            return 0.0 
            
        cutout = self.image.array[y_min:y_max, x_min:x_max]
        
        y_grid, x_grid = np.ogrid[-box_size:box_size, -box_size:box_size]
        
        sub_x = pixel_x - int(pixel_x)
        sub_y = pixel_y - int(pixel_y)
        r_grid = np.sqrt((x_grid - sub_x)**2 + (y_grid - sub_y)**2)
        
        star_mask = r_grid <= r_inner
        bg_mask = (r_grid >= r_bg_in) & (r_grid <= r_bg_out)
        
        n_px = np.sum(star_mask)
        n_b = np.sum(bg_mask)
        
        # 1. Use the GLOBAL median to subtract the sky glow
        N_S_measured = self.bg_stats['mean_adu']
        
        raw_star_flux = np.sum(cutout[star_mask])
        N_star_measured = raw_star_flux - (N_S_measured * n_px)
        
        if N_star_measured <= 0:
            return 0.0 
            
        # 2. Use the GLOBAL standard deviation to calculate the variance
        # We square the standard deviation to get variance
        global_bg_variance = self.bg_stats['std_e'] ** 2
        
        # 3. Final SNR Math
        bg_penalty = 1.0 + (n_px / n_b)
        A_term = n_px * bg_penalty * global_bg_variance
        
        measured_snr = N_star_measured / np.sqrt(N_star_measured + A_term)
        
        return round(measured_snr, 2)

    def extract_final_labels(self):
        """Loops through the saved coordinates and measures SNR on the noisy image."""
        for star in self.temp_star_coords:
            final_snr = self.measure_snr(star['x'], star['y'])
            
            self.star_snr_list.append(final_snr)
            self.label_data.append([
                star['id'], 
                round(star['x'], 2), 
                round(star['y'], 2), 
                star['mag'], 
                self.cfg['focal_length_mm'], 
                self.cfg['exposure_time'], 
                final_snr
            ])

    def measure_global_background(self):
        """
        Uses Sigma Clipping to mask out stars and measure the true background 
        mean and standard deviation in both ADU and Electrons.
        """
        # 1. Run the Sigma Clipping algorithm on your image array
        bg_mean_adu, bg_median_adu, bg_std_adu = sigma_clipped_stats(
            self.image.array, 
            sigma=3.0, 
            maxiters=5
        )
        
        # 2. Convert from ADU to Electrons
        gain_e_per_adu = 1.0 
        
        bg_mean_e = bg_mean_adu * gain_e_per_adu
        bg_std_e = bg_std_adu * gain_e_per_adu
        
        # Print or return the results
        # print(f"Background Mean: {bg_mean_adu:.2f} ADU ({bg_mean_e:.2f} e-)")
        # print(f"Background StdDev: {bg_std_adu:.2f} ADU ({bg_std_e:.2f} e-)")
        
        return {
            'mean_adu': round(bg_mean_adu, 2),
            'std_adu': round(bg_std_adu, 2),
            'mean_e': round(bg_mean_e, 2),
            'std_e': round(bg_std_e, 2)
        }

    def export_files(self):
        fits_path = os.path.join(self.cfg['dirs']['fits'], f'{self.image_id:07d}.fits')
        png_path = os.path.join(self.cfg['dirs']['png'], f'{self.image_id:07d}.png')
        csv_path = os.path.join(self.cfg['dirs']['csv'], f'{self.image_id:07d}.csv')
        
        self.image.write(fits_path)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['star_id', 'x_image', 'y_image', 'flux_mag', 'focal_length', 'exposure_time', 'snr'])
            writer.writerows(self.label_data)
            
        fig = plt.figure(dpi=300, figsize=(16, 9), facecolor='black')
        img_array = self.image.array
        vmin = max(1.0, np.percentile(img_array, 1))
        vmax = np.max(img_array)
        if vmax <= vmin:
            vmax = vmin + 10.0
            
        plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='none')
        plt.title(f"Image {self.image_id:07d} (RA:{self.target_ra:.2f}, DEC:{self.target_dec:.2f} | Roll: {self.cfg['roll']:.1f}deg)", color='white')
        plt.axis('off')
        plt.savefig(png_path, bbox_inches='tight', facecolor='black')
        plt.close(fig)

    def run_pipeline(self):
        start = time.time()
        
        # 1. Geometry and math
        self.setup_optics_and_wcs()
        
        # 2. Draw perfect stars and save coordinates
        self.draw_stars()
        
        # 3. Apply static and background glow
        self.apply_sensor_noise()
        
        # 4. Measure the global background statistics
        self.bg_stats =self.measure_global_background()

        # 5. Measure the SNR using the noisy pixels
        self.extract_final_labels()
        
        # 6. Save to disk
        self.export_files()
        
        median_snr = round(np.median(self.star_snr_list), 2) if self.star_snr_list else 0.0
        
        return {
            'image_id': f'{self.image_id:07d}', 
            'ra': self.target_ra, 
            'dec': self.target_dec,
            'roll': self.cfg['roll'], 
            'fov_x': round(self.pixel_scale * self.cfg['image_size_x'] / 3600.0, 2),
            'fov_y': round(self.pixel_scale * self.cfg['image_size_y'] / 3600.0, 2),
            'stars_drawn': self.stars_drawn, 
            'median_snr': median_snr,
            'bg_mean_e': self.bg_stats['mean_e'],
            'bg_std_e': self.bg_stats['std_e'],
            'time_s': round(time.time() - start, 2)
        }