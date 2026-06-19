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
        global_defocus = random.uniform(-0.04, 0.04)
        center_x, center_y = self.cfg['image_size_x'] / 2.0, self.cfg['image_size_y'] / 2.0
        max_radius = np.sqrt(center_x**2 + center_y**2) 

        # Noise math for SNR mapping
        star_radius = 2.5 
        n_px = np.pi * (star_radius ** 2)
        n_b = (np.pi * (7.0 ** 2)) - (np.pi * (4.0 ** 2))
        bg_penalty = 1.0 + (n_px / n_b)
        
        N_S, N_R, N_D, G = 10.0, 0.7, 0.0, 1.0
        quantization_variance = 1.0 / 12.0
        A_term = n_px * bg_penalty * (N_S + N_D + (N_R**2) + quantization_variance * (G**2))
        
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
                
                star_snr = flux / np.sqrt(flux + A_term)
                self.star_snr_list.append(star_snr)
                
                self.label_data.append([
                    int(row['source_id']), round(pixel_pos.x, 2), round(pixel_pos.y, 2), 
                    round(mag, 3), self.cfg['focal_length_mm'], self.cfg['exposure_time'], round(star_snr, 2)
                ])            
                self.stars_drawn += 1

    def apply_sensor_noise(self):
        self.image += 10.0 # background level
        rng = galsim.BaseDeviate(int(self.image_id))
        
        self.image.addNoise(galsim.PoissonNoise(rng))
        self.image.addNoise(galsim.GaussianNoise(rng, sigma=0.7))
        self.image.quantize()
        self.image.array[self.image.array > 4095] = 4095

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
        self.setup_optics_and_wcs()
        self.draw_stars()
        self.apply_sensor_noise()
        self.export_files()
        
        median_snr = round(np.median(self.star_snr_list), 2) if self.star_snr_list else 0.0
        
        return {
            'image_id': f'{self.image_id:07d}', 'ra': self.target_ra, 'dec': self.target_dec,
            'roll': self.cfg['roll'], 'fov_x': round(self.pixel_scale * self.cfg['image_size_x'] / 3600.0, 2),
            'fov_y': round(self.pixel_scale * self.cfg['image_size_y'] / 3600.0, 2),
            'stars_drawn': self.stars_drawn, 'median_snr': median_snr,
            'time_s': round(time.time() - start, 2)
        }