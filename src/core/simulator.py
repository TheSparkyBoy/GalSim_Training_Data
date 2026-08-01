# core/simulator.py
import galsim
import os
import numpy as np
import scipy.ndimage as ndi
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
        
        self.worker_seed = self.cfg['global_seed'] + int(self.image_id)
        random.seed(self.worker_seed)
        np.random.seed(self.worker_seed)
        
        self.image = galsim.ImageF(self.cfg['image_size_x'], self.cfg['image_size_y'])
        self.wcs = None
        self.pixel_scale = None
        self.stars_drawn = 0
        
        # --- GRANULAR ANOMALY TOGGLES ---
        self.anom_lens = self.cfg.get('anom_lens_distortion', False)
        self.anom_false = self.cfg.get('anom_false_stars', False)
        self.anom_drop = self.cfg.get('anom_drop_stars', False)
        self.anom_pos = self.cfg.get('anom_pos_variation', False)
        self.anom_mag = self.cfg.get('anom_mag_variation', False)
        self.anom_smear = self.cfg.get('anom_motion_smear', False)
        self.anom_defocus = self.cfg.get('anom_defocus', False)
        self.anom_dead_pix = self.cfg.get('anom_dead_pixels', False)
        self.anom_hot_pix = self.cfg.get('anom_hot_pixels', False)
        
        # Anomaly tracking variables
        self.stars_dropped = 0
        self.false_stars_injected = 0
        self.smear_applied = 0.0
        self.temp_false_coords = []
        
        self.temp_star_coords = [] 
        self.label_data = []
        self.star_snr_list = []
        
        self.universal_baseline = 1e6 
        self.aperature_area_cm2 = np.pi * (6.5/2)**2 
        self.quantum_efficiency = 0.91 * 0.9 
        self.F0 = self.universal_baseline * self.aperature_area_cm2 * self.cfg['exposure_time'] * self.quantum_efficiency
        
    def setup_optics_and_wcs(self):
        self.pixel_scale = 206.264806247096355 * (self.cfg['pixel_size_um'] / self.cfg['focal_length_mm'])
        
        theta = np.radians(self.cfg['roll'])
        dudx = -self.pixel_scale * np.cos(theta)
        dudy = self.pixel_scale * np.sin(theta)
        dvdx = self.pixel_scale * np.sin(theta)
        dvdy = self.pixel_scale * np.cos(theta)
        
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=self.image.true_center)
        world_origin = galsim.CelestialCoord(self.target_ra * galsim.degrees, self.target_dec * galsim.degrees)
        
        self.wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
        self.image.wcs = self.wcs

    def _apply_lens_distortion(self, x, y, cx, cy, max_r):
        # [Keep existing _apply_lens_distortion method identical]
        dx = x - cx
        dy = y - cy
        r_norm = np.sqrt(dx**2 + dy**2) / max_r
        k1 = 0.08 
        distortion_factor = 1.0 + (k1 * (r_norm ** 2))
        return cx + (dx * distortion_factor), cy + (dy * distortion_factor)

    def draw_stars(self):
        # --- DEFOCUS ANOMALY ---
        if self.anom_defocus:
            global_defocus = random.uniform(0.15, 0.35) * random.choice([-1, 1])
        else:
            global_defocus = random.uniform(-0.04, 0.04)
            
        # 1. OPTIMIZATION: Calculate dynamic FOV radius in degrees based on sensor size
        center_x, center_y = self.cfg['image_size_x'] / 2.0, self.cfg['image_size_y'] / 2.0
        max_pixel_radius = np.sqrt(center_x**2 + center_y**2) 
        max_fov_degrees = (max_pixel_radius * self.pixel_scale) / 3600.0
        safe_rejection_radius = (max_fov_degrees + 1.0) * galsim.degrees
        
        boresight = galsim.CelestialCoord(self.target_ra * galsim.degrees, self.target_dec * galsim.degrees)
        
        # --- EXPOSURE-DEPENDENT SMEAR ANOMALY ---
        if self.anom_smear:
            base_smear = random.uniform(0.1, 0.8)
            smear_length_pixels = base_smear * self.cfg['exposure_time']
        else:
            smear_length_pixels = 0.0
            
        smear_angle = random.uniform(0, 360) * galsim.degrees
        self.smear_applied = smear_length_pixels
        
        if smear_length_pixels > 0.1:
            minor_axis = 0.1 
            q_ratio = minor_axis / smear_length_pixels
            motion_smear = galsim.Gaussian(sigma=smear_length_pixels/2.0).shear(q=q_ratio, beta=smear_angle)
        else:
            motion_smear = None
            
        for row in self.cfg['master_table'].itertuples():
            base_mag = getattr(row, 'Gmag', np.nan)
            if pd.isna(base_mag): continue
                
            # --- UPDATED: Realistic Photometric Scintillation (std dev: 0.25 mag) ---
            mag = base_mag + np.random.normal(0, 0.25) if self.anom_mag else base_mag
            
            if self.anom_drop:
                p_detect = 1.0 / (1.0 + np.exp(3.0 * (mag - 10.5)))
                if random.random() > p_detect:
                    self.stars_dropped += 1 
                    continue 
                
            world_pos = galsim.CelestialCoord(row.RA_ICRS * galsim.degrees, row.DE_ICRS * galsim.degrees)
            
            # 2. OPTIMIZATION: Instantly drop stars outside the specific FOV + 1.0deg buffer
            if boresight.distanceTo(world_pos) > safe_rejection_radius:
                continue
                
            pixel_pos = self.wcs.toImage(world_pos)
            x, y = pixel_pos.x, pixel_pos.y
            
            if self.anom_lens:
                x, y = self._apply_lens_distortion(x, y, center_x, center_y, max_radius)
                
            if self.anom_pos:
                # --- UPDATED: Realistic Astrometric Jitter/Seeing (std dev: 0.6 pixels) ---
                x += np.random.normal(0, 0.6)
                y += np.random.normal(0, 0.6)
                
            final_pos = galsim.PositionD(x, y)
            
            if self.image.bounds.includes(final_pos):
                flux = self.F0 * 10 ** ((-mag) / 2.5)
                dx, dy = final_pos.x - center_x, final_pos.y - center_y
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
                
                if self.anom_smear and motion_smear is not None:
                    final_profile = galsim.Convolve([optical_psf, motion_smear])
                else:
                    final_profile = optical_psf
                    
                star = final_profile.withFlux(flux)
                star.drawImage(image=self.image, center=final_pos, add_to_image=True, method='phot')
                
                raw_id_str = str(getattr(row, 'source_id', getattr(row, 'star_id', '0'))).split('.')[0]
                
                self.temp_star_coords.append({
                    'id': raw_id_str,
                    'x': final_pos.x,
                    'y': final_pos.y,
                    'mag': round(mag, 3)
                })
                self.stars_drawn += 1

    def apply_sensor_noise(self):
        self.image += 1500.0 
        rng = galsim.BaseDeviate(int(self.image_id))
        self.image.addNoise(galsim.PoissonNoise(rng))
        self.image.addNoise(galsim.GaussianNoise(rng, sigma=0.7))
        
        # --- DEFECTIVE SENSOR ANOMALIES ---
        if self.anom_hot_pix:
            num_hot = np.random.poisson(200) # Inject random clusters of max-ADU pixels
            hx = np.random.randint(0, self.cfg['image_size_x'], num_hot)
            hy = np.random.randint(0, self.cfg['image_size_y'], num_hot)
            self.image.array[hy, hx] = 4095.0
            
        if self.anom_dead_pix:
            num_dead = np.random.poisson(200) # Inject random clusters of zero-ADU pixels
            dx = np.random.randint(0, self.cfg['image_size_x'], num_dead)
            dy = np.random.randint(0, self.cfg['image_size_y'], num_dead)
            self.image.array[dy, dx] = 0.0
            
        self.image.quantize()
        self.image.array[self.image.array > 4095] = 4095

    # [Keep the rest of your methods exactly the same (measure_snr, _evaluate_merline_howell_telemetry, extract_final_labels, measure_global_background, export_files, run_pipeline)]
    # ...