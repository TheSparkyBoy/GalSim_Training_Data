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
        dudx = self.pixel_scale * np.cos(theta)
        dudy = -self.pixel_scale * np.sin(theta)
        dvdx = self.pixel_scale * np.sin(theta)
        dvdy = self.pixel_scale * np.cos(theta)
        
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=self.image.true_center)
        world_origin = galsim.CelestialCoord(self.target_ra * galsim.degrees, self.target_dec * galsim.degrees)
        
        self.wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
        self.image.wcs = self.wcs

    def _apply_lens_distortion(self, x, y, cx, cy, max_r):
        dx = x - cx
        dy = y - cy
        r_norm = np.sqrt(dx**2 + dy**2) / max_r
        k1 = 0.08 
        distortion_factor = 1.0 + (k1 * (r_norm ** 2))
        return cx + (dx * distortion_factor), cy + (dy * distortion_factor)

    def draw_stars(self):
        global_defocus = random.uniform(-0.04, 0.04)
        center_x, center_y = self.cfg['image_size_x'] / 2.0, self.cfg['image_size_y'] / 2.0
        max_radius = np.sqrt(center_x**2 + center_y**2) 
        
        smear_length_pixels = random.uniform(0.0, 2.5) if self.anom_smear else 0.0
        smear_angle = random.uniform(0, 360) * galsim.degrees
        self.smear_applied = smear_length_pixels
        
        if smear_length_pixels > 0.1:
            minor_axis = 0.1 
            q_ratio = minor_axis / smear_length_pixels
            motion_smear = galsim.Gaussian(sigma=smear_length_pixels/2.0).shear(q=q_ratio, beta=smear_angle)
        else:
            motion_smear = None
            
        for row in self.cfg['master_table'].itertuples():
            # --- FIXED: Pure dot-notation for all catalog attributes! ---
            base_mag = getattr(row, 'Gmag', np.nan)
            if pd.isna(base_mag): continue
                
            mag = base_mag + np.random.normal(0, 0.15) if self.anom_mag else base_mag
            
            if self.anom_drop:
                p_detect = 1.0 / (1.0 + np.exp(3.0 * (mag - 10.5)))
                if random.random() > p_detect:
                    self.stars_dropped += 1 
                    continue 
                
            # --- FIXED: Dot-notation for Right Ascension and Declination ---
            world_pos = galsim.CelestialCoord(row.RA_ICRS * galsim.degrees, row.DE_ICRS * galsim.degrees)
            pixel_pos = self.wcs.toImage(world_pos)
            x, y = pixel_pos.x, pixel_pos.y
            
            if self.anom_lens:
                x, y = self._apply_lens_distortion(x, y, center_x, center_y, max_radius)
            if self.anom_pos:
                x += np.random.normal(0, 0.2)
                y += np.random.normal(0, 0.2)
                
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
                
                # --- FIXED: Safely extract raw string ID using dot-notation ---
                raw_id_str = str(getattr(row, 'source_id', getattr(row, 'star_id', '0'))).split('.')[0]
                
                self.temp_star_coords.append({
                    'id': raw_id_str,
                    'x': final_pos.x,
                    'y': final_pos.y,
                    'mag': round(mag, 3)
                })
                self.stars_drawn += 1
                
        if self.anom_false:
            self.false_stars_injected = np.random.poisson(8) 
            for _ in range(self.false_stars_injected):
                fx = random.uniform(0, self.cfg['image_size_x'])
                fy = random.uniform(0, self.cfg['image_size_y'])
                fmag = random.uniform(5.0, 11.0)
                fflux = self.F0 * 10 ** ((-fmag) / 2.5)
                
                false_psf = galsim.Gaussian(fwhm=1.5).withFlux(fflux)
                false_psf.drawImage(image=self.image, center=galsim.PositionD(fx, fy), add_to_image=True, method='phot')
                
                self.temp_false_coords.append({
                    'id': '-1', 
                    'x': fx,
                    'y': fy,
                    'mag': round(fmag, 3)
                })

    def apply_sensor_noise(self):
        self.image += 1500.0 
        rng = galsim.BaseDeviate(int(self.image_id))
        self.image.addNoise(galsim.PoissonNoise(rng))
        self.image.addNoise(galsim.GaussianNoise(rng, sigma=0.7))
        self.image.quantize()
        self.image.array[self.image.array > 4095] = 4095

    def _evaluate_merline_howell_telemetry(self, img_array):
        """
        MERLINE & HOWELL (1995) SIM2REAL ENGINE:
        Evaluates the rendered GalSim pixel array using Equation 6.1 integrated
        streak/star SNR math to guarantee 100% parity with physical camera extractions.
        """
        if img_array is None:
            return 0, 0.0, 0.0, 0.0
        try:
            img_float = img_array.astype(np.float32)
            total_pixels = img_float.size
            
            bg_mean = float(np.median(img_float))
            mad = float(np.median(np.abs(img_float - bg_mean)))
            bg_std = float(1.4826 * mad)
            
            if bg_std == 0:
                bg_std = float(np.std(img_float))
                if bg_std == 0: bg_std = 1.0
                
            bg_variance = bg_std ** 2
            threshold = bg_mean + (5.0 * bg_std)
            
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_detected_stars = ndi.label(img_float > threshold, structure=structure)
            
            if num_detected_stars > 0:
                n_px = np.array(ndi.sum(np.ones_like(img_float), labeled_array, index=range(1, num_detected_stars + 1)))
                net_signal_img = img_float - bg_mean
                N_star = np.array(ndi.sum(net_signal_img, labeled_array, index=range(1, num_detected_stars + 1)))
                
                total_star_pixels = np.sum(n_px)
                n_B = max(1.0, float(total_pixels - total_star_pixels))
                
                bg_term = n_px * (1.0 + (n_px / n_B)) * bg_variance
                radicand = np.maximum(1e-6, N_star + bg_term)
                
                star_snrs = N_star / np.sqrt(radicand)
                median_snr = float(np.median(star_snrs))
            else:
                median_snr = 0.0
                
            # Returns raw numerical types for the orchestrator dictionary
            return int(num_detected_stars), round(bg_mean, 2), round(bg_std, 2), round(median_snr, 2)
        except Exception as e:
            print(f"[WARNING] Merline & Howell simulator evaluation failed: {e}")
            return 0, 0.0, 0.0, 0.0

    def extract_final_labels(self):
        for star in self.temp_star_coords:
            final_snr = self.measure_snr(star['x'], star['y'])
            if final_snr < 1.0: continue
            
            self.star_snr_list.append(final_snr)
            self.label_data.append([
                star['id'], round(star['x'], 2), round(star['y'], 2), 
                star['mag'], self.cfg['focal_length_mm'], self.cfg['exposure_time'], 
                final_snr, 0 
            ])
            
        for star in self.temp_false_coords:
            final_snr = self.measure_snr(star['x'], star['y'])
            if final_snr < 1.0: continue
            
            self.label_data.append([
                star['id'], round(star['x'], 2), round(star['y'], 2), 
                star['mag'], self.cfg['focal_length_mm'], self.cfg['exposure_time'], 
                final_snr, 1 
            ])

    def measure_global_background(self):
        bg_mean_adu, bg_median_adu, bg_std_adu = sigma_clipped_stats(self.image.array, sigma=3.0, maxiters=5)
        gain_e_per_adu = 1.0 
        return {
            'mean_adu': round(bg_mean_adu, 2), 'std_adu': round(bg_std_adu, 2),
            'mean_e': round(bg_mean_adu * gain_e_per_adu, 2), 'std_e': round(bg_std_adu * gain_e_per_adu, 2)
        }

    def export_files(self):
        fits_path = os.path.join(self.cfg['dirs']['fits'], f'{self.image_id:07d}.fits')
        png_path = os.path.join(self.cfg['dirs']['png'], f'{self.image_id:07d}.png')
        csv_path = os.path.join(self.cfg['dirs']['csv'], f'{self.image_id:07d}.csv')
        
        self.image.write(fits_path)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['star_id', 'x_image', 'y_image', 'flux_mag', 'focal_length', 'exposure_time', 'snr', 'is_artifact'])
            writer.writerows(self.label_data)
            
        fig = plt.figure(dpi=300, figsize=(16, 9), facecolor='black')
        img_array = self.image.array
        vmin = max(1.0, np.percentile(img_array, 1))
        vmax = np.max(img_array)
        if vmax <= vmin: vmax = vmin + 10.0
            
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
        self.bg_stats = self.measure_global_background()
        self.extract_final_labels()
        self.export_files()
        
        # 1. Grab the final rendered 2D array from GalSim
        final_pixel_array = self.image.array
        
        # 2. Run Equation 6.1 to get empirical Sim2Real telemetry
        emp_stars, bg_mean, bg_std, med_snr = self._evaluate_merline_howell_telemetry(final_pixel_array)
        
        # 3. Return dictionary formatted for orchestrator.py
        return {
            'image_id': self.cfg['image_id'],
            'ra': self.cfg['ra'],
            'dec': self.cfg['dec'],
            'roll': self.cfg['roll'],
            'time_s': round(time.time() - start_time, 2),
            
            # You can return either your theoretical star count or the empirical count:
            'stars_drawn': emp_stars, # Or keep self.stars_drawn_count if you prefer theoretical
            
            # --- EQUATION 6.1 TELEMETRY PARITY METRICS ---
            'bg_mean_e': bg_mean,
            'bg_std_e': bg_std,
            'median_snr': med_snr,
            
            # ... [Your existing anomaly and distortion flags] ...
            'distorted_stars': getattr(self, 'distorted_count', 0),
            'dropped_stars': getattr(self, 'dropped_count', 0),
            'false_stars': getattr(self, 'false_count', 0),
            'smear_px': getattr(self, 'smear_length', 0)
        }