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
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as u

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

        # --- NINA GAIN CONVERSION MATH ---
        # 1 NINA gain unit = 0.1 dB. The linear multiplier is 10^(dB/20), which simplifies to 10^(nina_gain/200)
        self.nina_gain = self.cfg.get('nina_gain', 0)
        self.native_gain = self.cfg.get('native_gain_e_adu', 9.77)
        self.amp_multiplier = 10 ** (self.nina_gain / 200.0)
        self.system_gain = self.native_gain / self.amp_multiplier
        
        # --- GRANULAR ANOMALY TOGGLES ---
        self.anom_lens = self.cfg.get('anom_lens_distortion', False)
        self.anom_false = self.cfg.get('anom_false_stars', False)
        self.anom_drop = self.cfg.get('anom_drop_stars', False)
        self.anom_pos = self.cfg.get('anom_pos_variation', False)
        self.anom_mag = self.cfg.get('anom_mag_variation', False)
        self.anom_smear = self.cfg.get('anom_motion_smear', False)
        
        # --- NEW ANOMALY TOGGLES ---
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
        dx = x - cx
        dy = y - cy
        r_norm = np.sqrt(dx**2 + dy**2) / max_r
        
        # APO flat-field astrograph distortion coefficient
        k1 = 1e-6 
        distortion_factor = 1.0 + (k1 * (r_norm ** 2))
        return cx + (dx * distortion_factor), cy + (dy * distortion_factor)

    def draw_stars(self):
        # --- DYNAMIC ZENITH ANGLE CALCULATION ---
        obs_lat = self.cfg.get('obs_lat', 33.3062) 
        obs_lon = self.cfg.get('obs_lon', -111.8413) 
        obs_alt = self.cfg.get('obs_alt', 370.0) 
        obs_time_str = self.cfg.get('obs_time_utc', '2026-08-15T08:00:00')
        
        location = EarthLocation(lat=obs_lat*u.deg, lon=obs_lon*u.deg, height=obs_alt*u.m)
        obstime = Time(obs_time_str)
        boresight_coord = SkyCoord(ra=self.target_ra*u.deg, dec=self.target_dec*u.deg, frame='icrs')
        
        # Transform image center to local Altitude/Azimuth
        altaz = boresight_coord.transform_to(AltAz(obstime=obstime, location=location))
        zenith_angle_rad = (90.0 * u.deg - altaz.alt).to(u.rad).value
        
        # Cap Zenith Angle at 85 degrees to prevent Secant from shooting to infinity near the horizon
        if zenith_angle_rad >= (85.0 * np.pi / 180.0):
            dynamic_sec_zeta = 11.47 
        else:
            dynamic_sec_zeta = 1.0 / np.cos(zenith_angle_rad)

        # --- DEFOCUS ANOMALY (Thermal Expansion OPD) ---
        if self.anom_defocus:
            L = self.cfg['focal_length_mm'] * 1e-3         
            alpha = 23e-6                                  
            delta_T = self.cfg.get('delta_T_celsius', 2.0) 
            aperture_mm = self.cfg.get('aperture_mm', 65.0)                             
            f_ratio = self.cfg['focal_length_mm'] / aperture_mm 
            lam = 500e-9                                   
            
            OPD = (L * alpha * delta_T) / (8.0 * (f_ratio ** 2))
            global_defocus = (OPD / lam) * random.choice([-1, 1])
        else:
            global_defocus = random.uniform(-0.04, 0.04)
            
        # --- DYNAMIC FOV OPTIMIZATION ---
        center_x, center_y = self.cfg['image_size_x'] / 2.0, self.cfg['image_size_y'] / 2.0
        max_radius = np.sqrt(center_x**2 + center_y**2) 
        max_fov_degrees = (max_radius * self.pixel_scale) / 3600.0
        safe_rejection_radius = (max_fov_degrees + 1.0) * galsim.degrees
        
        boresight_gs = galsim.CelestialCoord(self.target_ra * galsim.degrees, self.target_dec * galsim.degrees)
        
        # --- EXPOSURE-DEPENDENT SMEAR ANOMALY (Tracking Error) ---
        if self.anom_smear:
            PE_rate = 0.2                                  
            t_exp = self.cfg['exposure_time']              
            F = self.cfg['focal_length_mm'] * 1e-3         
            d_pixel = self.cfg['pixel_size_um'] * 1e-6     
            arcsec_conversion = 206265.0                   
            
            smear_length_pixels = (PE_rate * t_exp * F) / (arcsec_conversion * d_pixel)
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
                
            # --- ATMOSPHERIC SCINTILLATION (Magnitude Variation) ---
            if self.anom_mag:
                D = self.cfg.get('aperture_mm', 65.0) / 10.0   # Convert mm to cm                           
                h = obs_alt                                    # Dynamic Observatory altitude (m)
                H_scale = 8000.0                               
                
                # Apply the dynamically calculated secant of the zenith angle
                variance = 0.09 * (D ** (-7.0/3.0)) * (dynamic_sec_zeta ** 3) * np.exp(-2.0 * h / H_scale)
                sigma_I = np.sqrt(variance)
                
                sigma_mag = 1.0857 * sigma_I
                mag = base_mag + np.random.normal(0, sigma_mag)
            else:
                mag = base_mag
            
            # --- DETECTION FLOOR (Faint Object Masking) ---
            if self.anom_drop:
                textbook_limit = 10.5
                deep_limit = textbook_limit + 1.8
                
                p_detect = 1.0 / (1.0 + np.exp(3.0 * (mag - deep_limit)))
                if random.random() > p_detect:
                    self.stars_dropped += 1 
                    continue 
                
            world_pos = galsim.CelestialCoord(row.RA_ICRS * galsim.degrees, row.DE_ICRS * galsim.degrees)
            
            if boresight_gs.distanceTo(world_pos) > safe_rejection_radius:
                continue
                
            pixel_pos = self.wcs.toImage(world_pos)
            x, y = pixel_pos.x, pixel_pos.y
            
            if self.anom_lens:
                x, y = self._apply_lens_distortion(x, y, center_x, center_y, max_radius)
                
            # --- ASTROMETRIC JITTER (Seeing Position Variation) ---
            if self.anom_pos:
                lam = 500e-9                               
                F = self.cfg['focal_length_mm'] * 1e-3     
                r_0 = 0.1                                  
                fwhm_factor = 2.355                        
                d_pixel = self.cfg['pixel_size_um'] * 1e-6 
                
                sigma_px = (0.98 * lam * F) / (r_0 * fwhm_factor * d_pixel)
                
                x += np.random.normal(0, sigma_px)
                y += np.random.normal(0, sigma_px)
                
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
        # 1. Define Input Variables
        full_well_e = self.cfg.get('full_well_capacity_e', 40000.0)      # Physical well limit
        bias_pedestal = self.cfg.get('bias_pedestal', 500.0)             # Electronic offset (ADU)
        dark_current_rate = self.cfg.get('dark_current_rate', 0.002)     # Thermal leak (e-/pixel/s)
        sky_background_rate = self.cfg.get('sky_background_rate', 25.0)  # Light pollution (e-/pixel/s)
        t_exp = self.cfg['exposure_time']                                # Exposure time (s)
        
        # --- ANALOG STAGE (Measurements in Electrons) ---
        
        # 2. Add thermal & sky background electrons (Signal electrons were already added by draw_stars)
        background_electrons = (dark_current_rate + sky_background_rate) * t_exp
        self.image += background_electrons 
        
        # 3. Apply Quantum (Poisson) and Amplifier (Gaussian) Noise
        rng = galsim.BaseDeviate(int(self.image_id))
        self.image.addNoise(galsim.PoissonNoise(rng))
        
        read_noise_e = self.cfg.get('read_noise', 0.7)
        self.image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise_e))
        
        # 4. Apply Physical Full Well Capacity Limit
        # Any pixel with more than 40,000 electrons physically bleeds/clips before hitting the ADC
        self.image.array[self.image.array > full_well_e] = full_well_e
        
        # --- DIGITAL STAGE (Conversion to ADU) ---
        
        # 5. Convert Electrons to ADU using System Gain
        self.image /= system_gain
        
        # 6. Add the electronic Bias Pedestal (Applied in ADU by the hardware)
        self.image += bias_pedestal
        
        # 7. Apply Defective Sensor Anomalies (In ADU)
        if self.anom_hot_pix:
            N_bits = 12                                    
            ADU_max = (2 ** N_bits) - 1
            num_hot = np.random.poisson(200) 
            hx = np.random.randint(0, self.cfg['image_size_x'], num_hot)
            hy = np.random.randint(0, self.cfg['image_size_y'], num_hot)
            self.image.array[hy, hx] = float(ADU_max)
            
        if self.anom_dead_pix:
            num_dead = np.random.poisson(200) 
            dx = np.random.randint(0, self.cfg['image_size_x'], num_dead)
            dy = np.random.randint(0, self.cfg['image_size_y'], num_dead)
            self.image.array[dy, dx] = 0.0
            
        # 8. ADC Quantization & Hard-Clipping to 12-bit limit
        self.image.quantize()
        self.image.array[self.image.array > 4095.0] = 4095.0
        
        # Failsafe: Ensure no mathematical negative values exist after noise fluctuation
        self.image.array[self.image.array < 0.0] = 0.0

    def measure_global_background(self):
        # Extract background statistics from the final ADU image
        bg_mean_adu, bg_median_adu, bg_std_adu = sigma_clipped_stats(self.image.array, sigma=3.0, maxiters=5)
        
        # Fetch the dynamic gain from config to calculate the true electron background
        gain_e_per_adu = self.cfg.get('system_gain', 1.0) 
        
        return {
            'mean_adu': round(bg_mean_adu, 2), 'std_adu': round(bg_std_adu, 2),
            'mean_e': round(bg_mean_adu * gain_e_per_adu, 2), 'std_e': round(bg_std_adu * gain_e_per_adu, 2)
        }

    def measure_snr(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        
        if not (0 <= ix < self.cfg['image_size_x'] and 0 <= iy < self.cfg['image_size_y']):
            return 0.0
            
        x_min = max(0, ix - 1)
        x_max = min(self.cfg['image_size_x'], ix + 2)
        y_min = max(0, iy - 1)
        y_max = min(self.cfg['image_size_y'], iy + 2)
        
        # 1. Extract the ADU cutout
        cutout_adu = self.image.array[y_min:y_max, x_min:x_max]
        n_pix = cutout_adu.size
        
        # 2. Convert ADU back to Electrons for accurate Poisson math
        gain = self.cfg.get('system_gain', 1.0)
        cutout_e = cutout_adu * gain
        
        # 3. Use the electron-based background statistics
        bg_mean_e = self.bg_stats['mean_e']
        bg_std_e = max(1e-5, self.bg_stats['std_e'])
        
        # 4. Calculate Net Signal and Noise in Electrons
        net_signal_e = float(np.sum(cutout_e - bg_mean_e))
        if net_signal_e <= 0:
            return 0.0
            
        noise_e = float(np.sqrt(net_signal_e + (n_pix * (bg_std_e ** 2))))
        
        return float(round(net_signal_e / noise_e, 2))

    def _evaluate_merline_howell_telemetry(self, img_array):
        if img_array is None:
            return 0, 0.0, 0.0, 0.0
        try:
            # --- CONVERT ENTIRE IMAGE BACK TO ELECTRONS FOR POISSON MATH ---
            gain = self.cfg.get('system_gain', 1.0)
            img_float_e = img_array.astype(np.float32) * gain
            total_pixels = img_float_e.size
            
            bg_mean_e = float(np.median(img_float_e))
            mad_e = float(np.median(np.abs(img_float_e - bg_mean_e)))
            bg_std_e = float(1.4826 * mad_e)
            
            if bg_std_e == 0:
                bg_std_e = float(np.std(img_float_e))
                if bg_std_e == 0: bg_std_e = 1.0
                
            bg_variance_e = bg_std_e ** 2
            threshold_e = bg_mean_e + (5.0 * bg_std_e)
            
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_detected_stars = ndi.label(img_float_e > threshold_e, structure=structure)
            
            if num_detected_stars > 0:
                n_px = np.array(ndi.sum(np.ones_like(img_float_e), labeled_array, index=range(1, num_detected_stars + 1)))
                net_signal_img_e = img_float_e - bg_mean_e
                N_star_e = np.array(ndi.sum(net_signal_img_e, labeled_array, index=range(1, num_detected_stars + 1)))
                
                total_star_pixels = np.sum(n_px)
                n_B = max(1.0, float(total_pixels - total_star_pixels))
                
                # Merline & Howell equation (Calculated entirely in electrons)
                bg_term_e = n_px * (1.0 + (n_px / n_B)) * bg_variance_e
                radicand_e = np.maximum(1e-6, N_star_e + bg_term_e)
                
                star_snrs = N_star_e / np.sqrt(radicand_e)
                median_snr = float(np.median(star_snrs))
            else:
                median_snr = 0.0
                
            return int(num_detected_stars), round(bg_mean_e, 2), round(bg_std_e, 2), round(median_snr, 2)
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
            
        # --- COSMIC RAYS / DEBRIS (False Stars) ---
        if self.anom_false:
            Phi = 1.5                                      
            W = self.cfg['image_size_x'] * self.cfg['pixel_size_um'] * 1e-4  
            H = self.cfg['image_size_y'] * self.cfg['pixel_size_um'] * 1e-4  
            t_exp_min = self.cfg['exposure_time'] / 60.0   
            
            N_expected = Phi * W * H * t_exp_min
            
            self.false_stars_injected = np.random.poisson(N_expected) 
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
            
        for star in self.temp_false_coords:
            final_snr = self.measure_snr(star['x'], star['y'])
            if final_snr < 1.0: continue
            
            self.label_data.append([
                star['id'], round(star['x'], 2), round(star['y'], 2), 
                star['mag'], self.cfg['focal_length_mm'], self.cfg['exposure_time'], 
                final_snr, 1 
            ])

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
        start_time = time.time()
        self.setup_optics_and_wcs()
        self.draw_stars()
        self.apply_sensor_noise()
        self.bg_stats = self.measure_global_background()
        self.extract_final_labels()
        self.export_files()
        
        final_pixel_array = self.image.array
        emp_stars, bg_mean, bg_std, med_snr = self._evaluate_merline_howell_telemetry(final_pixel_array)
        fov_x_deg = (self.cfg['image_size_x'] * self.pixel_scale) / 3600.0
        fov_y_deg = (self.cfg['image_size_y'] * self.pixel_scale) / 3600.0
        
        return {
            'image_id': self.cfg['image_id'],
            'ra': self.cfg['ra'],
            'dec': self.cfg['dec'],
            'roll': self.cfg['roll'],
            'fov_x_deg': round(fov_x_deg, 3), # Added to output
            'fov_y_deg': round(fov_y_deg, 3), # Added to output
            'time_s': round(time.time() - start_time, 2),
            'stars_drawn': emp_stars,
            'bg_mean_e': bg_mean,
            'bg_std_e': bg_std,
            'median_snr': med_snr,
            'distorted_stars': self.stars_drawn if self.anom_lens else 0, # All stars are distorted if lens anom is on
            'dropped_stars': self.stars_dropped,                          
            'false_stars': self.false_stars_injected,                     
            'smear_px': round(self.smear_applied, 2)                      
        }