# --- 1. The Worker Function (Independent Telescope Node) ---
def generate_single_image(args):
    """Each core downloads its own data and generates an image from start to finish."""
    image_id, target_ra, target_dec, camera_roll_degrees, image_size_x, image_size_y, pixel_size_um, focal_length_mm, exposure_time, master_table, fits_dir, png_dir, csv_dir = args
    
    process_start = time.time()
    
    # 2. Calculate the true optical pixel scale (arcseconds per pixel)
    pixel_scale = 206.264806247096355 * (pixel_size_um / focal_length_mm)
    fov_x = pixel_scale * image_size_x / 3600.0 # Convert arcseconds to degrees
    fov_y = pixel_scale * image_size_y / 3600.0 # Convert arcseconds to degrees
    
    fits_filename = os.path.join(fits_dir, f'{image_id:07d}.fits')
    png_filename = os.path.join(png_dir, f'{image_id:07d}.png')
    csv_filename = os.path.join(csv_dir, f'{image_id:07d}.csv')
    
    image = galsim.ImageF(image_size_x, image_size_y)
    
    # --- CAMERA ROLL MATH ---
    theta = np.radians(camera_roll_degrees)
    
    dudx = pixel_scale * np.cos(theta)
    dudy = -pixel_scale * np.sin(theta)
    dvdx = pixel_scale * np.sin(theta)
    dvdy = pixel_scale * np.cos(theta)
    
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=image.true_center)

    world_origin = galsim.CelestialCoord(target_ra * galsim.degrees, target_dec * galsim.degrees)
    
    wcs = galsim.TanWCS(affine, world_origin=world_origin, units=galsim.arcsec)
    image.wcs = wcs
    
    # --- C. Draw Stars ---
    universal_baseline = 1e6 # 1 million photons per second for a 0 mag star
    aperature_area_cm2 = np.pi * (6.5/2)**2 # cm^2 for a 65mm diameter lens
    quantum_efficiency = 0.91*0.9 # 91% for ASI585MM sensor, 90% for APO glass transmission
    F0 = universal_baseline * aperature_area_cm2 * exposure_time * quantum_efficiency # Approximately Aperature Area (33.18cm^2) * Exposure Time (30s) * Quantum Efficiency (10,000photons/s)
    label_data = []
    stars_drawn = 0
    star_snr_list = [] # [NEW] Keep track of SNRs for the image median

    # 1. Global Focus: The lens has one focus position for the entire image
    global_defocus = random.uniform(-0.04, 0.04)

    # 2. Find the optical center of your sensor
    center_x = image_size_x / 2.0
    center_y = image_size_y / 2.0
    max_radius = np.sqrt(center_x**2 + center_y**2) # Distance from center to the extreme corner

    # Assuming a 5x5 pixel bounding box, and a 100 pixel background ring
    # box_width = 5.0
    # box_height = 5.0
    # n_px = box_width * box_height
    # margin = 2.0 #two pixel around the bounding box
    # total_width = box_width + (2 * margin)
    # total_height = box_height + (2 * margin)
    # n_b = (total_width * total_height) - n_px

    # Assuming 2.5 pixel radius for bounding circle
    star_radius = 2.5 # pixels
    n_px = np.pi * (star_radius ** 2)
    bg_inner_radius = 4.0 
    bg_outer_radius = 7.0 
    n_b = (np.pi * (bg_outer_radius ** 2)) - (np.pi * (bg_inner_radius ** 2))

    bg_penalty = 1.0 + (n_px / n_b)
    N_S = 10.0  # Your sky background
    N_R = 0.7   # Your read noise
    N_D = 0.0   # Dark noise
    quantization_variance = 1.0 / 12.0

    # A = n_px * bg_penalty * (Sky + Dark + Read^2 + Quantization)
    G = 1.0 # Electrons per ADU (Assuming gain of 1 for simplicity)
    A_term = n_px * bg_penalty * (N_S + N_D + (N_R**2) + quantization_variance*(G**2))
    
    for i, row in master_table.iterrows():
        real_star_id = int(row['source_id'])
        ra_val = row['RA_ICRS'] * galsim.degrees
        dec_val = row['DE_ICRS'] * galsim.degrees
        mag = row['Gmag']
        
        if pd.isna(mag): continue
            
        flux = F0 * 10 ** ((-mag) / 2.5)
        # star = galsim.Gaussian(flux=flux, sigma=1.8) # Simple Gaussian PSF for testing
        
        world_pos = galsim.CelestialCoord(ra_val, dec_val)
        pixel_pos = wcs.toImage(world_pos)
        
        if image.bounds.includes(pixel_pos):
            # star.drawImage(image=image, center=pixel_pos, add_to_image=True, method='real_space')
            # label_data.append([real_star_id, round(pixel_pos.x, 2), round(pixel_pos.y, 2), round(flux, 2), mag])
            # stars_drawn += 1

            # --- Spatial Variance Math ---
            # 1. How far is this specific star from the center of the lens?
            dx = pixel_pos.x - center_x
            dy = pixel_pos.y - center_y
            r = np.sqrt(dx**2 + dy**2)
            
            # 2. Normalize the distance (0.0 is dead center, 1.0 is the extreme corner)
            r_norm = r / max_radius
            
            # 3. Scale the aberrations. We square r_norm so it degrades faster at the edges!
            # The FF65 APO is excellent, so max corner aberration is kept small (0.04)
            edge_coma = 0.04 * (r_norm ** 2)
            edge_astig = 0.03 * (r_norm ** 2)
            
            # 4. Calculate the angle so the "comet tail" points radially outward from the center
            angle = np.arctan2(dy, dx)
            
            # Force GalSim to draw much larger bounding boxes for bright stars
            # Default folding_threshold is 1e-3. We lower it to 1e-5.
            high_accuracy_params = galsim.GSParams(
                folding_threshold=1e-3,
                maximum_fft_size=32768 # Prevents memory errors when the box gets really big
            )
            # 5. Build the physically accurate PSF for this exact pixel location
            optical_psf = galsim.OpticalPSF(
                lam=500.0,              #nm Wavelength of light (green)  
                diam=0.065,             #meter 65mm aperature        
                defocus=global_defocus, # Focus is the same everywhere
                spher=0.01,             # Spherical aberration is inherent to the glass
                astig1=edge_astig * np.cos(2*angle),
                astig2=edge_astig * np.sin(2*angle),
                coma1=edge_coma * np.cos(angle),
                coma2=edge_coma * np.sin(angle),
                gsparams=high_accuracy_params
            )
            
            star = optical_psf.withFlux(flux)
            # star = galsim.Gaussian(flux=flux, sigma=0.85)
            star.drawImage(image=image, center=pixel_pos, add_to_image=True, method='phot')
            star_snr = flux / np.sqrt(flux + A_term)
            star_snr_list.append(star_snr)
                        
            label_data.append([\
                round(real_star_id),
                round(pixel_pos.x, 2), 
                round(pixel_pos.y, 2), 
                round(mag, 3), 
                focal_length_mm, 
                exposure_time,
                round(star_snr, 2)
            ])            
            stars_drawn += 1

    # --- D. Sensor Noise ---
    image += 10.0 # background level
    rng = galsim.BaseDeviate(image_id)
    
    # # 1. Physics of Light (Shot Noise based on background level)
    poisson_noise = galsim.PoissonNoise(rng)
    image.addNoise(poisson_noise)
    
    # # 2. Camera Electronics (Read Noise of the ASI585MM)
    read_noise = galsim.GaussianNoise(rng, sigma=0.7)
    image.addNoise(read_noise)

    # # 3. Analog-to-Digital Conversion
    # # The ASI585 uses a 12-bit ADC. We quantize the continuous electron 
    # # decimals into discrete integer ADU steps, capping at absolute white (4095).
    image.quantize()
    image.array[image.array > 4095] = 4095
    
    # --- E. Export Files ---
    image.write(fits_filename)
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['star_id', 'x_image', 'y_image', 'flux_mag', 'focal_length', 'exposure_time', 'snr'])
        writer.writerows(label_data)
        
    # --- PNG VISIBILITY FIX ---
    fig = plt.figure(dpi=300, figsize=(16, 9), facecolor='black')
    img_array = image.array
    
    dynamic_vmin = max(1.0, np.percentile(img_array, 1))
    dynamic_vmax = np.max(img_array)
    if dynamic_vmax <= dynamic_vmin:
        dynamic_vmax = dynamic_vmin + 10.0
        
    plt.imshow(img_array, cmap='gray', origin='lower', norm=LogNorm(vmin=dynamic_vmin, vmax=dynamic_vmax), interpolation='none')
    plt.title(f'Image {image_id:07d} (RA:{target_ra:.2f}, DEC:{target_dec:.2f} | Roll: {camera_roll_degrees:.1f}deg)', color='white')
    plt.axis('off')
    plt.savefig(png_filename, bbox_inches='tight', facecolor='black')
    plt.close(fig)

    image_median_snr = round(np.median(star_snr_list), 2) if star_snr_list else 0.0
    
    process_duration = time.time() - process_start
    return {
        'image_id': f'{image_id:07d}',
        'ra': target_ra,
        'dec': target_dec,
        'roll': camera_roll_degrees,
        'fov_x': round(fov_x, 2),
        'fov_y': round(fov_y, 2),
        'stars_drawn': stars_drawn,
        'median_snr': image_median_snr,
        'time_s': round(process_duration, 2)
    }