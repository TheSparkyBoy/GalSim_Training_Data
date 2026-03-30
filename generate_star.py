import galsim
import random
import csv

# 1. Configuration
image_size = 512
pixel_scale = 0.2
num_stars = 50  

print(f"Generating image with {num_stars} stars...")

# 2. Create the blank canvas
full_image = galsim.ImageF(image_size, image_size)
label_data = []

# 3. The Generation Loop
for i in range(num_stars):
    x_pos = random.uniform(10, image_size - 10)
    y_pos = random.uniform(10, image_size - 10)
    
    flux = random.uniform(5000, 150000)
    sigma = random.uniform(1.2, 2.5) 
    
    star = galsim.Gaussian(flux=flux, sigma=sigma)
    
    pos = galsim.PositionD(x_pos, y_pos)
    star.drawImage(image=full_image, center=pos, scale=pixel_scale, add_to_image=True)
    
    label_data.append([i, round(x_pos, 2), round(y_pos, 2), round(flux, 2)])

# 4. Add the realistic sensor noise
print("Applying thermal sensor noise...")
noise = galsim.GaussianNoise(sigma=12.0)
full_image.addNoise(noise)

# ==========================================
# 5. EXPORT THE FITS FILE (Changed from PNG)
# ==========================================
# GalSim automatically detects the .fits extension and saves the raw 32-bit math.
# You can use .fit or .fits, both are standard.
fits_filename = 'starfield_001.fits'
full_image.write(fits_filename)
print(f"Scientific image saved: {fits_filename}")

# 6. Export the Labels
csv_filename = 'starfield_001_labels.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['star_id', 'x_pixel', 'y_pixel', 'flux'])
    writer.writerows(label_data)
    
print(f"Labels saved: {csv_filename}")
