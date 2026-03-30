import galsim  # Import the GalSim library for astronomical image simulation
import random  # Import random module for generating random numbers
import csv  # Import csv module for writing CSV files

# 1. Configuration
image_size = 1024  # Set the size of the image to 1024x1024 pixels
pixel_scale = 0.2  # Set the pixel scale (arcseconds per pixel)
num_stars = 50  # Set the number of stars to generate

print(f"Generating image with {num_stars} stars...")  # Print a message indicating the start of generation

# 2. Create the blank canvas
full_image = galsim.ImageF(image_size, image_size)  # Create a blank floating-point image of the specified size
label_data = []  # Initialize an empty list to store label data for each star

# 3. The Generation Loop
for i in range(num_stars):  # Loop over the number of stars to generate
    x_pos = random.uniform(10, image_size - 10)  # Generate a random x-position for the star, avoiding edges
    y_pos = random.uniform(10, image_size - 10)  # Generate a random y-position for the star, avoiding edges
    
    flux = random.uniform(5000, 150000)  # Generate a random flux (brightness) for the star
    sigma = random.uniform(1.2, 2.5)  # Generate a random sigma (width) for the Gaussian star profile
    
    star = galsim.Gaussian(flux=flux, sigma=sigma)  # Create a Gaussian star object with the generated flux and sigma
    
    pos = galsim.PositionD(x_pos, y_pos)  # Create a position object for the star's location
    star.drawImage(image=full_image, center=pos, scale=pixel_scale, add_to_image=True)  # Draw the star onto the image at the specified position, adding it to the existing image
    
    label_data.append([i, round(x_pos, 2), round(y_pos, 2), round(flux, 2)])  # Append the star's ID, position, and flux to the label data list

# 4. Add the realistic sensor noise
print("Applying thermal sensor noise...")  # Print a message indicating noise application
noise = galsim.GaussianNoise(sigma=1.0)  # Create a Gaussian noise object with sigma=1.0 for sensor noise
full_image.addNoise(noise)  # Add the noise to the image to simulate realistic sensor effects

# ==========================================
# 5. EXPORT THE FITS FILE (Changed from PNG)
# ==========================================
# GalSim automatically detects the .fits extension and saves the raw 32-bit math.
# You can use .fit or .fits, both are standard.
fits_filename = 'starfield_001.fits'  # Set the filename for the FITS output file
full_image.write(fits_filename)  # Write the image to the FITS file
print(f"Scientific image saved: {fits_filename}")  # Print a message confirming the save

# 6. Export the Labels
csv_filename = 'starfield_001_labels.csv'  # Set the filename for the CSV labels file
with open(csv_filename, 'w', newline='') as csvfile:  # Open the CSV file for writing
    writer = csv.writer(csvfile)  # Create a CSV writer object
    writer.writerow(['star_id', 'x_pixel', 'y_pixel', 'flux'])  # Write the header row with column names
    writer.writerows(label_data)  # Write all the label data rows
    
print(f"Labels saved: {csv_filename}")  # Print a message confirming the labels save
