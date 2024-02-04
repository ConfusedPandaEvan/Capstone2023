import os
from PIL import Image
import pillow_avif

# Define the directory containing the images
directory = './'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a PNG image
    if filename.endswith('.png'):
        try:
            # Load the PNG image using PIL
            with Image.open(os.path.join(directory, filename)) as img:
                folder_name = os.path.splitext(filename)[0]  # Get name without extension
                output_folder = os.path.join(directory, folder_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                for quality in [10, 25, 50, 75]:
                    webp_filename = filename.rsplit('.', 1)[0] + '-{}.webp'.format(quality)
                    
                    # Convert and save the image to WebP format
                    img.save(os.path.join(output_folder, webp_filename), 'webp', quality = quality)
                
                    avif_filename = filename.rsplit('.', 1)[0] + '-{}.avif'.format(quality)
                
                    img.save(os.path.join(output_folder, avif_filename), format='AVIF', quality = quality)
        except:
            print(f"Failed for {filename}")

print("Conversion completed!")
