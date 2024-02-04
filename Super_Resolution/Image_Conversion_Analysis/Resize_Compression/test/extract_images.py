import json
import os
import shutil

# Load the list of images from the provided JSON file
file_path = 'images-list.json'
with open(file_path, 'r') as file:
    images_list = json.load(file)

# Directory containing the images
images_directory = './jpeg_original'

sample_folder = './sample' 
os.makedirs(sample_folder, exist_ok=True)

# Copy images and track those that are not found or cannot be copied
not_copied_images = []

for image in images_list:
    source_path = os.path.join(images_directory, image)
    destination_path = os.path.join(sample_folder, image)
    
    if os.path.isfile(source_path):
        try:
            shutil.copy(source_path, destination_path)
        except Exception as e:
            not_copied_images.append(image)
            print(f"Error copying {image}: {e}")
    else:
        not_copied_images.append(image)

# Return the list of images that were not copied
not_copied_images
