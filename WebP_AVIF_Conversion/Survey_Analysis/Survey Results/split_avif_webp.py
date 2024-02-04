import json
import os
import shutil

def sort_and_move_json_files(folder_path):
    # Paths for the new directories
    avif_dir = os.path.join(folder_path, 'avif')
    webp_dir = os.path.join(folder_path, 'webp')

    # Create directories if they don't exist
    if not os.path.exists(avif_dir):
        os.makedirs(avif_dir)
    if not os.path.exists(webp_dir):
        os.makedirs(webp_dir)

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Check the type of each image in the JSON file
            avif_count = 0
            webp_count = 0
            for key, value in data.items():
                if isinstance(value, dict) and 'type' in value:
                    if value['type'] == 'avif':
                        avif_count += 1
                    elif value['type'] == 'webp':
                        webp_count += 1

            # Move the file to the appropriate directory based on the predominant type
            if avif_count >= webp_count:
                shutil.move(file_path, os.path.join(avif_dir, filename))
            else:
                shutil.move(file_path, os.path.join(webp_dir, filename))

# Usage
folder_path = 'avif_webp'
sort_and_move_json_files(folder_path)
