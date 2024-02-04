import os
import csv

def is_image_file(filename):
    """ Check if a file is an image based on its extension """
    image_extensions = ['.jpg', '.jpeg', '.png']
    return os.path.splitext(filename)[1].lower() in image_extensions

def get_file_size(folder_path):
    """ Returns a dictionary of file sizes keyed by image name """
    file_sizes = {}
    for file_name in os.listdir(folder_path):
        if is_image_file(file_name):
            file_path = os.path.join(folder_path, file_name)
            file_sizes[file_name] = os.path.getsize(file_path)
    return file_sizes

def calculate_compression_percentage(original_size, new_size):
    """ Calculates the compression percentage """
    return (1 - new_size / original_size) * 100 if original_size != 0 else 0

def main():
    original_folder = 'original'
    resized_folder = 'resized'

    original_sizes = get_file_size(original_folder)
    resized_sizes = get_file_size(resized_folder)

    with open('image_compression_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Original Size (Bytes)', 'New Size (Bytes)', 'Compression Percentage'])

        for image_name, original_size in original_sizes.items():
            if image_name in resized_sizes:
                compression_percentage = calculate_compression_percentage(original_size, resized_sizes[image_name])
                writer.writerow([image_name, original_size, resized_sizes[image_name], compression_percentage])

if __name__ == "__main__":
    main()
