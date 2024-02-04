import os
import shutil

def copy_combined_images(root_dir, destination_dir_name="combined images"):
    # Create the destination directory if it doesn't exist
    destination_dir = os.path.join(root_dir, destination_dir_name)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Traverse the directory structure
    for subdir, dirs, files in os.walk(root_dir):
        # Skip the destination directory
        if destination_dir_name in dirs:
            dirs.remove(destination_dir_name)

        for file in files:
            if "Combined" in file and (file.endswith('.jpeg') or file.endswith('.png')):
                # Construct the full file path
                file_path = os.path.join(subdir, file)

                # Copy the file to the destination directory
                shutil.copy(file_path, destination_dir)
                print(f"Copied {file} to {destination_dir}")

# Example usage
root_directory = "./"  # Replace with your directory path
copy_combined_images(root_directory)
