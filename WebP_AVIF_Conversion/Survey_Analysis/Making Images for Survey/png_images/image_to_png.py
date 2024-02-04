import os
from PIL import Image
import pillow_avif

def convert_and_rename(directory, old_extension, new_suffix, converter_function=None):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(old_extension):
                old_path = os.path.join(root, file)
                new_name = file.rsplit(".", 1)[0] + new_suffix + ".png"
                new_path = os.path.join(root, new_name)
                
                try:
                    if converter_function == 1:
                        img = Image.open(old_path)
                        img.save(new_path)
                    if converter_function == 2:
                        img = Image.open(old_path)
                        img.save(new_path, format="png", lossless = True)
                    os.remove(old_path)
                except Exception as e:
                    print(f"Error processing file {old_path}: {e}")

def main():
    root_dir = "./"  # Current directory

    # Convert and rename avif files
    convert_and_rename(root_dir, ".avif", "-avif", converter_function=1)

    print("done with avif")

    # Convert and rename webp files
    convert_and_rename(root_dir, ".webp", "-webp", converter_function=2)

if __name__ == "__main__":
    main()
