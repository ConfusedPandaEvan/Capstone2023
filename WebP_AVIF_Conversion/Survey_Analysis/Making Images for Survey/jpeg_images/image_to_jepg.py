import os
from PIL import Image
import pillow_avif

def convert_and_rename(directory, old_extension, new_suffix, converter_function=None):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(old_extension):
                old_path = os.path.join(root, file)
                new_name = file.rsplit(".", 1)[0] + new_suffix + ".jpeg"
                new_path = os.path.join(root, new_name)
                
                if converter_function:
                    img = Image.open(old_path)
                    img.save(new_path) 
                else:
                    with Image.open(old_path) as img:
                        img.convert("RGB").save(new_path, "JPEG")
                os.remove(old_path)

def main():
    root_dir = "./"  # Current directory

    # Convert and rename avif files
    convert_and_rename(root_dir, ".avif", "-avif", converter_function=1)

    

    # Convert and rename webp files
    convert_and_rename(root_dir, ".webp", "-webp")

if __name__ == "__main__":
    main()
