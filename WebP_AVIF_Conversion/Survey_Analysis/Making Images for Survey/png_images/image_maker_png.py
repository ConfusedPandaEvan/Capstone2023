import os
from PIL import Image, ImageDraw, ImageFont

def combine_images(root_dir):
    # Iterate over all subdirectories in the root directory
    for subdir, _, files in os.walk(root_dir):
        # Filter out image files that match the criteria
        avif_images = sorted([f for f in files if 'avif' in f])
        webp_images = sorted([f for f in files if 'webp' in f])
        


        # If there are avif or webp images, process them
        for img_list, compression in [(avif_images, 'avif'), (webp_images, 'webp')]:
            if img_list:
                # Calculate the combined height based on resized images
                resized_heights = []
                for img in img_list:
                    image = Image.open(os.path.join(subdir, img))
                    if image.width > 720:
                        resized_height = int(720 * image.height / image.width)
                    else:
                        resized_height = image.height
                    resized_heights.append(resized_height)
                
                # Load a font for the text with size set to 48
                font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 48)

                # Calculate the text height for one instance (since it'll be the same for all)
                dummy_image = Image.new('RGBA', (1, 1))
                dummy_draw = ImageDraw.Draw(dummy_image)
                
                _, _, sample_text_width, sample_text_height = dummy_draw.textbbox((0, 0), "quality = 100", font=font) # assuming maximum width for text

                combined_height = sum(resized_heights) + sample_text_height * len(img_list) + 48 * (len(img_list) - 1) + 30

                # Create a new image with the combined dimensions
                combined_image = Image.new('RGBA', (720, combined_height), color='white')
                draw = ImageDraw.Draw(combined_image)

                # Paste each image onto the new image, center aligned and add quality text
                y_offset = 0
                for img, height in zip(img_list, resized_heights):
                    image = Image.open(os.path.join(subdir, img))
                    
                    # Resize if the image width is greater than 720
                    if image.width > 720:
                        image = image.resize((720, height))
                    
                    # Calculate x-offset for center alignment
                    x_offset = (720 - image.width) // 2
                    combined_image.paste(image, (x_offset, y_offset))
                    
                    quality = img.split('-')[1]
                    quality = "quality = " + quality

                    _, _, text_width, text_height = draw.textbbox((0, 0), quality, font=font)
                    draw.text(((720 - text_width) / 2, y_offset + image.height + 10), quality, font=font, fill="black")
                    y_offset += image.height + 48 + text_height

                # Save the combined image
                base_name = img_list[0].split('-')[0]
                combined_image.save(os.path.join(subdir, f'{base_name}-{compression}-Combined.png'), "PNG")

# Call the function on your root directory
combine_images('./')
