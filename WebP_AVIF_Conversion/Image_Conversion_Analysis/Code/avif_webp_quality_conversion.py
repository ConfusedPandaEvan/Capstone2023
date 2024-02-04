import os
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import pandas as pd
import time
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import pillow_avif

max_colors = 8
failed_image_list = []

# Code for counting the number of unique colors in an image.
def count_unique_colors(img):

    # Ensure the image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert the image into a numpy array
    img_np = np.array(img)

    # Reshape the data to 2D (just keeping color data)
    h, w, c = img_np.shape
    img_np = img_np.reshape(-1, c)

    # Use numpy's unique function to find unique rows (colors) in the data
    unique_colors = np.unique(img_np, axis=0)

    # The number of unique colors is the number of unique rows
    num_unique_colors = len(unique_colors)

    return num_unique_colors

#WebP Conversion of Images
def webp_conversion(directory, og_file_size, df):
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory, filename))
            for quality in [10, 25, 50, 75]:
                webp_filename = filename.rsplit('.', 1)[0] + '-{}.webp'.format(quality)
                start_time = time.time()
                img.save(os.path.join(directory, webp_filename), 'webp', quality = quality)
                compression_time = time.time() - start_time
                file_info = os.stat(os.path.join(directory, webp_filename))
                file_size = file_info.st_size
                num_colors = count_unique_colors(Image.open(os.path.join(directory, webp_filename)))
                max_colors = filename.split('.')[0].split('-')[-1]
                og_filename = filename.split('-')[0] + '.jpeg'
                if filename.split('-')[1] != '.jpeg' or filename.split('-')[1] != '.jpg':
                    cq_algo = filename.split('-')[1].split('.')[0]
                else:
                    cq_algo = 'None'
                df1 = pd.Series({'Image Name': og_filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': cq_algo + ' + WebP-{}'.format(quality), 'Max number of colors': max_colors, 'Number of unique colors': num_colors})
                df = pd.concat([df, df1.to_frame().T], ignore_index=True)

            webp_filename = filename.rsplit('.', 1)[0] + '-{}.webp'.format('lossless')
            start_time = time.time()
            img.save(os.path.join(directory, webp_filename), 'webp', lossless=True)
            compression_time = time.time() - start_time
            file_info = os.stat(os.path.join(directory, webp_filename))
            file_size = file_info.st_size
            num_colors = count_unique_colors(Image.open(os.path.join(directory, webp_filename)))
            max_colors = filename.split('.')[0].split('-')[-1]
            og_filename = filename.split('-')[0] + '.jpeg'
            if filename.split('-')[1] != '.jpeg' or filename.split('-')[1] != '.jpg':
                cq_algo = filename.split('-')[1].split('.')[0]
            else:
                cq_algo = 'None'
            df1 = pd.Series({'Image Name': og_filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                            'CQ_Algorithm': cq_algo + ' + WebP-{}'.format('lossless'), 'Max number of colors': max_colors, 'Number of unique colors': num_colors})
            df = pd.concat([df, df1.to_frame().T], ignore_index=True)

    return df


def avif_conversion(directory, og_file_size, df):
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(directory, filename))
            for quality in [10, 25, 50, 75]:
                avif_filename = filename.rsplit('.', 1)[0] + '-{}.avif'.format(quality)
                start_time = time.time()
                img.save(os.path.join(directory, avif_filename), format='AVIF', quality = quality)
                compression_time = time.time() - start_time
                file_info = os.stat(os.path.join(directory, avif_filename))
                file_size = file_info.st_size
                num_colors = count_unique_colors(Image.open(os.path.join(directory, avif_filename)))
                max_colors = filename.split('.')[0].split('-')[-1]
                og_filename = filename.split('-')[0] + '.jpeg'
                if filename.split('-')[1] != '.jpeg' or filename.split('-')[1] != '.jpg':
                    cq_algo = filename.split('-')[1].split('.')[0]
                else:
                    cq_algo = 'None'
                df1 = pd.Series({'Image Name': og_filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': cq_algo + ' + AVIF-{}'.format(quality), 'Max number of colors': max_colors, 'Number of unique colors': num_colors})
                df = pd.concat([df, df1.to_frame().T], ignore_index=True)
            avif_filename = filename.rsplit('.', 1)[0] + '-{}.avif'.format('lossless')
            start_time = time.time()
            img.save(os.path.join(directory, avif_filename), format='AVIF', lossless=True)
            compression_time = time.time() - start_time
            file_info = os.stat(os.path.join(directory, avif_filename))
            file_size = file_info.st_size
            num_colors = count_unique_colors(Image.open(os.path.join(directory, avif_filename)))
            max_colors = filename.split('.')[0].split('-')[-1]
            og_filename = filename.split('-')[0] + '.jpeg'
            if filename.split('-')[1] != '.jpeg' or filename.split('-')[1] != '.jpg':
                cq_algo = filename.split('-')[1].split('.')[0]
            else:
                cq_algo = 'None'
            df1 = pd.Series({'Image Name': og_filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                            'CQ_Algorithm': cq_algo + ' + AVIF-{}'.format('lossless'), 'Max number of colors': max_colors, 'Number of unique colors': num_colors})
            df = pd.concat([df, df1.to_frame().T], ignore_index=True)
    return df


directory = './'
def convert_images(filename):
    # for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):

        try:

            # Get file with jpeg extension
            img = Image.open(os.path.join(directory, filename))


            # Open different folder with the name of Original Image
            color_quant_dir = directory + filename.split('.')[0]
            if os.path.exists(color_quant_dir):
                return

            os.makedirs(color_quant_dir)

            # Save Original Image in the new directory
            image_name = filename.split('.')[0] + '-original.jpeg'
            img.save(os.path.join(color_quant_dir, image_name))

            # Create an empty DataFrame to hold your data
            df = pd.DataFrame(columns=('Image Name', 'Image File Size', 'Compression Time', 'Compression Ratio',
                            'CQ_Algorithm', 'Max number of colors', 'Number of unique colors'))

            # Get file info of original Image
            file_info = os.stat(os.path.join(color_quant_dir, image_name))
            og_file_size = file_info.st_size
            num_colors = count_unique_colors(img)

            # Add Current Image to pandas df
            # df1 = pd.Series([filename, file_size, -1, 'None', -1, num_colors], columns=['Image Name', 'Image File Size', 'Compression Time',
            #                                                                             'CQ_Algorithm', 'Max number of colors', 'Number of unique colors'])
            df1 = pd.Series({'Image Name': filename, 'Image File Size': og_file_size, 'Compression Time': -1, 'Compression Ratio': 1,
                                'CQ_Algorithm': 'None', 'Max number of colors': -1, 'Number of unique colors': num_colors})
            df = pd.concat([df, df1.to_frame().T], ignore_index=True)

            # changing all jpeg images to webp after all jpegs have been color quantized
            df = webp_conversion(color_quant_dir, og_file_size, df)

            df = avif_conversion(color_quant_dir, og_file_size, df)

            # TODO: Save Pandas DF in the new directory
            df_file = '{}.csv'.format(filename.split('.')[0])
            df.to_csv(os.path.join(color_quant_dir, df_file), index=False)
        except:
            failed_image_list.append(filename)
            pass


filenames = os.listdir('./')

# convert_images(filenames)
Parallel(n_jobs = 10, prefer= "threads")(delayed(convert_images)(filename)for filename in tqdm(filenames))


#Save failed images list as a pickle file
with open('failed_image_list.pkl', 'wb') as f: 
    pickle.dump(failed_image_list, f)