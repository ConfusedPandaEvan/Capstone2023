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


#This code is for Color Quantization of images. The code also converts images into WebP and AVIF format in a lossless manner for comparison.


max_colors = 8
# for median cut = [3*1, 3*2, 3*3, ... 3*8] or [3, 6, 9, ... 24]
# for Simple Color Reduction & Image Dithering = [2^1, 2^2, 2^3 ... 2^8]
# This will cover all color range up to 256 per rgb that is supported by JPEG format. 8 range used to have comparison between color quantization algorithms


# Code for Color Reduction for Simple Color Reduction & Image Dithering with Floyd-Steinberg Algorithm
def get_new_val(old_val, nc):
    # Get the "closest" colour to old_val in the range [0,1] per channel divided into nc values.
    return np.round(old_val * (nc - 1)) / (nc - 1)

# Floyd-Steinberg dither the image img into a palette with nc colours per channel.
# Source: https://scipython.com/blog/floyd-steinberg-dithering/
def fs_dither(img, nc):
    width, height = img.size
    arr = np.array(img, dtype=float) / 255

    for ir in range(height):
        for ic in range(width):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            # In this simple example, we will just ignore the border pixels.
            if ic < width - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < height - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < width - 1:
                    arr[ir+1, ic+1] += err / 16

    carr = np.array(arr/np.max(arr, axis=(0, 1)) * 255, dtype=np.uint8)
    return Image.fromarray(carr)

# Simple palette reduction without dithering.

def palette_reduce(img, nc):
    arr = np.array(img, dtype=float) / 255
    arr = get_new_val(arr, nc)

    carr = np.array(arr/np.max(arr) * 255, dtype=np.uint8)
    return Image.fromarray(carr)

# Median Cut Algorithm for Color Quantization
# Source: https://muthu.co/reducing-the-number-of-colors-of-an-image-using-median-cut-algorithm/
def median_cut(sample_img, colors):
    def median_cut_quantize(img, img_arr):
        # when it reaches the end, color quantize
        r_average = np.mean(img_arr[:, 0])
        g_average = np.mean(img_arr[:, 1])
        b_average = np.mean(img_arr[:, 2])

        for data in img_arr:
            sample_img[data[3]][data[4]] = [r_average, g_average, b_average]

    def split_into_buckets(img, img_arr, depth):
        if len(img_arr) == 0:
            return

        if depth == 0:
            median_cut_quantize(img, img_arr)
            return

        r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
        g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
        b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

        space_with_highest_range = 0

        if g_range >= r_range and g_range >= b_range:
            space_with_highest_range = 1
        elif b_range >= r_range and b_range >= g_range:
            space_with_highest_range = 2
        elif r_range >= b_range and r_range >= g_range:
            space_with_highest_range = 0

        # sort the image pixels by color space with highest range
        # and find the median and divide the array.
        img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
        median_index = int((len(img_arr) + 1) / 2)

        # split the array into two blocks
        split_into_buckets(img, img_arr[0:median_index], depth - 1)
        split_into_buckets(img, img_arr[median_index:], depth - 1)

    flattened_img_array = []
    for rindex, rows in enumerate(sample_img):
        for cindex, color in enumerate(rows):
            flattened_img_array.append(
                [color[0], color[1], color[2], rindex, cindex])

    flattened_img_array = np.array(flattened_img_array)

    # start the splitting process
    split_into_buckets(sample_img, flattened_img_array, colors)

    return sample_img

# This method will not be used due to it's high complexity of O(n^3) and is inefficient for the study.


def octree_quantize(img, n_colors):

    # Ensure the image is in the correct mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Quantize the image using the octree algorithm
    octree_img = img.quantize(colors=n_colors)

    # Convert the quantized image back to RGB mode
    octree_img = octree_img.convert('RGB')

    return octree_img

failed_image_list = []

def webp_conversion(directory, og_file_size, df):
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            for quality in [100]:
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
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(directory, filename))
            for quality in [100]:
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

# Code for counting the number of unique colors in an image. For using numpy 3d
def count_unique_colors_np(img_np):
    # Ensure the image is 3-dimensional (height x width x color channels)
    assert len(img_np.shape) == 3, "Input should be a 3D numpy array."

    # Reshape the data to 2D (just keeping color data)
    h, w, c = img_np.shape
    img_np = img_np.reshape(-1, c)

    # Use numpy's unique function to find unique rows (colors) in the data
    unique_colors = np.unique(img_np, axis=0)

    # The number of unique colors is the number of unique rows
    num_unique_colors = len(unique_colors)

    return num_unique_colors

# Code for getting image, making a folder and saving images in the directory,
# keep a pandas dataframe to store 'Image Name', 'Image File Size', 'Compression Ratio', 'Compression Time', 'CQ_Algorithm', 'max number of colors’, 'number of unique colors' per image generated by color quantization algorithm

directory = './'
def convert_images(filename):
    # for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):

        try:

            # Get file with jpeg extension
            img = Image.open(os.path.join(directory, filename))

            # Open different folder with the name of Original Image
            color_quant_dir = directory + filename.split('.')[0]
            if os.path.exists(color_quant_dir):
                return

            os.makedirs(color_quant_dir)

            # Check if image can be run through median cut
            sample_img = imread(filename)
            if sample_img.ndim != 3 or sample_img.shape[2] != 3:
                return

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

            # Common number of color quantization per image
            for nc in range(2, max_colors):
                color_count = 2**(3*nc)

                # print("CQ running for {} number of colors on {}.".format(color_count, filename))

                # Code for changing jpeg to Simple Color Reduction
                start_time = time.time()
                sr_image = palette_reduce(img, 2**nc)
                compression_time = time.time() - start_time

                num_colors = count_unique_colors(sr_image)
                image_name = filename.split(
                    '.')[0] + '-reduce-{}.jpeg'.format(color_count)
                # save image in folder name
                sr_image.save(os.path.join(color_quant_dir, image_name))

                # Get Meta Data and Store in DataFrame
                file_info = os.stat(os.path.join(color_quant_dir, image_name))
                file_size = file_info.st_size
                df1 = pd.Series({'Image Name': filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': 'Simple Color Reduction', 'Max number of colors': 2**(3*(nc)), 'Number of unique colors': num_colors})
                df = pd.concat([df, df1.to_frame().T], ignore_index=True)

                # Code for changing jpeg to Dithered Image
                start_time = time.time()
                dither_image = fs_dither(img, 2**nc)
                compression_time = time.time() - start_time

                num_colors = count_unique_colors(dither_image)
                image_name = filename.split(
                    '.')[0] + '-dither-{}.jpeg'.format(color_count)

                # save image in folder name
                dither_image.save(os.path.join(color_quant_dir, image_name))
                # Get Meta Data and Store in DataFrame
                file_info = os.stat(os.path.join(color_quant_dir, image_name))
                file_size = file_info.st_size
                df1 = pd.Series({'Image Name': filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': 'Image Dithering', 'Max number of colors': 2**(3*(nc)), 'Number of unique colors': num_colors})
                df = pd.concat([df, df1.to_frame().T], ignore_index=True)

                # Code for changing jpeg to Median Cut
                start_time = time.time()
                mc_image = median_cut(sample_img, 3*nc)
                compression_time = time.time() - start_time

                num_colors = count_unique_colors_np(mc_image)
                image_name = filename.split(
                    '.')[0] + '-medianCut-{}.jpeg'.format(color_count)
                imsave(os.path.join(color_quant_dir, image_name), mc_image)

                # Get Meta Data and Store in DataFrame
                file_info = os.stat(os.path.join(color_quant_dir, image_name))
                file_size = file_info.st_size
                df1 = pd.Series({'Image Name': filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': 'Median Cut', 'Max number of colors': 2**(3*(nc)), 'Number of unique colors': num_colors})
                df = pd.concat([df, df1.to_frame().T], ignore_index=True)

                # Code for changing jpeg to Octree
                start_time = time.time()
                octree_image = octree_quantize(img, 2**nc)
                compression_time = time.time() - start_time

                num_colors = count_unique_colors(octree_image)
                image_name = filename.split(
                    '.')[0] + '-Octree-{}.jpeg'.format(color_count)
                octree_image.save(os.path.join(color_quant_dir, image_name))

                # Get Meta Data and Store in DataFrame
                file_info = os.stat(os.path.join(color_quant_dir, image_name))
                file_size = file_info.st_size
                df1 = pd.Series({'Image Name': filename, 'Image File Size': file_size, 'Compression Time': compression_time, 'Compression Ratio': og_file_size / file_size,
                                'CQ_Algorithm': 'Octree', 'Max number of colors': 2**(3*(nc)), 'Number of unique colors': num_colors})
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


with open('failed_image_list.pkl', 'wb') as f: 
    pickle.dump(failed_image_list, f)
