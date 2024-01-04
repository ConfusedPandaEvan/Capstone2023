# Capstone2023
The repository contains code for different methods of optimizing images for compression to reduce bandwidth and increase mobile data usage. The methods looked into are 1) Color Quantization, 2) WebP &amp; AVIF Lossy Compression, and 3) ESRGAN.


# Color Quantization
The code in the repo uses algorithmic color quantization: Simple Color Reduction, Image Dithering (Floyd-Steinberg Algorithm), Median Cut, and Octree. The purpose of the code is to see if reducing colors in an algorithmic method would be a feasible stand-alone solution or a pre-process for WebP &amp; AVIF conversion.

The code* will reduce the color in an image using the aforementioned algorithms in varying numbers of maximum colors. Run this code in a directory with jpg and jpeg images. (PNG images will not work since the code will not work on RGBA images.) This code will generate a directory for each image containing different versions of the image. The code will also generate a CSV file containing metadata of the images.

The code* will collect the CSV files containing the metadata of the converted images.


# WebP & AVIF Lossy Compression
The code in the repo consists of 3 sections. 1) WebP & AVIF conversion with 10, 25, 50, and 75 compression qualities. A code to collect the results. 2) Resizing of images for survey analysis. 3) Data analysis for real-world impact analysis.

# ESRGAN Conversion
The code in the repo consists of 1) Simple code to resize the images to a. fit the mobile screen of the phone and b. resize the image to 1/4 of the image for later ESRGAN conversion. 2) ipynb file to convert the resized images to their original size using super-resolution.

# Data Visualization
The code in the repo is for visualizing the results from the three methods of reducing image sizes.


please get in touch with me through hwl278@nyu.edu if you have any questions.
