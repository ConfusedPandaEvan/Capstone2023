import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
df = pd.read_csv('combined_data_sample2500.csv')

# Define the color quantization (CQ) algorithms of interest
# cq_algorithms = ['Simple Color Reduction', 'Image Dithering', 'Median Cut', 'Octree',
#                  'medianCut + WebP-60', 'reduce + WebP-60', 'Octree + WebP-60', 'dither + WebP-60',
#                  'medianCut + WebP-30', 'reduce + WebP-30', 'Octree + WebP-30', 'dither + WebP-30',
#                  'medianCut + WebP-90', 'reduce + WebP-90', 'Octree + WebP-90', 'dither + WebP-90', 'original + WebP-30', 'original + WebP-60', 'original + WebP-90']
cq_algorithms = ['Median Cut', 'Octree', 'Simple Color Reduction', 'Image Dithering', 'original + WebP-100', 'original + WebP-lossless', 'original + AVIF-100', 'original + AVIF-lossless']
medianCut = ['Median Cut', 'medianCut + WebP-100', 'medianCut + WebP-lossless', 'medianCut + AVIF-lossless', 'medianCut + AVIF-100', 'original + WebP-100', 'original + WebP-lossless', 'original + AVIF-100', 'original + AVIF-lossless']
Octree = ['Octree', 'Octree + WebP-100', 'Octree + WebP-lossless', 'Octree + AVIF-100', 'Octree + AVIF-lossless', 'original + WebP-100', 'original + WebP-lossless', 'original + AVIF-100', 'original + AVIF-lossless']
SimpleReduction = ['Simple Color Reduction', 'reduce + WebP-100', 'reduce + WebP-lossless', 'reduce + AVIF-100', 'reduce + AVIF-lossless', 'original + WebP-100', 'original + WebP-lossless', 'original + AVIF-100', 'original + AVIF-lossless']
ImageDithering = ['Image Dithering', 'dither + WebP-100', 'dither + WebP-lossless', 'dither + AVIF-100', 'dither + AVIF-lossless', 'original + WebP-100', 'original + WebP-lossless', 'original + AVIF-100', 'original + AVIF-lossless']

plots = [cq_algorithms, medianCut,  Octree, SimpleReduction, ImageDithering]

# Filter dataframe by CQ algorithms
for plot in plots:
    df1 = df[df['CQ_Algorithm'].isin(plot)]
    df1['Compression Percentage'] = 100 - 1/df1['Compression Ratio'] * 100

    #Boxplot Normal Scale
    plt.figure(figsize=(15,10))
    sns.boxplot(x='Max number of colors', y='Compression Percentage', hue='CQ_Algorithm', data=df1, showfliers = False)
    plt.title('Box Plot of Compression Ratio vs Max Number of Colors')
    plt.show()

    # Boxplot Log Scale
    plt.figure(figsize=(15,10))
    sns.boxplot(x='Max number of colors', y='Compression Percentage', hue='CQ_Algorithm', data=df1, showfliers = False)
    plt.yscale('log')
    plt.title('Box Plot of Compression Ratio vs Max Number of Colors')
    plt.show()



