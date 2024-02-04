import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('combined_data_sample2500.csv')

# Filter the original images to get their sizes
original_sizes = df[df['CQ_Algorithm'].isna()][['Image Name', 'Image File Size']].set_index('Image Name')

# Join with the rest of the dataframe to associate original sizes as new column
df = df.join(original_sizes, on='Image Name', rsuffix='_original') 

# Filter the original image unique colors
original_color_count = df[df['CQ_Algorithm'].isna()][['Image Name', 'Number of unique colors']].set_index('Image Name')

# Join with the rest of the dataframe to associate original color counts
df = df.join(original_color_count, on='Image Name', rsuffix='_original') 

df['Compression Percentage'] = 100 - 1/df['Compression Ratio'] * 100

# Define a function to compute correlations
def compute_correlations(df, condition, original_size_col, target_col):
    subset = df[condition]

    # pd.set_option('display.max_columns', None)
    # print(subset[original_size_col])
    # print(subset[target_col])

    return subset[original_size_col].corr(subset[target_col])

# WebP and AVIF in lossless mode
for method in ['WebP', 'AVIF']:
    condition = df['CQ_Algorithm'] == 'original + ' + method + '-lossless'
    print(f"Correlation between original file size and compression ratio for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Image File Size_original', 'Compression Ratio'))

    print(f"Correlation between original file size and compression percentage for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Image File Size_original', 'Compression Percentage'))

    print(f"Correlation between original file size and compression time for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Image File Size_original', 'Compression Time'))

    print(f"Correlation between original number of colors and compression ratio for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Ratio'))

    print(f"Correlation between original number of colors and compression percentage for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Percentage'))

    print(f"Correlation between original number of colors and compression time for {method} (Lossless):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Time'))
    print("\n")

# WebP and AVIF with quality as 100
for method in ['WebP', 'AVIF']:
    condition = df['CQ_Algorithm'] == 'original + ' + method + '-100'
    print(f"Correlation between original file size and compression ratio for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Image File Size', 'Compression Ratio'))

    print(f"Correlation between original file size and compression percentage for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Image File Size', 'Compression Percentage'))

    print(f"Correlation between original file size and compression time for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Image File Size', 'Compression Time'))

    print(f"Correlation between original number of colors and compression percentage for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Percentage'))

    print(f"Correlation between original number of colors and compression ratio for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Ratio'))

    print(f"Correlation between original number of colors and compression time for {method} (Quality 100):")
    print(compute_correlations(df, condition, 'Number of unique colors_original', 'Compression Time'))
    print("\n")
