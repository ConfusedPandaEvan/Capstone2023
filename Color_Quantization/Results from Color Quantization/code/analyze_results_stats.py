import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('combined_data_sample2500.csv')

# Define a function to compute the stats for a given column and condition
def compute_stats(df, condition, column):
    subset = df[condition]
    return {
        'max': subset[column].max(),
        'min': subset[column].min(),
        'mean': subset[column].mean(),
        'first_quantile': subset[column].quantile(0.25),
        'third_quantile': subset[column].quantile(0.75)
    }

# Color Quantization methods
cq_methods = ['Simple Color Reduction', 'Image Dithering', 'Median Cut', 'Octree']

# Compute the stats for compression ratio and compression time for each CQ method
for method in cq_methods:
    condition = df['CQ_Algorithm'] == method
    print(f"Stats for {method} (Compression Ratio):")
    print(compute_stats(df, condition, 'Compression Ratio'))
    print(f"Stats for {method} (Compression Time):")
    print(compute_stats(df, condition, 'Compression Time'))
    print("\n")

# Image conversion methods
conversion_methods = [' + WebP-lossless', ' + WebP-100', ' + AVIF-lossless', ' + AVIF-100']

# Compute the stats for compression ratio and compression time for each conversion method
for method in conversion_methods:
    condition = df['CQ_Algorithm'] == 'original' + method
    print(f"Stats for original {method} (Compression Ratio):")
    print(compute_stats(df, condition, 'Compression Ratio'))
    print(f"Stats for original {method} (Compression Time):")
    print(compute_stats(df, condition, 'Compression Time'))
    print("\n")

# Image conversion after CQ
cq_abbrev_methods = ['reduce', 'dither', 'medianCut', 'Octree']

for cq_method in cq_methods:
    for conversion_method in conversion_methods:
        for cq_abbrev_method in cq_abbrev_methods:
            condition = df['CQ_Algorithm'] ==  cq_abbrev_method + conversion_method
            print(f"Stats for {cq_abbrev_method} -> {conversion_method} (Compression Ratio):")
            print(compute_stats(df, condition, 'Compression Ratio'))
            print(f"Stats for {cq_abbrev_method} -> {conversion_method} (Compression Time):")
            print(compute_stats(df, condition, 'Compression Time'))
            print("\n")
