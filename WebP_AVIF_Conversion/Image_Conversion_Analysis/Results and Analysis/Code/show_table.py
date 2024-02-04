import pandas as pd

# Load the data
df = pd.read_csv('./combined_data.csv')

# Filter out blank entries in CQ_Algorithm
df = df[df['CQ_Algorithm'].notna()]

# List of columns to compute statistics on
# columns_to_compute = ['Compression Time', 'Compression Ratio']
columns_to_compute = ['Compression Ratio']

# Define a function to compute statistics
def compute_stats(series):
    return pd.Series({
        'min': series.min(),
        '1st quantile': series.quantile(0.25),
        'mean': series.mean(),
        'median': series.median(),
        '3rd quantile': series.quantile(0.75),
        'max': series.max(),
    })

# Create an empty DataFrame to store results
webp_results = pd.DataFrame()
avif_results = pd.DataFrame()

webp_compression = ['original + WebP-10', 'original + WebP-25', 'original + WebP-50', 'original + WebP-75', 'original + WebP-lossless']
avif_compression = ['original + AVIF-10', 'original + AVIF-25', 'original + AVIF-50', 'original + AVIF-75', 'original + AVIF-lossless']


def compression_ratio(compression_methods, compression_type):
    results = pd.DataFrame()

    for algo in compression_methods:
        subset = df[df['CQ_Algorithm'] == algo]
        
        stats = compute_stats(subset['Compression Ratio'])
        quality = algo.split("-")[-1]
        results[f"{quality}"] = stats

    results.to_csv(f'{compression_type}_Compression_Ratio.csv', index=True)

def compression_percentage(compression_methods, compression_type):
    results = pd.DataFrame()

    for algo in compression_methods:
        subset = df[df['CQ_Algorithm'] == algo]

        stats = compute_stats(subset['Compression Ratio'])
        quality = algo.split("-")[-1]

        #Convert Compress Ratio ro Compress Percentage
        stats = stats.apply(lambda x: 100 - 1/x * 100)
        results[f"{quality}"] = stats

    results.to_csv(f'{compression_type}_Compression_Percentage.csv', index=True)

#Results for WebP
compression_ratio(webp_compression, 'WebP')
compression_percentage(webp_compression, 'WebP')

#Results for AVIF
compression_ratio(avif_compression, 'AVIF')
compression_percentage(avif_compression, 'AVIF')