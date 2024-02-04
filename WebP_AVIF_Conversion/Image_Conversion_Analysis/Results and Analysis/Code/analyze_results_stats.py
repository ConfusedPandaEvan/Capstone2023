import pandas as pd

# Load the data
df = pd.read_csv('combined_data.csv')

# Filter out blank entries in CQ_Algorithm
df = df[df['CQ_Algorithm'].notna()]

# List of columns to compute statistics on
columns_to_compute = ['Compression Time', 'Compression Ratio']

# Loop through each unique CQ_Algorithm
for algo in df['CQ_Algorithm'].unique():
    subset = df[df['CQ_Algorithm'] == algo]
    print(f"\nStatistics for CQ_Algorithm: {algo}")
    
    # Loop through the columns to compute statistics on
    for col in columns_to_compute:
        print(f"\n{col}:")
        print(f"Min: {subset[col].min()}")
        print(f"Max: {subset[col].max()}")
        print(f"1st Quantile: {subset[col].quantile(0.25)}")
        print(f"3rd Quantile: {subset[col].quantile(0.75)}")
        print(f"Mean: {subset[col].mean()}")
