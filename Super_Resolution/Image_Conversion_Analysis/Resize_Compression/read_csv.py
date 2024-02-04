import pandas as pd

def calculate_statistics(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Calculating the required statistical values for the 'Compression Percentage' column
    mean_value = data['Compression Percentage'].mean()
    median_value = data['Compression Percentage'].median()
    quantile_1st = data['Compression Percentage'].quantile(0.25)  # 1st Quantile
    quantile_3rd = data['Compression Percentage'].quantile(0.75)  # 3rd Quantile
    min_value = data['Compression Percentage'].min()
    max_value = data['Compression Percentage'].max()

    # Printing the results
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"1st Quantile: {quantile_1st}")
    print(f"3rd Quantile: {quantile_3rd}")
    print(f"Min: {min_value}")
    print(f"Max: {max_value}")

if __name__ == "__main__":
    file_path = 'image_compression_data.csv'
    calculate_statistics(file_path)
