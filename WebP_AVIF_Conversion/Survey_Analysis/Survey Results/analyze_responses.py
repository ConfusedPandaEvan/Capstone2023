import pandas as pd

def print_stats(df, target_column):
    # Calculate and print the average of the target_column
    average_value = df[target_column].mean()
    print(f"Mean value of {target_column}: {average_value}")

    # Calculate and print the minimum, 1st quantile, 3rd quantile, and maximum value of target_column
    min_value = df[target_column].min()
    first_quantile = df[target_column].quantile(0.25)
    second_quantile = df[target_column].quantile(0.5)
    third_quantile = df[target_column].quantile(0.75)
    max_value = df[target_column].max()

    print(f"Minimum value: {min_value}")
    print(f"1st Quantile: {first_quantile}")
    print(f"2nd Quantile: {second_quantile}")
    print(f"3rd Quantile: {third_quantile}")
    print(f"Maximum value: {max_value}")

def analyze_and_export_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    for column in ['mean_choice', 'median_choice']:
        print_stats(df, column)

file_path = 'avif_output.csv'
analyze_and_export_csv(file_path)
