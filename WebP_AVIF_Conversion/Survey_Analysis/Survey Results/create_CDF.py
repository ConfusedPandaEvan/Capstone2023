import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate CDF
def calculate_cdf(data, column):
    a = np.array(data[column])
    a.sort()
    n = a.size
    cdf = np.arange(1, n+1) / n
    return a, cdf

# Load the data from both CSV files
file_path1 = 'avif_output.csv'
file_path2 = 'webp_output.csv'
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# Calculate the CDF for both datasets
median_values1, median_cdf1 = calculate_cdf(data1, 'median_choice')
median_values2, median_cdf2 = calculate_cdf(data2, 'median_choice')

# Plotting the CDF for both datasets
plt.figure(figsize=(10, 6))
plt.plot(median_values1, median_cdf1, label='Median Choice CDF - AVIF', color='blue')
plt.plot(median_values2, median_cdf2, label='Median Choice CDF - WebP', color='red')
plt.title('Cumulative Distribution Function for Median Choices')
plt.xlabel('Quality')
plt.ylabel('CDF')

# Set x-axis to have discrete numbers from 1 to 5 with specific labels
plt.xticks([1, 2, 3, 4, 5], ['10', '25', '50', '75', 'original'])

plt.legend()
plt.grid(True)
plt.show()
