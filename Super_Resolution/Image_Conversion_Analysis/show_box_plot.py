import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files
avif_df = pd.read_csv('./AVIF_Compression_Percentage.csv')
webp_df = pd.read_csv('./WebP_Compression_Percentage.csv')
resize_df = pd.read_csv('./RESIZE_Compression_Percentage.csv')

# Reshape the dataframes into a long format for AVIF and WebP
avif_melted = avif_df.melt(id_vars=["Unnamed: 0"], 
                        #    value_vars=["10", "25", "50", "75", "lossless"],
                           value_vars=["10", "25", "50", "75"],
                           var_name="Quality Setting", 
                           value_name="Compression Percentage")
avif_melted["Image Type"] = ["AVIF-" + i for i in avif_melted["Quality Setting"]]

webp_melted = webp_df.melt(id_vars=["Unnamed: 0"], 
                        #    value_vars=["10", "25", "50", "75", "lossless"],
                           value_vars=["10", "25", "50", "75"],
                           var_name="Quality Setting", 
                           value_name="Compression Percentage")
webp_melted["Image Type"] = ["WebP-" + i for i in webp_melted["Quality Setting"]]

# Reshaping the resize_df
resize_melted = pd.DataFrame({
    "Unnamed: 0": resize_df["Unnamed: 0"],
    "Quality Setting": "compression",
    "Compression Percentage": resize_df["compression percentage"],
    "Image Type": "resized"
})

# Combining the dataframes for plotting
all_data_corrected = pd.concat([avif_melted, webp_melted, resize_melted], ignore_index=True)

# Creating the boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(data=all_data_corrected, x="Image Type", y="Compression Percentage", palette="Set2")
plt.title("Boxplot of Compression Percentages")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.yscale('log')
plt.show()
