import os
import pandas as pd

#This code is to collect all the CSV files containing the metadata from each directory into one CSV file.

def gather_data(root_dir):
    data_frames = []

    # walk through the root directory and each subdirectory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # check if the file is a CSV
            if filename.endswith('.csv'):
                # construct the full filepath
                filepath = os.path.join(dirpath, filename)
                # read the CSV file into a pandas DataFrame
                df = pd.read_csv(filepath)
                # append this data frame to the list
                data_frames.append(df)

    # concatenate all the data frames together
    combined_df = pd.concat(data_frames, ignore_index=True)

    return combined_df

root_dir = './'  # replace with your root directory
combined_df = gather_data(root_dir)

# check the combined dataframe
print(combined_df)
combined_df.to_csv('combined_data.csv', index=False)
