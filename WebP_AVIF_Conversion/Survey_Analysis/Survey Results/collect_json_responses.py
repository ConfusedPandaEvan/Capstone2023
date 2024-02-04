import json
import os
import statistics
import pandas as pd
import numpy as np

#read all json files and convert them to one dictionary with name of image as key and list of integer 'choice' values 
def process_json_files(directory):
    result = {}

    for filename in os.listdir(directory):


        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)


            with open(file_path, 'r') as file:
                data = json.load(file)
            
            for key, value in data.items():

                if key.endswith('.jpeg') or key.endswith('.jpg') or key.endswith('.png'):
                    
                    image_name = key

                    if image_name not in result:
                        result[image_name] = []
                    
                    result[image_name].append(int(value['choice']))
    
    return result


def get_average_choice_as_pdDf(dictionary):

    data = {
        'filename': [],
        'mean_choice': [],
        'median_choice': [],
        'number_of_responses': [],
        'variance': []
    }

    for key, values in dictionary.items():
        avg = sum(values) / len(values) if values else 0

        data['filename'].append(key)
        data['mean_choice'].append(avg)
        data['median_choice'].append(statistics.median(values))
        data['variance'].append(np.var(values, ddof=1)) #ddof=1 for sample variance
        data['number_of_responses'].append(len(values))

    df = pd.DataFrame(data)

    return df


#Choose directory with json files with responses
dir = './avif'

dict = process_json_files(dir)


df = get_average_choice_as_pdDf(dict)

print(df)

csv_file_name = 'avif_output.csv'
df.to_csv(csv_file_name, index=False)
