import numpy as np
import os


def get_data():

    data_folder = './datasets/gas/driftdataset'
    data_arrays = []

    # loop over all the dat files in the folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.dat'):
            filepath = os.path.join(data_folder, filename)
            
            # read the data from the file and convert it to a numpy array
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    # split the line into label and values
                    label, values = line.strip().split(';')

                    # parse the values into a dictionary or list
                    values = values.split()
                    values = values[1:]
                    values_dict = {}
                    for v in values:
                        key, value = v.split(':')
                        values_dict[int(key)] = float(value)

                    # add the label and values to the data list
                    data.append((int(label), values_dict))

            # convert the list of tuples into a numpy array
            labels = np.array([d[0] for d in data])
            values = np.array([list(d[1].values()) for d in data])
            data_array = np.hstack((labels.reshape(-1,1), values))
            
            data_arrays.append(data_array)

    # stack all the data arrays vertically into a single numpy array
    data_array_all = np.vstack(data_arrays)
    
    return data_array_all
    
    
