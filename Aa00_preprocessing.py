# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""

import os
import pandas as pd
import json
import Aa00_funcs_preprocessing as funcs


with open('config.json') as json_file:
    config = json.load(json_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
eye_tracking_data = pd.read_csv(path_file)

eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)

output_file = os.path.join(script_dir, config["preprocessed_data_file"])
eye_tracking_data_cm2deg_new.to_csv(output_file, index=False)


# testing data with 100Hz instead of 44Hz (current dataset)

path_file = os.path.join(script_dir,  config["preprocessed_data_file"])
preproc_et= pd.read_csv(path_file)


import numpy as np

# source: https://stackoverflow.com/questions/58427391/how-to-add-24-rows-under-each-row-in-python-pandas-dataframe 



N = 1 # add 1 row - aprox 86 Hz:
    
preproc_et.index = preproc_et.index * (N + 1)
preproc_et = preproc_et.reindex(np.arange(preproc_et.index.max() + N + 1))
print (preproc_et.head())


# interpolate rows with NaN
preproc_interp = preproc_et.interpolate()



preproc_interp_no_GTS = preproc_interp.drop(columns=['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])


# merge columns of df preproc_et (GTs with NaN) in the preproc_interp_no_GTS df. After that, fill the NAN values in columns GT1 to GT7 with the values of the previous column
# do it for each observer.

GT1_GT7 = preproc_et[['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]

# merge GT1_GT7 in preproc_interp_no_GTS based on index (there is no problem if there is no index, its the same df that i splited and wat to join again)

preproc_merged = preproc_interp_no_GTS.merge(GT1_GT7, left_index=True, right_index=True, how='left')

# Fill NaN from GT to GT7 with previous row values
preproc_merged[['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']] = preproc_merged[
    ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
].fillna(method='ffill')

print(preproc_merged.head())



    
# Drop all rows where column 'observer' is not an integer. 

preproc_merged_drop = preproc_merged.drop(columns=['coordinates', 'time_diff']) # create new columns time_diff because interval now is different


# Check the data type of the 'observer' column
print(preproc_merged['observer'].dtype) # float 64



# Remove rows where the observer value is a float and its decimal part is not 0
preproc_merged_no_float_zero = preproc_merged[preproc_merged['observer'] % 1 == 0] # example row 4031

# Print the resulting dataframe
print(preproc_merged_no_float_zero.head())


# Reset the index (optional)
preproc_merged_no_float_zero.reset_index(drop=True, inplace=True)

# Print the result
print(preproc_merged_no_float_zero.head())



output_file_86Hz = os.path.join(script_dir, config["preprocessed_data_86Hz_file"])
preproc_merged_no_float_zero.to_csv(output_file_86Hz, index=False)




## if __name__ == "__main__":
    
#    
    





