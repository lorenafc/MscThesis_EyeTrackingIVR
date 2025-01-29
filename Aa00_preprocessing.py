# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""

import os
import pandas as pd
import numpy as np
import json
import Aa00_funcs_preprocessing as funcs
import Aa01_funcs_extracting_features as funcs_feat


with open('config.json') as json_file:
    config = json.load(json_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])


eye_tracking_data = pd.read_csv(path_file)

#### Add here the frequency preprocessing

#################### MOVE FREQUENCY FOR THE FIRST THING IN PREPROCESSING

###################### 0 (#12) FREQUENCY ############################

######################### CREATE DIFFERENT DATASETS, EACH OBSERVER A DIFFERENT FREQUENCY #################################

## The observers cannot be split in the sequency for each frequency, otherwise it can hard the cross validation, which is in sequence. And higher frequency
#will have much more wors than the given frequency

# VARJO-3 and VARJO-4 - 100hZ and 200 Hz : https://developer.varjo.com/docs/get-started/eye-tracking-with-varjo-headset 
# HTC VIVE Pro Eye - 120Hz  - Gaze data output frequency (binocular) - Interface HTC SRanipal SDK # https://www.tobii.com/products/integration/xr-headsets/device-integrations/htc-vive-pro-eye 



obs_freq = funcs_feat.select_observer_freq()
et_data_time_diff_freq = funcs_prep.calc_time_diff_freq(eye_tracking_data)


### ROWS 44HZ ORIGINAL



rows_obs_44Hz = eye_tracking_data[eye_tracking_data["observer"].isin(obs_freq['freq_N0_44Hz'])]  # '44Hz': [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51]                          
# dont interpolate original data
et_data_44Hz_original = rows_obs_44Hz.copy()

# et_data_44Hz_original = pd.read_csv('data/Aa01_et_data_44Hz_original_split_observers_frequency.csv')
et_data_44Hz_original["count_freq"] = range(1, len(et_data_44Hz_original) + 1) # https://stataiml.com/posts/add_increment_number_pandas/



funcs_feat.save_df(et_data_44Hz_original, "data/Aa01_et_data_44Hz_original_split_observers_frequency.csv") 

## ROWS 87 HZ    

                 
rows_obs_87Hz = eye_tracking_data[eye_tracking_data["observer"].isin(obs_freq["freq_N1_87Hz"])]  # '87Hz': [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52]                         
et_data_87Hz_reset_index = funcs_feat.interpolate_and_GTs_ff_reset_index(rows_obs_87Hz, ["time",'L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z'], 1 ) # # source: https://stackoverflow.com/questions/58427391/how-to-add-24-rows-under-each-row-in-python-pandas-dataframe 

# et_data_87Hz_reset_index = pd.read_csv("data/Aa01_et_data_87Hz_reset_index.csv")
et_data_87Hz_reset_index["count_freq"] = range(1, len(et_data_87Hz_reset_index) + 1) #  # len(et_data_44Hz_original) + 2 https://stataiml.com/posts/add_increment_number_pandas/ 

funcs_feat.save_df(et_data_87Hz_reset_index, "data/Aa01_et_data_87Hz_reset_index.csv")

### ROWS 130HZ
rows_obs_130Hz = eye_tracking_data[eye_tracking_data["observer"].isin(obs_freq['freq_N2_130Hz'])]  # '130Hz': [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53]                    
et_data_130Hz_reset_index = funcs_feat.interpolate_and_GTs_ff_reset_index(rows_obs_130Hz, ["time",'L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z'], 2)  


# et_data_130Hz_reset_index = pd.read_csv("data/Aa01_et_data_130Hz_reset_index.csv")
et_data_130Hz_reset_index["count_freq"] = range(1, len(et_data_130Hz_reset_index) + 1) # https://stataiml.com/posts/add_increment_number_pandas/


funcs_feat.save_df(et_data_130Hz_reset_index, "data/Aa01_et_data_130Hz_reset_index.csv")

### ROWS 174HZ

rows_obs_174Hz = eye_tracking_data[eye_tracking_data["observer"].isin(obs_freq['freq_N3_174Hz'])]  # '174Hz': [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]                           
et_data_174Hz_reset_index = funcs_feat.interpolate_and_GTs_ff_reset_index(rows_obs_174Hz, ["time",'L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z'], 3)  

# et_data_174Hz_reset_index = pd.read_csv("data/Aa01_et_data_174Hz_reset_index.csv")
et_data_174Hz_reset_index["count_freq"] = range(1, len(et_data_174Hz_reset_index) + 1) # https://stataiml.com/posts/add_increment_number_pandas/


funcs_feat.save_df(et_data_174Hz_reset_index, "data/Aa01_et_data_174Hz_reset_index.csv")

### ROWS 217HZ

rows_obs_217Hz = eye_tracking_data[eye_tracking_data["observer"].isin(obs_freq['freq_N4_217Hz'])]  # '217Hz': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]                      
et_data_217Hz_reset_index = funcs_feat.interpolate_and_GTs_ff_reset_index(rows_obs_217Hz, ["time",'L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z'], 4)  



# et_data_217Hz_reset_index = pd.read_csv("data/Aa01_et_data_217Hz_reset_index.csv")
et_data_217Hz_reset_index["count_freq"] = range(1, len(et_data_217Hz_reset_index) + 1) # https://stataiml.com/posts/add_increment_number_pandas/

funcs_feat.save_df(et_data_217Hz_reset_index, "data/Aa01_et_data_217Hz_reset_index.csv")



## rename the colum time_diff of all of the rows (but you change the time_diff function so need to do it only once)

# Dictionary of DataFrames
df_dict = {
    "et_data_44Hz_original": et_data_44Hz_original,
    "et_data_87Hz_reset_index": et_data_87Hz_reset_index,
    "et_data_130Hz_reset_index": et_data_130Hz_reset_index,
    "et_data_174Hz_reset_index": et_data_174Hz_reset_index,
    "et_data_217Hz_reset_index": et_data_217Hz_reset_index
}

# Rename "time_diff" in each DataFrame
df_dict = {key: df.rename(columns={"time_diff": "time_diff_freq_NaN"}) for key, df in df_dict.items()}

# Access updated DataFrames
et_data_44Hz_original = df_dict["et_data_44Hz_original"]
et_data_87Hz_reset_index = df_dict["et_data_87Hz_reset_index"]
et_data_130Hz_reset_index = df_dict["et_data_130Hz_reset_index"]
et_data_174Hz_reset_index = df_dict["et_data_174Hz_reset_index"]
et_data_217Hz_reset_index = df_dict["et_data_217Hz_reset_index"]

funcs_feat.save_df(et_data_44Hz_original, "data/Aa01_et_data_44Hz_original_split_observers_frequency.csv") 
funcs_feat.save_df(et_data_87Hz_reset_index, "data/Aa01_et_data_87Hz_reset_index.csv")
funcs_feat.save_df(et_data_130Hz_reset_index, "data/Aa01_et_data_130Hz_reset_index.csv")
funcs_feat.save_df(et_data_174Hz_reset_index, "data/Aa01_et_data_174Hz_reset_index.csv")
funcs_feat.save_df(et_data_217Hz_reset_index, "data/Aa01_et_data_217Hz_reset_index.csv")





## - REMOVE IT TO THE END OF FEATURE EXTRACTION - CONCAT THEM ALL TOGETHER - needs to be done after extracting all features - because of the windows.

vertical_concat_all_wrong_order_observ = pd.concat([et_data_44Hz_original, et_data_87Hz_reset_index, et_data_130Hz_reset_index, et_data_174Hz_reset_index, et_data_217Hz_reset_index], axis=0, ignore_index=True)   
# et_data_concat_observer_right_order = vertical_concat_all_wrong_order_observ.sort_values(by=['observer'])

vertical_concat_all_wrong_reset = vertical_concat_all_wrong_order_observ.reset_index()
et_data_concat_observer_right_order = vertical_concat_all_wrong_reset.sort_values(['observer', 'index'], ascending=[True, True]) # https://stackoverflow.com/questions/17141558/how-to-sort-a-pandas-dataframe-by-two-or-more-columns

et_data_concat_observer_right_order = et_data_concat_observer_right_order.rename(columns={"time_diff": "time_diff_freq_NaN"})
# eye_tracking_data = eye_tracking_data.drop(columns="") better rename than drop to track the NaN

funcs_feat.save_df(et_data_concat_observer_right_order, "data/Aa01_et_data_concat_observer_right_order_count_freq_checked.csv")


preprocessed_et_data = et_data_concat_observer_right_order.copy()


#### Preprocess and extract features of the 5- diff frequencies separately:

    
eye_tracking_data = pd.read_csv("data/Aa01_et_data_concat_observer_right_order_count_freq_checked.csv")
 
eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)
funcs_feat.save_df(eye_tracking_data_cm2deg_new, "data/Aa00_preprocessed_eye_tracking_freq.csv")
funcs_feat.save_df(eye_tracking_data_cm2deg_new, config["preprocessed_data_file"])


#### preprocess all the 5 df frequencies:
    
## optimize the  preprocessing with the code below code messes the columns coordinates_dist and cm_to_deg_inside_VE with lots of NaN
# freqs = [44, 87,130, 174, 217]
# df_dict = {
#     44: et_data_44Hz_original,
#     87: et_data_87Hz_reset_index,  
#     130: et_data_130Hz_reset_index,
#     174: et_data_174Hz_reset_index,
#     217: et_data_217Hz_reset_index
# }


# for freq, df in df_dict.items():
#     et_cm2deg = funcs.process_eye_tracking_data(df)
#     filename = f"data/Aa00_preprocessed_eye_tracking_{freq}Hz.csv"
#     funcs_feat.save_df(et_cm2deg, filename)
  
    
et_data_44Hz_original = pd.read_csv("data/Aa01_et_data_44Hz_original_split_observers_frequency.csv")
et_44Hz_cm2deg = funcs.process_eye_tracking_data(et_data_44Hz_original)
funcs_feat.save_df(et_44Hz_cm2deg, "data/Aa00_preprocessed_eye_tracking_44Hz.csv")

et_data_87Hz_reset_index = pd.read_csv("data/Aa01_et_data_87Hz_reset_index.csv")
et_87Hz_cm2deg = funcs.process_eye_tracking_data(et_data_87Hz_reset_index)
funcs_feat.save_df(et_87Hz_cm2deg, "data/Aa00_preprocessed_eye_tracking_87Hz.csv")

et_data_130Hz_reset_index = pd.read_csv("data/Aa01_et_data_130Hz_reset_index.csv")
et_130Hz_cm2deg = funcs.process_eye_tracking_data(et_data_130Hz_reset_index)
funcs_feat.save_df(et_130Hz_cm2deg, "data/Aa00_preprocessed_eye_tracking_130Hz.csv")

et_data_174Hz_reset_index = pd.read_csv("data/Aa01_et_data_174Hz_reset_index.csv")
et_174Hz_cm2deg = funcs.process_eye_tracking_data(et_data_174Hz_reset_index)
funcs_feat.save_df(et_174Hz_cm2deg, "data/Aa00_preprocessed_eye_tracking_174Hz.csv")

et_data_217Hz_reset_index = pd.read_csv("data/Aa01_et_data_217Hz_reset_index.csv")
et_217Hz_cm2deg = funcs.process_eye_tracking_data(et_data_217Hz_reset_index)
funcs_feat.save_df(et_217Hz_cm2deg, "data/Aa00_preprocessed_eye_tracking_217Hz.csv")


# import numpy as np

# # source: https://stackoverflow.com/questions/58427391/how-to-add-24-rows-under-each-row-in-python-pandas-dataframe 



# N = 1 # add 1 row - aprox 86 Hz: # N - number of empty rows created
    
# preproc_et.index = preproc_et.index * (N + 1) 
# preproc_et = preproc_et.reindex(np.arange(preproc_et.index.max() + N + 1))
# print (preproc_et.head())


# # interpolate rows with NaN
# preproc_interp = preproc_et.interpolate()



# preproc_interp_no_GTS = preproc_interp.drop(columns=['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']) # drop all the columns to make the calculation=====================================================================================


# # merge columns of df preproc_et (GTs with NaN) in the preproc_interp_no_GTS df. After that, fill the NAN values in columns GT1 to GT7 with the values of the previous column
# # do it for each observer.

# GT1_GT7 = preproc_et[['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]

# # merge GT1_GT7 in preproc_interp_no_GTS based on index (there is no problem if there is no index, its the same df that i splited and wat to join again)

# preproc_merged = preproc_interp_no_GTS.merge(GT1_GT7, left_index=True, right_index=True, how='left')

# # Fill NaN from GT to GT7 with previous row values
# preproc_merged[['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']] = preproc_merged[
#     ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
# ].fillna(method='ffill')

# print(preproc_merged.head())



    
# # Drop all rows where column 'observer' is not an integer. 

# preproc_merged_drop = preproc_merged.drop(columns=['coordinates', 'time_diff']) # create new columns time_diff because interval now is different


# # Check the data type of the 'observer' column
# print(preproc_merged['observer'].dtype) # float 64



# # Remove rows where the observer value is a float and its decimal part is not 0
# preproc_merged_no_float_zero = preproc_merged[preproc_merged['observer'] % 1 == 0] # example row 4031

# # Print the resulting dataframe
# print(preproc_merged_no_float_zero.head())


# # Reset the index (optional)
# preproc_merged_no_float_zero.reset_index(drop=True, inplace=True)

# # Print the result
# print(preproc_merged_no_float_zero.head())



# output_file_86Hz = os.path.join(script_dir, config["preprocessed_data_86Hz_file"])
# preproc_merged_no_float_zero.to_csv(output_file_86Hz, index=False)




## if __name__ == "__main__":
    
#    
    





