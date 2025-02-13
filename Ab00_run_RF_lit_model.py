# -*- coding: utf-8 -*-
"""
Created on Tue Jan 7 2025

@author: Lorena Carpes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
import Aa01_funcs_extracting_features as funcs_feat

# import Aa01_funcs_extracting_features_C_gitignore as funcs


import sys
print(sys.executable)
print(sys.path)


with open('config.json') as json_file:
    config = json.load(json_file)

with open('params.json') as params_file:
    params = json.load(params_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

# path_file = os.path.join(script_dir, config["data_path"])
# extracted_features = pd.read_csv(config["only_extracted_features_file_and_GTs"])
# extracted_features = pd.read_csv(config["only_extracted_features_and_GTs_86Hz_file"]) #86Hz


extracted_features = pd.read_csv( "data/RF/prA01_et_data_all_features_extracted_all_freq_CONCAT_RF_rem_23.csv") #/RF/Aa01_test_xy_yz_zx_rf.csv") # bcea_diff 3d  degree added
# # extracted_features_add_feature = pd.read_csv("data/Aa01_test_only_bcea_yz_3d_GTs_rf.csv")
extracted_features = extracted_features[['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', 'disp_degree',
       'med_diff_deg', 'std_deg', 'std_diff_deg', 'rms_diff_deg', 'rms_deg',
       'bcea_2d_yz_deg', 'bcea_diff_deg', 'GT1', 'GT2', 'GT3',
       'GT4', 'GT5', 'GT6', 'GT7']] #'bcea_L_xL_y', 'bcea_L_zL_x', 
##removed xz an xy

### REMOVE OUTLIERS!!!!

Q1 = extracted_features.quantile(0.25)
Q3 = extracted_features.quantile(0.75) # 22% data droped  - 83508 remaining (no bcea xz and xy)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR


print(f'Original data shape: {extracted_features.shape}')

# Filter out outliers
extracted_features = extracted_features[(extracted_features >= lower_bound) & (extracted_features <= upper_bound)].dropna()

print(f'Data shape after removing outliers: {extracted_features.shape}')



# # Define the columns to check for outliers
# columns_to_check = [
#     'velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', 'med_diff_deg',
#            'disp_degree', 'std_deg',  'bcea_L_yL_z', 'bcea_diff_deg'
# ]

# # Calculate Q1, Q3, and IQR for the selected columns
# Q1 = extracted_features[columns_to_check].quantile(0.25)
# Q3 = extracted_features[columns_to_check].quantile(0.75)
# IQR = Q3 - Q1

# # Define outlier bounds with a larger multiplier
# lower_bound = Q1 - 3.0 * IQR  # Use 5.0 instead of 3.0
# upper_bound = Q3 + 3.0 * IQR

# # Debugging: Print bounds and check for missing values
# print("Lower Bound:")
# print(lower_bound)
# print("\nUpper Bound:")
# print(upper_bound)
# print("\nMissing values in each column:")
# print(extracted_features[columns_to_check].isnull().sum())

# # Drop rows with missing values in the selected columns
# extracted_features_clean = extracted_features.dropna(subset=columns_to_check)

# # Filter out outliers (only for the selected columns)
# mask = (extracted_features_clean[columns_to_check] >= lower_bound) & (extracted_features_clean[columns_to_check] <= upper_bound)
# extracted_features_no_out = extracted_features_clean[mask.all(axis=1)]

# print(f'Original data shape: {extracted_features.shape}')
# print(f'Data shape after removing outliers: {extracted_features_no_out.shape}') # 24% dropped out - 781083



####### REMEMBER TO RECALCULATE THE BEST PARAMS WHEN YOU USE ZEMBLYS STUDY!!!

# bcea_yz_only_rf = extracted_features.copy()

# # preprocessing + 3d +yz
# bcea_yz_3d_noise_rf = pd.concat([extracted_features, three_D_noise["bcea_3d_noise"]], axis=1)

# extracted_features = bcea_yz_3d_noise_rf.copy()

# XY_YZ_ZX.head()
# extracted_features = extracted_features.drop(columns=["bcea_L_yL_z.1"])


###### X train and X test without GT1


eye_tracking_data_without_GT1 = extracted_features.drop(columns=['GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])
train_split = params["training"]["train_test_split"] #0.75

# training and test splits  source: LSTM autoencoder time series https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/blob/main/code/main.py
dataset_size = len(eye_tracking_data_without_GT1)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))

X_et_train_without_GT1 = eye_tracking_data_without_GT1.iloc[:split, :]
X_et_test_without_GT1 = eye_tracking_data_without_GT1.iloc[split:, :]

# save X train and X test dfs
X_train_output_file = os.path.join(script_dir, config["X_train_data_file"])
X_test_output_file = os.path.join(script_dir, config["X_test_data_file"])

# X_train_output_file = os.path.join(script_dir, config["X_train_data_86Hz_file"])
# X_test_output_file = os.path.join(script_dir, config["X_test_data_86Hz_file"])

print("X train and X test saved")

X_et_train_without_GT1.to_csv(X_train_output_file, index=False)
X_et_test_without_GT1.to_csv(X_test_output_file, index=False)

#convert df in numpy arrays for RF model
X_train = X_et_train_without_GT1.values
X_test = X_et_test_without_GT1.values


## y - label - GT


y_train_dict = {}
y_test_dict = {}

for i in range (1,8):
    gt_col = f"GT{i}"

    y_GT = extracted_features[[gt_col]]
    
    y_GT_train = y_GT.iloc[:split, :]
    y_GT_test = y_GT.iloc[split:, :]
    
    # save y train and y test GT1 to GT7 dfs
    y_train_output_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
    y_test_output_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")
    
    # y_train_output_file = os.path.join(script_dir, f"{config['y_train_data_86Hz_file']}_GT{i}.csv") # put a generic name test for testing other parameters/stuff
    # y_test_output_file = os.path.join(script_dir, f"{config['y_test_data_86Hz_file']}_GT{i}.csv")
    
    print(f" y train and test of GT{i} saved")

    y_GT_train.to_csv(y_train_output_file, index=False)
    y_GT_test.to_csv(y_test_output_file, index=False)
    
    #convert in numpy 1D arrays for RF model
    y_train_dict[i] = y_GT_train.values.ravel() #.ravel() # if dont add ravel(), this message appears: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    y_test_dict[i] = y_GT_test.values.ravel()#.ravel()
    




# ## RANDOM FOREST - FIND BEST PARAMS:
# results = []

# for i in range(1,8):
    
#     print(f"Training Random Forest for GT{i}")
    
#     y_train = y_train_dict[i]
#     y_test = y_test_dict[i]
    
#     rfc=RandomForestClassifier()
    
# #     param_grid = { 
# #     'n_estimators': [10, 25],  
# #     'max_depth': range(1, 3)   # Adjusted the comment to avoid syntax error
# # }
    
    
#     param_grid = { 
#         'n_estimators': [10, 25, 50, 100, 200], # like joep and zemblys[10,12,15,25,50,100,150,200,250], # add 25  - # 10, 25, 50, 100, 200. final version (there is 25 in joep best param - selected GT1 and GT2)
#         'max_depth' : range(1,15,1) #change to 15  # like joep [1,2,3,4,5,6,7,8,9,10,11,12,15] # add 2, 7, 4 joep best param 2 - GT2 and 7 GT6 
#     }
#     CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#     CV_rfc.fit(X_train, y_train)
    
#     best_params = CV_rfc.best_params_
#     print(f" for the ground truth GT{i} the best parameters are: {CV_rfc.best_params_}")

#         # Append the results to the list
#     results.append({'Ground Truth': f'GT{i}', 'Best Parameters': best_params})

# # Convert the results list to a DataFrame
# results_df = pd.DataFrame(results)

# # Save the DataFrame to a CSV file
# results_df.to_csv('data/best_params_CONCAT_deg_rem23.csv', index=False)

# print("Best parameters have been saved to best_params.csv")







  
    # best_params = CV_rfc.best_params_
    
    # rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], n_jobs=-1) # replave by params.json instead of best params #GT1  max dep 6, n est 50.
    # rf.fit(X_train, y_train)
    
    # train_accuracy = rf.score(X_train, y_train)
    # test_accuracy = rf.score(X_test, y_test)
    # print(f"GT{i} - Train Accuracy: {train_accuracy * 100:.2f}%")
    # print(f"GT{i} - Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # y_pred = rf.predict(X_test)
    # print(f"Classification Report GT{i}:\n", classification_report(y_test, y_pred))
    # print(f"Confusion Matrix GT{i}:\n", confusion_matrix(y_test, y_pred))
    








###### UPDATE BEST PARAMS IN JSON FILE

# print(f"Best Parameters: {best_params}")

# #Update the parameters file # ChatGPT
# params["random-forest"]["n_estimators"] = best_params["n_estimators"]
# params["random-forest"]["max_depth"] = best_params["max_depth"]

# with open('params.json', 'w') as json_file:
#     json.dump(params, json_file, indent=4)



