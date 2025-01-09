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

# import Aa01_funcs_extracting_features_C_gitignore as funcs


with open('config.json') as json_file:
    config = json.load(json_file)

with open('params.json') as params_file:
    params = json.load(params_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
# extracted_features = pd.read_csv(config["only_extracted_features_file_and_GTs"])
extracted_features = pd.read_csv(config["only_extracted_features_and_GTs_86Hz_file"])


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
# X_train_output_file = os.path.join(script_dir, config["X_train_data_file"])
# X_test_output_file = os.path.join(script_dir, config["X_test_data_file"])

X_train_output_file = os.path.join(script_dir, config["X_train_data_86Hz_file"])
X_test_output_file = os.path.join(script_dir, config["X_test_data_86Hz_file"])

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
    # y_train_output_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
    # y_test_output_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")
    
    y_train_output_file = os.path.join(script_dir, f"{config['y_train_data_86Hz_file']}_GT{i}.csv")
    y_test_output_file = os.path.join(script_dir, f"{config['y_test_data_86Hz_file']}_GT{i}.csv")
    
    print(f" y train and test of GT{i} saved")

    y_GT_train.to_csv(y_train_output_file, index=False)
    y_GT_test.to_csv(y_test_output_file, index=False)
    
    #convert in numpy 1D arrays for RF model
    y_train_dict[i] = y_GT_train.values #.ravel()
    y_test_dict[i] = y_GT_test.values#.ravel()
    


### feature importace z!!!

### RANDOM FOREST:

# for i in range(1,8):
    
#     print(f"Training Random Forest for GT{i}")
    
#     y_train = y_train_dict[i]
#     y_test = y_test_dict[i]
    
#     rfc=RandomForestClassifier()
#     param_grid = { 
#         'n_estimators': [10,12,15,25,50,100,150,200,250], # add 25  - final version (there is 25 in joep best param - selected GT1 and GT2)
#         'max_depth' : [1,2,3,4,5,6,7,8,9,10,11,12,15] # add 2, 7, 4 joep best param 2 - GT2 and 7 GT6 
#     }
#     CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
#     CV_rfc.fit(X_train, y_train)
#     print(f" for the ground truth GT{i} the best parameters are: {CV_rfc.best_params_}")

  
    
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



