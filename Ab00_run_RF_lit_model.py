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
extracted_features = pd.read_csv(config["extracted_features_file"])



print("teste Ab00")





# X train and X test without GT1


eye_tracking_data_without_GT1 = extracted_features.drop(columns=['GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])
eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['time', 'observer', 'coordinates', 'coordinates_dist','cm_to_deg_inside_VE'])

## only acceleration and velocity:   
eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'viewing_distance','time_diff'])

train_split = params["training"]["train_test_split"] #0.75

# Creating data indices for training and test splits: source: LSTM autoencoder time series https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/blob/main/code/main.py
dataset_size = len(eye_tracking_data_without_GT1)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))

X_et_train_without_GT1 = eye_tracking_data_without_GT1.iloc[:split, :]
X_et_test_without_GT1 = eye_tracking_data_without_GT1.iloc[split:, :]

print("et_train_without_GT1 shape:", X_et_train_without_GT1.shape)

X_train = X_et_train_without_GT1.values
X_test = X_et_test_without_GT1.values


## y - label - GT1 - just to check, after do with all labels!!!


y_train_dict = {}
y_test_dict = {}

for i in range (1,8):
    gt_col = f"GT{i}"

    y_GT = extracted_features[[gt_col]]
    
    y_GT_train = y_GT.iloc[:split, :]
    y_GT_test = y_GT.iloc[split:, :]
    
    y_train_dict[i] = y_GT_train.values.ravel()
    y_test_dict[i] = y_GT_test.values.ravel()
    

### RANDOM FOREST:

for i in range(1,8):
    
    print(f"Training Random Forest for GT{i}")
    
    y_train = y_train_dict[i]
    y_test = y_test_dict[i]
    
    rfc=RandomForestClassifier()
    param_grid = { 
        'n_estimators': [10,12,14,16,20,50,100,150,200,250], # add 25  - final version (there is 25 in joep best param - selected GT1 and GT2)
        'max_depth' : [1,3,6,9,10,11,12,15] # add 2, 7, 4 joep best param 2 - GT2 and 7 GT6 
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)
    
    print(f" for the ground truth GT{i} the best parameters are: {CV_rfc.best_params_}")
    
    
    best_params = CV_rfc.best_params_
    
    rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], n_jobs=-1) # replave by params.json instead of best params #GT1  max dep 6, n est 50.
    rf.fit(X_train, y_train)
    
    train_accuracy = rf.score(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)
    print(f"GT{i} - Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"GT{i} - Test Accuracy: {test_accuracy * 100:.2f}%")
    
    y_pred = rf.predict(X_test)
    print(f"Classification Report GT{i}:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix GT{i}:\n", confusion_matrix(y_test, y_pred))

# using 2 features - velocity and acceleration:
    
# GT1 - {'max_depth': 6, 'n_estimators': 50} f1 = 82% - 1 - fixation, 75% 0 - undefined
# GT1 - Train Accuracy: 78.32%
# GT1 - Test Accuracy: 78.94%

# GT2 - {'max_depth': 6, 'n_estimators': 50} f1 = 76% - 1 - fixation, 79% 0 - undefined
# GT2 - Train Accuracy: 76.71%
# GT2 - Test Accuracy: 77.69%


# GT3 - {'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 74% 0 - undefined
#GT3 - Train Accuracy: 76.78%
#GT3 - Test Accuracy: 77.82%

# GT4 - {'max_depth': 6, 'n_estimators': 50} f1 =  80 % - 1 - fixation, 72 % 0 - undefined
# GT4 - Train Accuracy: 75.81%
# GT4 - Test Accuracy: 76.48%


# GT5 -{'max_depth': 6, 'n_estimators': 200} f1 = 80% - 1 - fixation, 71% 0 - undefined
# GT5 - Train Accuracy: 76.01%
# GT5 - Test Accuracy: 76.40%


# GT6 -{'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 70% 0 - undefined
# GT6 - Train Accuracy: 76.16%
# GT6 - Test Accuracy: 76.45%



# GT7 -{'max_depth': 6, 'n_estimators': 200} f1 = 78% - 1 - fixation, 72% 0 - undefined
# GT7 - Train Accuracy: 74.85%
# GT7 - Test Accuracy: 75.38%



### feature importace and divide in 100 Hz!!!
    

###### UPDATE BEST PARAMS IN JSON FILE

# print(f"Best Parameters: {best_params}")

# #Update the parameters file # ChatGPT
# params["random-forest"]["n_estimators"] = best_params["n_estimators"]
# params["random-forest"]["max_depth"] = best_params["max_depth"]

# with open('params.json', 'w') as json_file:
#     json.dump(params, json_file, indent=4)


#USE BEST PARAMS:
#rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
#rfc1.fit(x_train, y_train)


# RF with the best parameters: 
    
#param_grid = { 'n_estimators': range(10, 60, 10),'max_features': ['auto', 'sqrt', 'log2'],
    #'max_depth' : [3,6,9,12,15],
    #'criterion' :['gini', 'entropy']
    # features: ['L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'viewing_distance',
          # 'time_diff', 'coordinates_dist', 'cm_to_deg_inside_VE'],
        
# Train Accuracy n est 10 max_depth 3: 78.38%
# Test Accuracy n est 10 max_depth 3: 78.90%

## with acceleration and velocity:
# columns : ['L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'viewing_distance',
       #'time_diff', 'velocity_deg_s', 'acceler_deg_s']
#Train Accuracy: 78.36%
#Test Accuracy: 79.18%



##ONLY VELOCITY AND ACC DEG/S
#Train Accuracy: 78.31% n est 16 max depth 6
#Test Accuracy: 78.96%


# rf = RandomForestClassifier(n_estimators=params["gt[i]"]["random_forest"]["n_estimators"] , max_depth=params["random_forest"]["max_depth"], n_jobs=-1) # params["random_forest"]["n_estimators"] n_est 16, max_dep = 6  # I want these values to be retrieved from the CV_rfc.best_params
# rf.fit(X_train, y_train) # #source: https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected

# print(f'Train Accuracy: {rf.score(X_train, y_train) * 100:.2f}%')
# print(f'Test Accuracy: {rf.score(X_test, y_test) * 100:.2f}%')

    
# from sklearn.metrics import classification_report, confusion_matrix

# y_pred = rf.predict(X_test)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


""" f1-score = 0.82 (1 fixation), 0.75 (0 no fixation)

Class 0 (non-fixation):

Precision = 0.80: 80% of the predictions for non-fixation are correct.
Recall = 0.71: 71% of the actual non-fixations are identified by the model.
F1-Score = 0.75:

Class 1 (fixation):

Precision = 0.78: 78% of the predicted fixations are correct.
Recall = 0.85: 85% of the actual fixations are correctly identified.
F1-Score = 0.82:

True Negatives (TN): Correctly identified non-fixations (8545).
False Positives (FP): Non-fixations incorrectly identified as fixations (3407).
False Negatives (FN): Fixations incorrectly identified as non-fixations (2194).
True Positives (TP): Correctly identified fixations (12417).


""" 