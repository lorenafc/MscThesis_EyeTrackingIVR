# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:46:58 2025

@author: carpe
"""


import pandas as pd
import numpy as np
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# import Aa01_funcs_extracting_features_C_gitignore as funcs


with open('config.json') as json_file:
    config = json.load(json_file)

with open('params.json') as params_file:
    params = json.load(params_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

X_train_file = os.path.join(script_dir, config["X_train_data_file"])
            # X_train_file = os.path.join(script_dir, config["X_train_data_86Hz_file"])

            # X_test_file = os.path.join(script_dir, config["X_test_data_86Hz_file"]) # X test must be original. not the 86Hz
X_test_file = os.path.join(script_dir, config["X_test_data_file"])


X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)

#convert df in numpy arrays for RF model
X_train = X_train #.values
X_test = X_test #.values

y_train_dict = {}
y_test_dict = {}

for i in range(1, 8):  
    y_train_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
                # y_train_file = os.path.join(script_dir, f"{config['y_train_data_86Hz_file']}_GT{i}.csv")
    
                # y_test_file = os.path.join(script_dir, f"{config['y_test_data_86Hz_file']}_GT{i}.csv") # Y test must be original. not the 86Hz
    y_test_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")
    
    
    y_train_dict[i] = pd.read_csv(y_train_file)
    y_test_dict[i] = pd.read_csv(y_test_file)
    

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

for i in range(1, 8):
    print(f"GT{i} y_train shape:", y_train_dict[i].shape)
    print(f"GT{i} y_test shape:", y_test_dict[i].shape)



# calculate F1:
    
for i in range(1,8):
    
    #convert df to array and then 1D arrays for RF model
    y_train = y_train_dict[i].values.ravel()
    y_test = y_test_dict[i].values.ravel()
    
    n_estimators = params[f"gt{i}"]['random_forest']['n_estimators']
    max_depth = params[f"gt{i}"]['random_forest']['max_depth']
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, n_jobs=-1) 
    rf.fit(X_train, y_train)
    
    train_accuracy = rf.score(X_train, y_train)
    test_accuracy = rf.score(X_test, y_test)
    print(f"GT{i} - Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"GT{i} - Test Accuracy: {test_accuracy * 100:.2f}%")
    
    y_pred = rf.predict(X_test)
    print(f"Classification Report GT{i}:\n", classification_report(y_test, y_pred))
    print(f"Confusion Matrix GT{i}:\n", confusion_matrix(y_test, y_pred))
    
    
    
    #####################

# using 2 features only - velocity and acceleration:
# for the 86Hz I use the best params of the 44Hz because the code take too many hours too long, so 86Hz best params could be a different value, but it outperfors the original dataset
# f1 original 79.71 %, f1 86Hz: 80.14 % 
    
# GT1 - {'max_depth': 6, 'n_estimators': 50} f1 = 82% - 1 - fixation, 75% 0 - undefined
# GT1 - Train Accuracy: 78.32%
# GT1 - Test Accuracy: 78.94%

#- 86Hz -  but for test I used the original dataset.

# GT1 - {'max_depth': 6, 'n_estimators': 50} f1 = 83% - 1 - fixation, 77% 0 - undefined
# GT1 - Train Accuracy: 79.48%
# GT1 - Test Accuracy: 78.96%

# including mean_dist_m inthe original dataset:

# GT1 - {'max_depth': 6, 'n_estimators': 50} f1 = 83% - 1 - fixation, 77% 0 - undefined    
# GT1 - Train Accuracy: 82.36%
# GT1 - Test Accuracy: 82.48%


# using mean_diff degree original dataset:
    
# GT1 - {'max_depth': 6, 'n_estimators': 50} f1 = 85% - 1 - fixation, 81% 0 - undefined  
# GT1 - Train Accuracy: 82.68%
# GT1 - Test Accuracy: 82.89%

########

# GT2 - {'max_depth': 6, 'n_estimators': 50} f1 = 76% - 1 - fixation, 79% 0 - undefined
# GT2 - Train Accuracy: 76.71%
# GT2 - Test Accuracy: 77.69%

# 86Hz-  but for test I used the original dataset.
# GT2 - {'max_depth': 6, 'n_estimators': 50} f1 = 79% - 1 - fixation, 76% 0 - undefined    
# GT2 - Train Accuracy: 77.90%
# GT2 - Test Accuracy: 77.72%


# using mean_diff degree original dataset:
    
# GT2 - {'max_depth': 6, 'n_estimators': 50} f1 = 83% - 1 - fixation, 82% 0 - undefined  
# GT2 - Train Accuracy: 81.79%
# GT2 - Test Accuracy: 82.45%

#######
# GT3 - {'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 74% 0 - undefined
#GT3 - Train Accuracy: 76.78%
#GT3 - Test Accuracy: 77.82%

#86Hz -  but for test I used the original dataset.
# GT3 - {'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 73% 0 - undefined
# GT3 - Train Accuracy: 77.89%
# GT3 - Test Accuracy: 77.73%

# using mean_diff degree original dataset:
    
# GT3 - {'max_depth': 6, 'n_estimators': 50} f1 = 84% - 1 - fixation, 80% 0 - undefined  
# GT3 - Train Accuracy: 81.51%
# GT3 - Test Accuracy: 81.90%


########

# GT4 - {'max_depth': 6, 'n_estimators': 50} f1 =  82 % - 1 - fixation, 79 % 0 - undefined
# GT4 - Train Accuracy: 75.81%
# GT4 - Test Accuracy: 76.48%

#86Hz - but for test I used the original dataset.

# GT4 - {'max_depth': 6, 'n_estimators': 50} f1 =  80 % - 1 - fixation, 72 % 0 - undefined
# GT4 - Train Accuracy: 76.82%
# GT4 - Test Accuracy: 76.42%

# using mean_diff degree original dataset:
    
# GT4 - {'max_depth': 6, 'n_estimators': 50} f1 = 80% - 1 - fixation, 84% 0 - undefined  
# GT4 - Train Accuracy: 80.98%
# GT4 - Test Accuracy: 80.89%

#######

# GT5 -{'max_depth': 6, 'n_estimators': 200} f1 = 80% - 1 - fixation, 71% 0 - undefined
# GT5 - Train Accuracy: 76.01%
# GT5 - Test Accuracy: 76.40%

#86Hz - but test original dataset.

# GT5 -{'max_depth': 6, 'n_estimators': 200} f1 = 80% - 1 - fixation, 71% 0 - undefined
# GT5 - Train Accuracy: 76.91%
# GT5 - Test Accuracy: 76.32%

# using mean_diff degree original dataset:
    
# GT5 - {'max_depth': 6, 'n_estimators': 50} f1 = 83% - 1 - fixation, 77% 0 - undefined  
# GT5 - Train Accuracy: 80.85%
# GT5 - Test Accuracy: 80.77%


####

# GT6 -{'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 70% 0 - undefined
# GT6 - Train Accuracy: 76.16%
# GT6 - Test Accuracy: 76.45%

#86Hz - but test original dataset.

# GT6 -{'max_depth': 6, 'n_estimators': 200} f1 = 81% - 1 - fixation, 69% 0 - undefined
# GT6 - Train Accuracy: 77.16%
# GT6 - Test Accuracy: 76.32%

# using mean_diff degree original dataset:
    
# GT6 - {'max_depth': 6, 'n_estimators': 50} f1 = 84% - 1 - fixation, 76% 0 - undefined  
# GT6 - Train Accuracy: 81.08%
# GT6 - Test Accuracy: 80.66%


######



# GT7 -{'max_depth': 6, 'n_estimators': 200} f1 = 78% - 1 - fixation, 72% 0 - undefined
# GT7 - Train Accuracy: 74.85%
# GT7 - Test Accuracy: 75.38%

#86Hz - but test original dataset.

# GT7 -{'max_depth': 6, 'n_estimators': 200} f1 = 78% - 1 - fixation, 72% 0 - undefined
# GT7 - Train Accuracy: 75.91%
# GT7 - Test Accuracy: 75.45%


# using mean_diff degree original dataset:
    
# GT6 - {'max_depth': 6, 'n_estimators': 50} f1 = 81% - 1 - fixation, 79% 0 - undefined  
# GT7 - Train Accuracy: 80.29%
# GT7 - Test Accuracy: 80.30%

# dataset original - 44Hz
from statistics import mean 

f1 = [82,76,81,80,80,81,78]
average_f1_extracted_features = mean(f1)
print(round(average_f1_extracted_features,2),"%") # 79.71 % joep  μsF1 0.753 - No extracted features


# dataset interpolated- 86Hz
from statistics import mean 

f1 = [82,79,81,80,80,81,78]
average_f1_extracted_features = mean(f1)
print(round(average_f1_extracted_features,2),"%") # 80.14 % joep  μsF1 0.753  - No extracted features

# dataset original vel, acc and mean dist m:
    
f1 = [84,82,84,82,83,84,81]
average_f1_extracted_features = mean(f1)
print(round(average_f1_extracted_features,2),"%") # 82.86 % joep  μsF1 0.753 - No extracted features 

# dataset original vel, acc and MEAN DIFF:
from statistics import mean    
f1 = [85,83,84,82,83,84,81]
average_f1_extracted_features = mean(f1)
print(round(average_f1_extracted_features,2),"%") # 83.14 % joep  μsF1 0.753 - No extracted features  





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

# Dataset 86Hz 