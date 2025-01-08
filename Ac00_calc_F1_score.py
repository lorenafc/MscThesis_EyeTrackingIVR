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
X_test_file = os.path.join(script_dir, config["X_test_data_file"])
X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)


y_train_dict = {}
y_test_dict = {}

for i in range(1, 8):  
    y_train_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
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
    
    y_train = y_train_dict[i]
    y_test = y_test_dict[i]

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