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




# Get the output path from config
output_file = os.path.join(script_dir, config["preprocessed_data_file"])

# Save the processed data to a CSV file
eye_tracking_data_cm2deg_new.to_csv(output_file, index=False)


# def get_preprocessed_et_data(): 
#     eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)
#     return eye_tracking_data_cm2deg_new


# eye_tracking_data_cm2deg_new = get_preprocessed_et_data()

print(eye_tracking_data_cm2deg_new.head(2))


# if __name__ == "__main__":
    
#    eye_tracking_data_cm2deg_new
    




# X train and X test without GT1
import numpy as np

eye_tracking_data_without_GT1 = eye_tracking_data_cm2deg_new.drop(columns=['GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])

eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['time', 'observer', 'coordinates'])

eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['GT1'])



# Convert columns to float32 for compatibility with PyTorch
# eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.astype('float32')
train_split = 0.75

# Creating data indices for training and test splits: source: LSTM autoencoder time series https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/blob/main/code/main.py
dataset_size = len(eye_tracking_data_without_GT1)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))

X_et_train_without_GT1 = eye_tracking_data_without_GT1.iloc[:split, :]
X_et_test_without_GT1 = eye_tracking_data_without_GT1.iloc[split:, :]

print("et_train_without_GT1 shape:", X_et_train_without_GT1.shape)


## y - label - GT1 - just to check, after do with all labels!!!


y_GT1 = eye_tracking_data_cm2deg_new[['GT1']]
print(y_GT1.head(3))

y_GT1_train = y_GT1.iloc[:split, :]
y_GT1_test = y_GT1.iloc[split:, :]


y_train = y_GT1_train.values.ravel()
y_test = y_GT1_test.values.ravel()


X_train = X_et_train_without_GT1.values
X_test = X_et_test_without_GT1.values


### RANDOM FOREST:
    
#### USE GRID SEARCH!!!!!
import sklearn

from sklearn.ensemble import RandomForestClassifier
    

# # X train and X test without GT1
# import numpy as np

# eye_tracking_data_without_GT1 = eye_tracking_data_cm2deg_new.drop(columns=['GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])

# eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['time', 'observer', 'coordinates'])

# eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.drop(columns=['GT1'])



# # Convert columns to float32 for compatibility with PyTorch
# # eye_tracking_data_without_GT1 = eye_tracking_data_without_GT1.astype('float32')
# train_split = 0.75

# # Creating data indices for training and test splits: source: LSTM autoencoder time series https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series/blob/main/code/main.py
# dataset_size = len(eye_tracking_data_without_GT1)
# indices = list(range(dataset_size))
# split = int(np.floor(train_split * dataset_size))

# X_et_train_without_GT1 = eye_tracking_data_without_GT1.iloc[:split, :]
# X_et_test_without_GT1 = eye_tracking_data_without_GT1.iloc[split:, :]

# print("et_train_without_GT1 shape:", X_et_train_without_GT1.shape)


# ## y - label - GT1 - just to check, after do with all labels!!!


# y_GT1 = eye_tracking_data_cm2deg_new[['GT1']]
# print(y_GT1.head(3))

# y_GT1_train = y_GT1.iloc[:split, :]
# y_GT1_test = y_GT1.iloc[split:, :]


# y_train = y_GT1_train.values.ravel()
# y_test = y_GT1_test.values.ravel()


# X_train = X_et_train_without_GT1.values
# X_test = X_et_test_without_GT1.values


### RANDOM FOREST:
    
#### USE GRID SEARCH!!!!!
import sklearn

from sklearn.ensemble import RandomForestClassifier
    
# Random Forest Classifier - USING OVERLAP X FROM ENCODER AND OVERLAP Y (LABELS) FOR TRAIN. USING ORIGINAL X FROM AUTOENCODER AND Y FOR TEST


from sklearn.model_selection import GridSearchCV

rfc=RandomForestClassifier()
param_grid = { 
    'n_estimators': range(10, 61, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [3,6,9,12,15],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)


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

rf = RandomForestClassifier(n_estimators=10, max_depth=3, n_jobs=-1, criterion='gini', max_features='sqrt' )
rf.fit(X_train, y_train) # #source: https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected

print(f'Train Accuracy: {rf.score(X_train, y_train) * 100:.2f}%')
print(f'Test Accuracy: {rf.score(X_test, y_test) * 100:.2f}%')
    











