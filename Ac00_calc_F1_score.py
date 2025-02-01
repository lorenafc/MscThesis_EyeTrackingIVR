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
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import Aa01_funcs_extracting_features as funcs_feat

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
# X_train = X_train #.values
# X_test = X_test #.values

y_train_dict = {}
y_test_dict = {}

####################### WITHOUT CROSS VALIDATION ##########################

for i in range(1, 8):  
    y_train_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
                # y_train_file = os.path.join(script_dir, f"{config['y_train_data_86Hz_file']}_GT{i}.csv")
    
                # y_test_file = os.path.join(script_dir, f"{config['y_test_data_86Hz_file']}_GT{i}.csv") # Y test must be original. not the 86Hz
    y_test_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")
    
    
    y_train_dict[i] = pd.read_csv(y_train_file)
    y_test_dict[i] = pd.read_csv(y_test_file)
    
results_df = pd.DataFrame(columns=[
    "GT", "n_estimators", "max_depth", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1", "RMS", "MAE", "std error"
])

  
for i in range(1,8):
    
    #convert df to array and then 1D arrays for RF model
    y_train = y_train_dict[i].values.ravel()
    y_test = y_test_dict[i].values.ravel()
    
    n_estimators = params[f"gt{i}"]['random_forest']['n_estimators']
    max_depth = params[f"gt{i}"]['random_forest']['max_depth']
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, n_jobs=-1) 
    rf.fit(X_train, y_train)
    
    train_accuracy = round(rf.score(X_train, y_train),4)
    test_accuracy = round(rf.score(X_test, y_test), 4)
    print(f"GT{i} - Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"GT{i} - Test Accuracy: {test_accuracy * 100:.2f}%")
    
    y_pred = rf.predict(X_test)
    # print(f"Classification Report GT{i}:\n", classification_report(y_test, y_pred))
    # print(f"Confusion Matrix GT{i}:\n", confusion_matrix(y_test, y_pred))
  
    
    # Classification metrics - source: Joep Robben

    precision = round(precision_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
    recall = round(recall_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
    f1 = round(f1_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
    rms = round(mean_squared_error(y_test, y_pred, squared=False),4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    std_error = round(np.std(np.abs(y_test - y_pred)), 4)
    
    # Append metrics to results_row dict
    results_row = {
        "GT": f"GT{i}",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "RMS": rms,
        "MAE": mae,
        "std error": std_error
        
    }
    
    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)


# Calculate average metrics across all GTs
average_metrics = results_df.mean(numeric_only=True)
average_metrics_rounded = average_metrics.round(4)
print("\nAverage Metrics Across All GTs:")
print(average_metrics)

average_metrics_df = average_metrics_rounded.reset_index()

average_metrics_df.columns = ["Metric", "Value"]

#save average metrics 
funcs_feat.save_df(average_metrics_df, "data/train_test_data/testing_features/prAc00_average_results_bcea_YZ_3D_noise_simple.csv")

# save all GTs metrics
funcs_feat.save_df(results_df, "data/train_test_data/testing_features/prAc00_results_bcea_YZ_3D_noise_simple_only.csv")

    
    
## FEATURE IMPORTANCE:
    
# source https://machinelearningmastery.com/calculate-feature-importance-with-python/    



# get importance
from matplotlib import pyplot as plt

feature_names = X_train.columns
importance = rf.feature_importances_


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df = importance_df.reset_index(drop=True)

# Plot feature importance with feature names on the x-axis
plt.bar(importance_df.index, importance_df['importance'])
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance')

# Set the x-axis labels to the feature names
plt.xticks(importance_df.index, importance_df['feature'], rotation=45, ha='right')

plt.show()


# summarize feature importance
for i,v in enumerate(importance):
 	print('Feature: %0d, Score: %.5f' % (i,v))

   
importance_list = [{"feature": i, "importance": v} for i, v in enumerate(importance)]
importance_df = pd.DataFrame(importance_list)

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df = importance_df.reset_index(drop=True)


plt.bar(importance_df.index, importance_df['importance'])
plt.xlabel('Feature Number')
plt.ylabel('Importance Score')
plt.title('Feature Importance')


# Set the x-axis labels to the feature numbers
plt.xticks(importance_df.index, importance_df['feature'])

plt.show()




from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# 1 all preproc file

# X_train_cv = pd.read_csv("data/train_test_data/prAb00_X_train.csv")
# # X_test = pd.read_csv(X_test_file)


# df_cv = X_train_cv.copy()  # Your training data
# observer_ids = df_cv["observer"].unique()  # Get unique observers

# # Define folds (manually distributing observers)
# fold_sizes = [8, 8, 8, 8, 8]  # Number of observers per fold
# folds = []
# start = 0

# # Divide observer IDs into folds
# for size in fold_sizes:
#     folds.append(observer_ids[start:start + size])
#     start += size

# # Create train-test splits based on observer groups
# for fold_idx, test_observers in enumerate(folds):
#     print(f"Fold {fold_idx + 1}: Test Observers - {test_observers}")
    
#     # Test set: Data from current fold's observers
#     test_data = df_cv[df_cv["observer"].isin(test_observers)]
    
#     # Train set: Data from all other observers
#     train_data = df_cv[~df_cv["observer"].isin(test_observers)]
    
#     print(f"Fold {fold_idx + 1} Train Size: {train_data.shape[0]}, Test Size: {test_data.shape[0]}")
    
# df_cv_rf = df_cv.drop(columns=["observer"])

# X_train = df_cv_rf


# ## now I need to adjust the code below to consider my division of observers for train and test
# from sklearn.model_selection import KFold
# from sklearn.metrics import f1_score

# k = 5
# kf_5 = KFold(n_splits = k, random_state = 24)




y_train_dict = {}
y_test_dict = {}

for i in range(1, 8):  
    y_train_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
                # y_train_file = os.path.join(script_dir, f"{config['y_train_data_86Hz_file']}_GT{i}.csv")
    
                # y_test_file = os.path.join(script_dir, f"{config['y_test_data_86Hz_file']}_GT{i}.csv") # Y test must be original. not the 86Hz
    y_test_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")
    
    
    y_train_dict[i] = pd.read_csv(y_train_file)
    y_test_dict[i] = pd.read_csv(y_test_file)
    
results_df = pd.DataFrame(columns=[
    "GT", "n_estimators", "max_depth", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1", "RMS", "MAE", "std error"
])

  
for i in range(1,8):
    
    #convert df to array and then 1D arrays for RF model
    y_train = y_train_dict[i].values.ravel()
    y_test = y_test_dict[i].values.ravel()
    
    n_estimators = params[f"gt{i}"]['random_forest']['n_estimators']
    max_depth = params[f"gt{i}"]['random_forest']['max_depth']
    
    
    # source: https://stackoverflow.com/questions/71615078/how-to-do-cross-validation-on-random-forest
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, n_jobs=-1) 

  
    for train_index, test_index in kf_5.split(X_train):
        X_train_CV, X_test_CV = X_train[train_index], X_train[test_index]
        y_train_CV, y_test_CV = y_train[train_index], y_train[test_index]
        rf.fit(X_train_CV, y_train_CV)
        # rf.fit(X_train, y_train)

        
        # rf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, n_jobs=-1) 
        # rf.fit(X_train, y_train)
        
        train_accuracy = round(rf.score(X_train, y_train),4)
        test_accuracy = round(rf.score(X_test, y_test), 4)
        print(f"GT{i} - Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"GT{i} - Test Accuracy: {test_accuracy * 100:.2f}%")
        
        y_pred = rf.predict(X_test)
        # print(f"Classification Report GT{i}:\n", classification_report(y_test, y_pred))
        # print(f"Confusion Matrix GT{i}:\n", confusion_matrix(y_test, y_pred))
      
        
        # Classification metrics - source: Joep Robben
    
        precision = round(precision_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
        recall = round(recall_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
        f1 = round(f1_score(y_true=y_test, y_pred=y_pred, zero_division=0),4)
        rms = round(mean_squared_error(y_test, y_pred, squared=False),4)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        std_error = round(np.std(np.abs(y_test - y_pred)), 4)
        
        # Append metrics to results_row dict
        results_row = {
            "GT": f"GT{i}",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "RMS": rms,
            "MAE": mae,
            "std error": std_error
            
        }
        
        results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)
    


   
# Calculate average metrics across all GTs
average_metrics = results_df.mean(numeric_only=True)
average_metrics_rounded = average_metrics.round(4)
print("\nAverage Metrics Across All GTs:")
print(average_metrics)

average_metrics_df = average_metrics_rounded.reset_index()

average_metrics_df.columns = ["Metric", "Value"]

#save average metrics 
funcs_feat.save_df(average_metrics_df, "data/train_test_data/testing_features/prAc00_average_results_bcea_3D_noise_cv.csv")

# save all GTs metrics
funcs_feat.save_df(results_df, "data/train_test_data/testing_features/prAc00_results_bcea_3D_noise_cv.csv")

    
    
## FEATURE IMPORTANCE:
    
# source https://machinelearningmastery.com/calculate-feature-importance-with-python/    



# get importance
from matplotlib import pyplot as plt

feature_names = X_train.columns
importance = rf.feature_importances_


importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df = importance_df.reset_index(drop=True)

# Plot feature importance with feature names on the x-axis
plt.bar(importance_df.index, importance_df['importance'])
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.title('Feature Importance')

# Set the x-axis labels to the feature names
plt.xticks(importance_df.index, importance_df['feature'], rotation=45, ha='right')

plt.show()


# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

   
importance_list = [{"feature": i, "importance": v} for i, v in enumerate(importance)]
importance_df = pd.DataFrame(importance_list)


importance_df = importance_df.sort_values(by="importance", ascending=False)
importance_df = importance_df.reset_index(drop=True)


plt.bar(importance_df.index, importance_df['importance'])
plt.xlabel('Feature Number')
plt.ylabel('Importance Score')
plt.title('Feature Importance')


plt.xticks(importance_df.index, importance_df['feature'])

plt.show()



