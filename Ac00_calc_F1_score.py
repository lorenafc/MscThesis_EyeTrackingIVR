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


###

############## CROSS VALIDATION AND CALC F1 AND OTHER SCORES, RUN 10 TIMES AND GET THE AVERAGE:

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

num_runs = 10  # Number of different runs per GT
k = 5
kf_5 = KFold(n_splits=k)

# Load train/test labels
y_train_dict = {}
y_test_dict = {}

for i in range(1, 8):  
    y_train_file = os.path.join(script_dir, f"{config['y_train_data_file']}_GT{i}.csv")
    y_test_file = os.path.join(script_dir, f"{config['y_test_data_file']}_GT{i}.csv")

    y_train_dict[i] = pd.read_csv(y_train_file)
    y_test_dict[i] = pd.read_csv(y_test_file)

# Initialize results storage
results_df = pd.DataFrame(columns=[
    "GT", "n_estimators", "max_depth", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1", "RMS", "MAE", "std error", "ROC AUC"
])

for i in range(1, 8):
    # Convert labels to arrays
    y_train = y_train_dict[i].values.ravel()
    y_test = y_test_dict[i].values.ravel()

    # Get hyperparameters
    n_estimators = params[f"gt{i}"]['random_forest']['n_estimators']
    max_depth = params[f"gt{i}"]['random_forest']['max_depth']

    # Store metrics for averaging
    run_metrics = {
        "train_acc": [],
        "test_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "rms": [],
        "mae": [],
        "std_error": [],
        "roc_auc": []
    }

    for run in range(num_runs):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=run)

        for train_index, test_index in kf_5.split(X_train):
            X_train_CV, X_test_CV = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_CV, y_test_CV = y_train[train_index], y_train[test_index]

            # Train the model
            rf.fit(X_train_CV, y_train_CV)

        # Evaluate on full train/test set
        train_accuracy = round(rf.score(X_train, y_train), 4)
        test_accuracy = round(rf.score(X_test, y_test), 4)

        y_pred = rf.predict(X_test)

        # Compute classification metrics
        precision = round(precision_score(y_test, y_pred, zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        rms = round(root_mean_squared_error(y_test, y_pred), 4)
        # rms = round(mean_squared_error(y_test, y_pred, squared=False), 4)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        std_error = round(np.std(np.abs(y_test - y_pred)), 4)
        # y_pred_proba = rf.predict_proba(X_test_ae)[:, 1]  
        roc_auc = round(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),4)

        # Store metrics for averaging
        run_metrics["train_acc"].append(train_accuracy)
        run_metrics["test_acc"].append(test_accuracy)
        run_metrics["precision"].append(precision)
        run_metrics["recall"].append(recall)
        run_metrics["f1"].append(f1)
        run_metrics["rms"].append(rms)
        run_metrics["mae"].append(mae)
        run_metrics["std_error"].append(std_error)
        run_metrics["roc_auc"].append(std_error)

    # Compute average metrics for this GT
    results_row = {
        "GT": f"GT{i}",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "Train Accuracy": np.mean(run_metrics["train_acc"]),
        "Test Accuracy": np.mean(run_metrics["test_acc"]),
        "Precision": np.mean(run_metrics["precision"]),
        "Recall": np.mean(run_metrics["recall"]),
        "F1": np.mean(run_metrics["f1"]),
        "RMS": np.mean(run_metrics["rms"]),
        "MAE": np.mean(run_metrics["mae"]),
        "std error": np.mean(run_metrics["std_error"]),       
        "roc_auc": np.mean(run_metrics["roc_auc"]), # [:, 1] to get the probabilities for class 1
    }

    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)

# Save final results
funcs_feat.save_df(results_df, "data/train_test_data/real_thesis/random_forest_metrics_avg10runs_NO_OUT_only_bceayz_8featZemb_.csv")





    
    
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







