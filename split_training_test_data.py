# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 2024

@author: Lorena Carpes

"""

# Generative AI was used to debug and enhance the code, as well as to assist with parts of the script that I could not complete myself

import os
import pandas as pd


# gets the current working directory
script_dir = 'path/to/your/script' 

# change to the script's working directory
os.chdir(script_dir)

path_file = 'path/to/your/file.csv' 

eye_tracking_data = pd.read_csv(path_file)


# split training - 75% and test 25%
       
# training data until observer 39. observer 40- 53 will be test data
training_data_observer_limit = round((3 * eye_tracking_data["observer"].iloc[-1] / 4), 2)  # the number of the last observer 
# training_data_observer_limit = round((3 * eye_tracking_data["observer"].max() / 4), 2) # alternatively, get the observer with the biggest number

print(training_data_observer_limit)

#training data ends on gaze index 78451
training_data = eye_tracking_data.iloc[:78452,:]
training_data.shape

# test data
test_data = eye_tracking_data.iloc[78452:,:]
test_data.shape


##### TRAINING DATA 1 - SEQUENTIALLY ##########

# split the data into subsets of 10s (or other time interval) in "time" column sequentially



training_data_seq = training_data.copy()


def subset_training_data_seq(time_interval: int=10) -> None:
    
    """
    Splits the training_data_seq DataFrame into sequential subsets based on time intervals and resets.
       
    Parameters:
        time_interval (int): The time threshold in seconds for starting a new subset.
    
    Returns:
        None. Modifies the 'subset' column of the training_data_seq DataFrame in place.
   

"""
    
    subset = 1
    beginning_subset = 0
    training_data_seq["subset"] = ""

    for index, gaze in training_data_seq.iterrows():
        if index < len(training_data_seq) - 1:  # Prevent out-of-bounds error
            time_difference: float = gaze["time"] - training_data_seq.iloc[beginning_subset]["time"]
            #time_reset = training_data_seq.iloc[index + 1]["time"] < gaze["time"]
            time_reset: int = training_data_seq.iloc[index + 1]["observer"] != gaze["observer"]

            if time_difference >= time_interval or time_reset:
                training_data_seq.loc[beginning_subset:index, "subset"] = subset
                subset += 1
                beginning_subset = index + 1
    
    # Fill any remaining rows in the last subset if the loop ends before assigning
    if beginning_subset < len(training_data_seq):
        training_data_seq.loc[beginning_subset:, "subset"] = subset

subset_training_data_seq() # 195 subsets are created with time_interval = 10s




##### TRAINING DATA 2 - OVERLAPPING ##########

# in training - split in 10s overlapping 
training_data_overlap = training_data.copy()

def subset_training_data_overlap(training_data_overlap: pd.DataFrame, time_interval: int = 10, overlap: int = 3) -> pd.DataFrame:
    """
    Splits the training_data DataFrame into overlapping subsets.
    
    Parameters:
        training_data_overlap (pd.DataFrame): The input DataFrame with a 'time' column.
        time_interval (int): The total time in seconds for each subset.
        overlap (int): The overlap time in seconds for the next subset.
        
    Returns:
        pd.DataFrame: A new DataFrame with repeated overlapping rows.
    """
    
    # Sort by time to ensure correct interval calculation
    #training_data_overlap = training_data_overlap.sort_values(by="time").reset_index(drop=True)
    
    # Calculate the number of rows to skip for each new subset (time_interval - overlap)
    subset_step = time_interval - overlap
    subsets = []  # List to collect subsets
    subset_id = 1

    # Start creating subsets based on calculated time intervals
    for start_idx in range(0, len(training_data_overlap), subset_step):
        # Determine the end time for the current subset
        end_time = training_data_overlap.iloc[start_idx]["time"] + time_interval
        
        # Select rows for the current subset based on the time interval
        current_subset = training_data_overlap[(training_data_overlap["time"] >= training_data_overlap.iloc[start_idx]["time"]) &
                                               (training_data_overlap["time"] < end_time)].copy()
        
        # Label this subset with a unique subset ID
        current_subset["subset"] = subset_id
        subsets.append(current_subset)
        
        subset_id += 1  # Increment subset ID for the next iteration

    # Concatenate all subsets into one DataFrame
    result = pd.concat(subsets, ignore_index=True)
    return result

subset_training_data_overlap() # xxxxxx subsets are created with time_interval = 10s