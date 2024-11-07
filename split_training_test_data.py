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




##### TRAINING DATA 1 - OVERLAPPING ##########

# in training - split in 10s overlapping 
training_data_overlap = training_data.copy()


def subset_training_data_overlap(time_interval: int = 10, overlap: int = 3) -> pd.DataFrame:
    """
    Splits the training_data DataFrame into overlapping subsets by creating new rows.
    Each subset overlaps the previous subset by the specified overlap period.
    
    Parameters:
        time_interval (int): The total time in seconds for each subset.
        overlap (int): The overlap time in seconds for the next subset.
        
    Returns:
        pd.DataFrame: A new DataFrame with repeated overlapping rows.
    """    
  
    subsets = []
    subset = 1
    beginning_subset_index = 0

    while beginning_subset_index < len(training_data_overlap):
        # Determine the end of the current subset
        end_subset_index = beginning_subset_index
        while (end_subset_index < len(training_data_overlap) - 1 and
               training_data_overlap.iloc[end_subset_index]["time"] - training_data_overlap.iloc[beginning_subset_index]["time"] < time_interval):
            end_subset_index += 1

        # Select the current subset
        current_subset = training_data_overlap.iloc[beginning_subset_index:end_subset_index].copy()
        current_subset["subset"] = subset

        # Append the current subset to the list of subsets
        subsets.append(current_subset)

        # Create overlapping rows  # without this part it might overwrite the overlapping rows
        overlap_start_index = max(0, end_subset_index - overlap)
        overlap_rows = training_data_overlap.iloc[overlap_start_index:end_subset_index].copy()
        overlap_rows["subset"] = subset + 1  # Assign to the next subset

        # Append the overlap rows as the beginning of the next subset
        subsets.append(overlap_rows)

        # Beginning of the next subset index
        beginning_subset_index = end_subset_index - overlap
        subset += 1

    # Concatenate all subsets into one df and reset the index
    result = pd.concat(subsets, ignore_index=True)
    return result

subset_training_data_overlap()

# Example usage
overlapping_data = subset_training_data_overlap()
print(overlapping_data)


subset_training_data_overlap() # xxxxxx subsets are created with time_interval = 10s