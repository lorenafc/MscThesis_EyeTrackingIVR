# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes


The velocity and acceleration functions were made based on GazeParser library, by Hiroyuki Sogo. 

.. Part of GazeParser package.
.. Copyright (C) 2012-2015 Hiroyuki Sogo.
.. Distributed under the terms of the GNU General Public License (GPL).

website: https://gazeparser.sourceforge.net/#

Sogo, H. (2013) GazeParser: an open-source and multiplatform library for low-cost eye tracking and analysis. 
Behavior Reserch Methods 45, pp 684-695, doi:10.3758/s13428-012-0286-x

"""


import os
print("Current working directory:", os.getcwd())
print("Looking for config.json in:", os.getcwd())


import pandas as pd
import json

import Aa00_funcs_preprocessing as funcs_prep
import Aa01_funcs_extracting_features as funcs_feat


with open('config.json') as json_file:
    config = json.load(json_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

# path_file = os.path.join(script_dir, config["data_path"])

preprocessed_et_data = pd.read_csv(config["preprocessed_data_file"])
preprocessed_et_data_TEST = pd.read_csv(config["preprocessed_data_TEST_file"]) #always change the file you are testing in the config folder




####################################### VELOCITY AND ACCELERATION #################################################################
###################################################################################################################################



# # add velocity to the eye tracking data - when I add this inside a function it keeps running forever.
# eye_tracking_data_cm2deg_new = funcs.velocity(eye_tracking_data_cm2deg_new)

# # add acceleration to the eye tracking data - when I add this inside a function it keeps running forever.
# eye_tracking_data_cm2deg_new = funcs.acceleration(eye_tracking_data_cm2deg_new)


# Velocity (◦/s) and acceleration (◦/s2) of the gaze points.
# calculated using a Savitzky–Golay filter with polynomial
# order 2 and a window size of 12 ms—half the duration of shortest saccade, as suggested by Nystrom ¨
# and Holmqvist (2010)
# SAVITZKY-GOLD filter: https://medium.com/pythoneers/introduction-to-the-savitzky-golay-filter-a-comprehensive-guide-using-python-b2dd07a8e2ce#:~:text=The%20Savitzky%2DGolay%20filter%20is,explained%20further%20in%20this%20post).


#calculate velocity - when I add this inside a function it keeps running forever.
preprocessed_et_data["velocity_deg_s"] = preprocessed_et_data['cm_to_deg_inside_VE']/preprocessed_et_data["time_diff"]
preprocessed_et_data["acceler_deg_s"] = preprocessed_et_data['velocity_deg_s']/preprocessed_et_data["time_diff"]
extracted_features = preprocessed_et_data

output_file_preprocesed_and_extracted = os.path.join(script_dir, config["prepr_and_features_file"])
extracted_features.to_csv(output_file_preprocesed_and_extracted, index=False)

# keep only the extracted features and GTs: 

##### UPDATE LINE BELOW WITH NEW FEATURES!!!
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_dist_m','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'] 

# Create a new DataFrame with only the selected columns
only_extracted_features_and_GTs = preproc_add_mean_dist_m[columns_to_keep]



### feature importance !!!





# 3 - BCEA - Bivariate contour ellipse area (◦2). 
# Measures the area in which the recorded gaze position lies 
# within a 100-ms window in (P %) of the time. (Blignaut and Beelders, 2012)

# Blignaut, P., & Beelders, T. (2012). The precision of eye-trackers:
# a case for a new measure. In Proceedings of the symposium on
# eye tracking research and applications, ETRA ’12, (pp. 289–292).
# New York, NY, USA: ACM






##############################3#### MEAN DIFFF ###################################
##################################################################################


# 8 - mean-diff - Distance (◦) between the mean gaze position in a 
# 100-ms window before the sample and a 100-ms window after the sample.
# Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden

# Pg. 15 Olsson 2007

# Take the r number of points that are before the sample (so the interval are 100ms) and r points after 
# take the average for both of them. measure the distance (in degrees). this is the mean-diff

preprocessed_and_features = pd.read_csv(config["prepr_and_features_file"])
preprocessed_and_features.columns



 # add this part to the preprocessing script Aa00 . could not use the rolling function because it includes the current value of the row
#in the calculations and I wanted to exclude it, and consider average of the previous and folllowing only. 


#### PREPROCESSING BELOW! ITS IN SCRIPT A00

# def calculate_average_window(df, columns, window=4): #win = 14 to be close to 100ms (approx 90ms)

#     for column in columns:
#         results = []
#         for row_index in range(len(df)):
#             # Get the range before and after the row
#             start_before = max(0, row_index - window)
#             end_after = min(len(df), row_index + window + 1)

#             # Samples before
#             samples_before = df[start_before:row_index]
#             avg_before = np.nanmean(samples_before[column]) if len(samples_before) > 0 else np.nan

#             # Samples after
#             samples_after = df[row_index + 1:end_after]
#             avg_after = np.nanmean(samples_after[column]) if len(samples_after) > 0 else np.nan

#             # Append the result as a tuple (avg_before, avg_after)
#             results.append((avg_before, avg_after))

#         # Add the results to the DataFrame
#         df[f'{column}_Avg_Before_Win{window}'] = [result[0] for result in results]
#         df[f'{column}_Avg_After_Win{window}'] = [result[1] for result in results]

#     return df



# Example usage
# # Example DataFrame
# df3 = pd.DataFrame({
#     'L_x': [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9],
#     'L_y': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
#     'L_z': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# })

# # Specify the columns
# columns = ['L_x', 'L_y', 'L_z']

# # Call the function
# df3 = calculate_rolling_average(df3, columns, window=4)

# print(df3)

xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']

# cxcycz = ['C_x', 'C_y', 'C_z']

# win_4_x_y_z = funcs_prep.calculate_average_window(preprocessed_and_features, xyz, window=4)   
win_4_x_y_z_cx_cy_cz = funcs_prep.calculate_average_window(preprocessed_and_features, xyzcxcycz, window=4)

# diff mean current and previous:

    
# calc dist before and after - adjust function to before and after instead of current and previous row 
# import math

# def calc_mean_dist_m(df): # a bit slow to run
    
#     if "mean_dist_m" not in df.columns:
#         df["mean_dist_m"] = ""
        
#     for gaze in range(1, len(df)):
        
#         x, y, z  = df.iloc[gaze]["L_x_Avg_After_Win4"], df.iloc[gaze]["L_y_Avg_After_Win4"], df.iloc[gaze]["L_z_Avg_After_Win4"]
#         prev_x, prev_y, prev_z = df.iloc[gaze]["L_x_Avg_Before_Win4"], df.iloc[gaze]["L_y_Avg_Before_Win4"], df.iloc[gaze]["L_z_Avg_Before_Win4"]
    
#         sqr_dist_x = (x - prev_x)**2
#         sqr_dist_y = (y - prev_y)**2
#         sqr_dist_z = (z - prev_z)**2

#         sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
#         mean_distance_meters = round(math.sqrt(sum_square_distances), 4)
       
#         df.at[gaze,'mean_dist_m'] = mean_distance_meters   
#         df['mean_dist_m'] = pd.to_numeric(df['mean_dist_m'], errors='coerce')
      
#     return df    


preproc_add_mean_dist_m = funcs_feat.calc_mean_dist_m(win_4_x_y_z_cx_cy_cz) # in meters, convert then to degrees. 


output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
preproc_add_mean_dist_m.to_csv(output_file_features_GTs_updated, index=False)




# Define the list of columns to keep
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_dist_m','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

# Create a new DataFrame with only the selected columns
only_vel_acc_mean_dist_m_and_GTs = preproc_add_mean_dist_m[columns_to_keep]

output_file_mean_dist_m = os.path.join(script_dir, "data/Aa01_only_vel_acc_mean_dist_m_and_GTs.csv")
only_vel_acc_mean_dist_m_and_GTs.to_csv(output_file_mean_dist_m, index=False)




# def calc_viewing_distance(lx,ly,lz,cx,cy,cz):   # CALCULATE ONLU BEFORE
    
#     sqr_dist_x = (cx-lx)**2
#     sqr_dist_y = (cy-ly)**2
#     sqr_dist_z = (cz-lz)**2

#     sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
#     viewing_distance = round(math.sqrt(sum_square_distances),4)
    
#     return viewing_distance


# def apply_viewing_distance_df(df): # calculate only viewing dist before to use in the formula to convert the mean_dist_m to mean_diff in degrees.
    
#     if "viewing_distance" not in df.columns:
#         df["viewing_distance"] = ""
    
#     for gaze, row in df.iterrows():
#         viewing_distance = calc_viewing_distance(row["L_x_Avg_Before_Win4"], row["L_y_Avg_Before_Win4"],row["L_z_Avg_Before_Win4"], row["C_x_Avg_Before_Win4"],row["C_y_Avg_Before_Win4"], row["C_z_Avg_Before_Win4"])    
     
#         df.at[gaze,'viewing_distance'] = viewing_distance
        
#     return df
 

# def mean_diff_degree_inside_VE(df): # based on GazeParser library, by Hiroyuki Sogo.
    
#     if "cm_to_deg_inside_VE" not in df.columns:
#         df["cm_to_deg_inside_VE"] = np.nan   
        
#     df['coordinates_dist'] = pd.to_numeric(df['coordinates_dist'], errors='coerce')
#     df["viewing_distance"] = pd.to_numeric(df["viewing_distance"], errors='coerce')


#     df = df.dropna(subset=['coordinates_dist', 'viewing_distance']).reset_index(drop=True)
    
#     # Calculate cm_to_deg_inside_VE using vectorized operations
#     df["cm_to_deg_inside_VE"] = (180 / np.pi * np.arctan(df["coordinates_dist"] / (2 * df["viewing_distance"])))
#     df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
        
#     return df

# preprocessed_et_data_TEST = pd.read_csv(config["preprocessed_data_TEST_file"]) 


# NOW CONVERT TO MEAN-DIFF TO DEGREES:


preproc_add_mean_dist_m_view_dist = funcs_feat.apply_viewing_distance_df(preproc_add_mean_dist_m)
preproc_add_mean_diff_degree = funcs_feat.mean_diff_degree_inside_VE(preproc_add_mean_dist_m_view_dist)

# output_file_features_GTs = os.path.join(script_dir, config["only_extracted_features_file_and_GTs"])
# only_extracted_features_and_GTs.to_csv(output_file_features_GTs, index=False)


output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
preproc_add_mean_diff_degree.to_csv(output_file_features_GTs_updated, index=False)



preproc_add_mean_diff_degree.columns


# Define the list of columns to keep
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

# Create a new DataFrame with only the selected columns
only_vel_acc_mean_diff_and_GTs = preproc_add_mean_diff_degree[columns_to_keep]
output_file_features_GTs_TEST = os.path.join(script_dir, config["only_extracted_features_and_GTs_TEST_file"])
only_vel_acc_mean_diff_and_GTs.to_csv(output_file_features_GTs_TEST, index=False)


preprocessed_and_features.columns 


    
    # after having both means (and therefore the coordinates_distance) use this function from preprocessing script: convert_cm_to_degree_inside_VE(df)
    

# use columns cm_to_degree to calculate the values for the mean - I dont know if I should do with x, y and z distance and in the end convert to degree

# 1 - select 4 samples before the sample of interest aprox 90ms).

# - divide by 4 and sum them

# 2 - do the same with 4 samples after

# 5 - calculate the diff of the average before and after (if it is already in degree just subtract, if it is the coordinates, calculate the euclidean distance and then convert to degrees)
# convert to degrees in the end








# 4 - BCEA-DIFF Difference in bivariate contour ellipse area (◦2) 
#between two 100ms windows, one before and one after the sample. Olsson (2007)



# 5- disp - Dispersion (◦). Calculated as (xmax −xmin)+(ymax −ymin) 
# over a 100-ms window. (Salvucci & Goldberg,
# 2000).
# Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and
# saccades in eye-tracking protocols. In Proceedings of the 2000
# symposium on eye tracking research & applications, ETRA ’00
# (pp. 71–78).


# 6 - fs - Sampling frequency (Hz). Mean sampling rate: 44.27 Hz  (SMI BeGaze)



# 7 - i2mc - A feature used to detect saccades in very noisy data. 
# The final weights come from the two-means clustering procedure as 
# per the original implementation by Hessels et al. (2016). 
# A 200-ms window was used, centered on the sample.
# Hessels, R. S., Niehorster, D. C., Kemner, C., & Hooge, I. T. C. (2016).
# Noise-robust fixation detection in eye movement data: Identification by two-means clustering (i2mc). Behavior Research Methods,
# 1–22. doi:10.3758/s13428-016-0822-1





# 9 - med-diff - Distance (◦) between the median gaze position in a 
# 100-ms window before the sample and a 100-ms window after the sample.
# Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden


# 10 - Rayleightest - Tests whether the sample-to-sample directions in
# a 22-ms window are uniformly distributed. Larsson et al. (2015)
# Larsson, L., Nystrom, M., & Stridh, M. (2013). Detection of sac- ¨
# cades and postsaccadic oscillations in the presence of smooth
# pursuit. IEEE Transactions on Biomedical Engineering, 60(9),
# 2484–2493.


# 11 - rms - Root mean square (◦) of the sample-to-sample displacement in a 
# 100-ms window centered on a sample.  (Holmqvist et al., 2011) # book 152 dollars
# Holmqvist, K., Andersson, R., Jarodzka, H., Kok, E., Nystrom, M., ¨
# & Dewhurst, R. (2016). Eye tracking. A comprehensive guide to
# methods and measures. Oxford: Oxford University Press


# 12 - rms-diff - Difference in root mean square (◦) between two 100-ms 
# windows before and after the sample. Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden

#zemblys 2017:
# Next to these,
# we also propose several new features, which we hypothesize are likely to be useful for the detection of the onset and offset
# of saccades: rms-diff, std-diff and bcea-diff. These new features are inspired by Olsson (2007) and are calculated
# by taking the difference in the RMS, STD, and BCEA precision measures calculated for 100-ms windows preceding
# and following the current sample. Obviously, the largest differences (and therefore peaks in the feature) should occur
# around the onset and offset of the saccades.


# 13 - std - Standard deviation (◦) of the recorded gaze position in a 
# 100-ms window centered on the sample.  (Holmqvist et al., 2011) # book 152 dollars
# Holmqvist, K., Andersson, R., Jarodzka, H., Kok, E., Nystrom, M., ¨
# & Dewhurst, R. (2016). Eye tracking. A comprehensive guide to
# methods and measures. Oxford: Oxford University Press



# 14 - std - diff Difference in standard deviation (◦) between two 100-ms windows, 
# one before and one after the sample. Olsson (2007)

# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden






