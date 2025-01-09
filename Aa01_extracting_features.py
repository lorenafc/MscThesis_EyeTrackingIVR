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
import pandas as pd
import json

# import Aa01_funcs_extracting_features


with open('config.json') as json_file:
    config = json.load(json_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
preprocessed_et_data = pd.read_csv(config["preprocessed_data_file"])



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

# only the extracted features and GTs:
only_extracted_features_and_GTs = extracted_features.drop(columns=['time', 'observer', 'coordinates', 'coordinates_dist','cm_to_deg_inside_VE','L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'viewing_distance','time_diff'])
right_order = ['velocity_deg_s', 'acceler_deg_s', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
only_extracted_features_and_GTs_reordered = extracted_features[right_order]

output_file_features_GTs = os.path.join(script_dir, config["only_extracted_features_file_and_GTs"])
only_extracted_features_and_GTs_reordered.to_csv(output_file_features_GTs, index=False)




### feature importace and divide in 86 Hz!!!


#### Data 86 Hz 

preprocessed_et_data_86Hz = pd.read_csv(config["preprocessed_data_86Hz_file"])




#calculate velocity - when I add this inside a function it keeps running forever.
preprocessed_et_data_86Hz["velocity_deg_s"] = preprocessed_et_data_86Hz['cm_to_deg_inside_VE']/preprocessed_et_data_86Hz["time_diff"]
preprocessed_et_data_86Hz["acceler_deg_s"] = preprocessed_et_data_86Hz['velocity_deg_s']/preprocessed_et_data_86Hz["time_diff"]
extracted_features_86Hz = preprocessed_et_data_86Hz

output_file_preprocesed_and_extracted_86Hz = os.path.join(script_dir, config["prepr_and_features_file_86Hz"])
extracted_features_86Hz.to_csv(output_file_preprocesed_and_extracted_86Hz, index=False)

# only the extracted features and GTs:
only_extracted_features_and_GTs_86Hz = extracted_features_86Hz.drop(columns=['time', 'observer', 'coordinates', 'coordinates_dist','cm_to_deg_inside_VE','L_x', 'L_y', 'L_z', 'C_x', 'C_y', 'C_z', 'viewing_distance','time_diff'])
right_order = ['velocity_deg_s', 'acceler_deg_s', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
only_extracted_features_and_GTs_reordered_86Hz = extracted_features_86Hz[right_order]

output_file_features_GTs_86Hz = os.path.join(script_dir, config["only_extracted_features_and_GTs_86Hz_file"])
only_extracted_features_and_GTs_reordered_86Hz.to_csv(output_file_features_GTs_86Hz, index=False)



 
# def calc_velocity_deg_s(df):
    
#     import numpy as np
    
      
#     # if "velocity" not in df.columns:
#     #     df["velocity"] = ""  
    

#     df = df[df["time_diff"] != 0].reset_index(drop=True)
    
#     # df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
#     # df["time_diff"] = pd.to_numeric(df["time_diff"], errors='coerce')
       
#     # df["velocity"] = df['cm_to_deg_inside_VE']/df["time_diff"]
    

    
#     # df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
#     # df["time_diff"] = pd.to_numeric(df["time_diff"], errors='coerce')
       
#     df["velocity"] = df['cm_to_deg_inside_VE']/df["time_diff"]
    
    
#     # for gaze in range(0, len(df)):
#     #     velocity = df.iloc[gaze]['cm_to_deg_inside_VE']/df.iloc[gaze]["time_diff"]
#     #     df.at[gaze,'velocity'] = velocity  
    
#     return df




# 3 - BCEA - Bivariate contour ellipse area (◦2). 
# Measures the area in which the recorded gaze position lies 
# within a 100-ms window in (P %) of the time. (Blignaut and Beelders, 2012)

# Blignaut, P., & Beelders, T. (2012). The precision of eye-trackers:
# a case for a new measure. In Proceedings of the symposium on
# eye tracking research and applications, ETRA ’12, (pp. 289–292).
# New York, NY, USA: ACM










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


# 8 - mean-diff - Distance (◦) between the mean gaze position in a 
# 100-ms window before the sample and a 100-ms window after the sample.
# Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden

# Pg. 15 Olsson 2007

# Take the r number of points that are before the sample (so the interval are 100ms) and r points after 
# take the average for both of them. measure the distance (in degrees). this is the mean-diff

preprocessed_and_features = pd.read_csv(config["prepr_and_features_file"])
preprocessed_and_features.columns

rolling_win5 = df.close.rolling(5).mean()


import pandas as pd
import numpy as np

# Example DataFrame
df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9]})

# Function to calculate the average of 4 samples before and after a given row
def calculate_rolling_average(row_index, column, data, window=4):
    # Get the range before and after the row
    start_before = max(0, row_index - window)
    end_after = min(len(data), row_index + window + 1)
    
    # Samples before
    samples_before = data[start_before:row_index]
    avg_before = np.nanmean(samples_before[column]) if len(samples_before) > 0 else np.nan
    
    # Samples after
    samples_after = data[row_index + 1:end_after]
    avg_after = np.nanmean(samples_after[column]) if len(samples_after) > 0 else np.nan
    
    return avg_before, avg_after

# Apply the function row by row
column = 'B'

results = [
    calculate_rolling_average(i, column, df, window=4) for i in range(len(df))
]

# Add the results to the DataFrame
df[f'Avg_Before_{column}'] = [result[0] for result in results]
df['Avg_After'] = [result[1] for result in results]

print(df)


win_4_L_x = calculate_rolling_average(i, 'L_x', df, window=4) for i in range(len(preprocessed_and_features))
]

    
import pandas as pd
import numpy as np

def calculate_windows_average(df, column, window=4):
    """
    Calculates the average of a specified number of rows before and after each row,
    adds the results as new columns to the DataFrame, and returns the updated DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The column name for which rolling averages are calculated.
    - window (int): The number of rows before and after to consider for averaging.

    Returns:
    - pd.DataFrame: The updated DataFrame with new columns for rolling averages.
    """
    results = []
    for row_index in range(len(df)):
        # Get the range before and after the row
        
        start_before = max(0, row_index - window)
        end_after = min(len(df), row_index + window + 1)

        # Samples before
        samples_before = df[start_before:row_index]
        avg_before = np.nanmean(samples_before[column]) if len(samples_before) > 0 else np.nan

        # Samples after12a
        samples_after = df[row_index + 1:end_after]
        avg_after = np.nanmean(samples_after[column]) if len(samples_after) > 0 else np.nan

        # Append the result as a tuple (avg_before, avg_after)
        results.append((avg_before, avg_after))

    # Add the results to the DataFrame
    df[f'{column}_Avg_Before_Win{window}'] = [result[0] for result in results]
    df[f'{column}_Avg_After_Win{window}'] = [result[1] for result in results]

    return df

# Example usage
# Example DataFrame
df2 = pd.DataFrame({'B': [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9]})

# Call the function
df2 = calculate_windows_average(df2, 'B', window=4)

print(df2)

win_4_L_x =  calculate_windows_average(preprocessed_and_features, 'L_x', window=4)

win_4_L_x_y =  calculate_windows_average(win_4_L_x , 'L_y', window=4)

    
xyz = ['L_x', 'L_y', 'L_z']

win_4_Lxyz =   calculate_windows_average(preprocessed_and_features, xyz, window=4)




def calculate_average_window(df, columns, window=4): # add this part to the preprocessing script Aa00
    """
    Calculates the average of a specified number of rows before and after each row
    for a list of columns, adds the results as new columns to the DataFrame, and returns the updated DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list of str): A list of column names for which rolling averages are calculated.
    - window (int): The number of rows before and after to consider for averaging.

    Returns:
    - pd.DataFrame: The updated DataFrame with new columns for rolling averages.
    """
    for column in columns:
        results = []
        for row_index in range(len(df)):
            # Get the range before and after the row
            start_before = max(0, row_index - window)
            end_after = min(len(df), row_index + window + 1)

            # Samples before
            samples_before = df[start_before:row_index]
            avg_before = np.nanmean(samples_before[column]) if len(samples_before) > 0 else np.nan

            # Samples after
            samples_after = df[row_index + 1:end_after]
            avg_after = np.nanmean(samples_after[column]) if len(samples_after) > 0 else np.nan

            # Append the result as a tuple (avg_before, avg_after)
            results.append((avg_before, avg_after))

        # Add the results to the DataFrame
        df[f'{column}_Avg_Bef_Win{window}'] = [result[0] for result in results]
        df[f'{column}_Avg_Aft_Win{window}'] = [result[1] for result in results]

    return df

# Example usage
# Example DataFrame
df3 = pd.DataFrame({
    'L_x': [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9],
    'L_y': [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    'L_z': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Specify the columns
columns = ['L_x', 'L_y', 'L_z']

# Call the function
df3 = calculate_rolling_average3(df3, columns, window=4)

print(df3)

xyz = ['L_x', 'L_y', 'L_z']

cxcycz = ['C_x', 'C_y', 'C_z']

win_4_x_y_z = calculate_average_window(preprocessed_and_features, xyz, window=4)   

win_4_x_y_z_cx_cy_cz = calculate_average_window(win_4_x_y_z, cxcycz, window=4)

# diff mean current and previous:

win_4_Lxyz.columns
    
# calc dist before and after - adjust function to before and after instead of current and previous row 
import math

def calc_mean_dist_m(df): # a bit slow to run
    
    if "mean_dist_m" not in df.columns:
        df["mean_dist_m"] = ""
        
    for gaze in range(1, len(df)):
        
        x, y, z  = df.iloc[gaze]["L_x_Avg_After_Win4"], df.iloc[gaze]["L_y_Avg_After_Win4"], df.iloc[gaze]["L_z_Avg_After_Win4"]
        prev_x, prev_y, prev_z = df.iloc[gaze]["L_x_Avg_Before_Win4"], df.iloc[gaze]["L_y_Avg_Before_Win4"], df.iloc[gaze]["L_z_Avg_Before_Win4"]
    
        sqr_dist_x = (x - prev_x)**2
        sqr_dist_y = (y - prev_y)**2
        sqr_dist_z = (z - prev_z)**2

        sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
        mean_distance_meters = round(math.sqrt(sum_square_distances), 4)
       
        df.at[gaze,'mean_dist_m'] = mean_distance_meters   
        df['mean_dist_m'] = pd.to_numeric(df['mean_dist_m'], errors='coerce')
      
    return df    


preproc_add_mean_dist_m = calc_mean_dist_m(win_4_x_y_z_cx_cy_cz) # in meters, convert then to degrees. 

# Define the list of columns to keep
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_dist_m']

# Create a new DataFrame with only the selected columns
selected_columns_df = preproc_add_mean_diff[columns_to_keep]

# Display the new DataFrame
print(selected_columns_df)



def calc_viewing_distance(lx,ly,lz,cx,cy,cz):   
    
    sqr_dist_x = (cx-lx)**2
    sqr_dist_y = (cy-ly)**2
    sqr_dist_z = (cz-lz)**2

    sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
    viewing_distance = round(math.sqrt(sum_square_distances),4)
    
    return viewing_distance

def apply_viewing_distance_df(df): # calculate only viewing dist before to use in the formula to convert the mean_dist_m to mean_diff in degrees.
    
    if "viewing_distance" not in df.columns:
        df["viewing_distance"] = ""
    
    for gaze, row in df.iterrows():
        viewing_distance = calc_viewing_distance(row["L_x_Avg_Before_Win4"], row["L_y_Avg_Before_Win4"],row["L_z_Avg_Before_Win4"], row["C_x_Avg_Before_Win4"],row["C_y_Avg_Before_Win4"], row["C_z_Avg_Before_Win4"])    
     
        df.at[gaze,'viewing_distance'] = viewing_distance
        
    return df

    



def mean_diff_degree_inside_VE(df): # based on GazeParser library, by Hiroyuki Sogo.
    
    if "cm_to_deg_inside_VE" not in df.columns:
        df["cm_to_deg_inside_VE"] = np.nan   
        
    df['coordinates_dist'] = pd.to_numeric(df['coordinates_dist'], errors='coerce')
    df["viewing_distance"] = pd.to_numeric(df["viewing_distance"], errors='coerce')


    df = df.dropna(subset=['coordinates_dist', 'viewing_distance']).reset_index(drop=True)
    
    # Calculate cm_to_deg_inside_VE using vectorized operations
    df["cm_to_deg_inside_VE"] = (180 / np.pi * np.arctan(df["coordinates_dist"] / (2 * df["viewing_distance"])))
    df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
        
    return df




output_file_features_GTs_86Hz = os.path.join(script_dir, config["only_extracted_features_and_GTs_86Hz_file"])
only_extracted_features_and_GTs_reordered_86Hz.to_csv(output_file_features_GTs_86Hz, index=False)
    

preprocessed_and_features.columns 
    
    # after having both means (and therefore the coordinates_distance) use this function from preprocessing script: convert_cm_to_degree_inside_VE(df)
    

# use columns cm_to_degree to calculate the values for the mean - I dont know if I should do with x, y and z distance and in the end convert to degree

# 1 - select 4 samples before the sample of interest aprox 90ms).

# - divide by 4 and sum them

# 2 - do the same with 4 samples after

# 5 - calculate the diff of the average before and after (if it is already in degree just subtract, if it is the coordinates, calculate the euclidean distance and then convert to degrees)
# convert to degrees in the end


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




### GAZE PARSER VELOCITY:
    
    #######

    # Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixa- ¨
    # tion, saccade, and glissade detection in eyetracking data. Behavior
    # Research Methods, 42(1), 188–204

    ## Llanes Jurado study - use window of 0.3 ms and 1 degree (GT1) - to consider fixation

    # CONVERT PIXELS TO DEGREES

    #(Llanes-Jurado 2020, p.3) The virtual environment was displayed through an HTC Vive Pro Eye, an HMD with an
    #integrated ET system, offering a field of view of 110◦. The scene is displayed with a
    # resolution of 1440 × 1600 pixels per eye

    #	1440 x 1600 pixels per eye (2880 x 1600 pixels combined), field of view: 110◦. 
    #source: https://developer.vive.com/resources/hardware-guides/vive-pro-eye-specs-user-guide/

    # I am a bit confused because I think there are 2 observers. one from the eye to the equipment screen, 
    # and another inside the virtual environment to the objects

    # pixels_h = 1440 #considering individual eye
    # pixels_v = 1600
    # field_of_view = 110 # degrees

    # pix_to_deg = field_of_view/np.array([pixels_h, pixels_v]) 
    # pix_to_deg # degrees per pixel - array([0.07638889, 0.06875   ])



    # I dont know how can I relate the relation pix to degree of the screen with the virtual environment to calculate the speed.
    # maybe just use yes moviment from the virtual environment?

    # calculate differente between 2 folowing coordinates:



    # eye_tracking_data_coord_dist.columns
    # Tdiff = np.diff(T)#.reshape(-1, 1)
    # HVdeg = np.zeros(HV.shape)
    # HVdeg[:, 0] = HV[:, 0] * pix_to_deg[0]
    # HVdeg[:, 1] = HV[:, 1] * pix_to_deg[1]

    ### I dont think I need to use it because I already have the field of view (110o). the distance between the 
    # eyes and screen is fixed using 

    # eye_tracking_data["cm_to_deg"] = ""




    # dots_per_cm_h = 
    # dots_per_cm_v = 

    # deg_to_pix = numpy.array([dots_per_cm_h, dots_per_cm_v])/cm2deg

    ## from gaze parser below: 
        
    #     # cm2deg = 180/numpy.pi*numpy.arctan(1.0/config.VIEWING_DISTANCE) # 1rad * 45 degrees
    #     # deg2pix = numpy.array([config.DOTS_PER_CENTIMETER_H, config.DOTS_PER_CENTIMETER_V])/cm2deg
    #     # pix2deg = 1.0/deg2pix

    #     Tdiff = numpy.diff(T)#.reshape(-1, 1)
    #     HVdeg = numpy.zeros(HV.shape)
    #     # HVdeg[:, 0] = HV[:, 0] * pix2deg[0]
    #     # HVdeg[:, 1] = HV[:, 1] * pix2deg[1]




