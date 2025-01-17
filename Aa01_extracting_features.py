# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes


The velocity and acceleration functions were made based on GazeParser library, by Hiroyuki Sogo. 

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


preprocessed_et_data = pd.read_csv(config["preprocessed_data_file"])



####################################### VELOCITY AND ACCELERATION #################################################################
###################################################################################################################################

# # add velocity to the eye tracking data - when I add this inside a function it keeps running forever.
# eye_tracking_data_cm2deg_new = funcs.velocity(eye_tracking_data_cm2deg_new)

# # add acceleration to the eye tracking data - when I add this inside a function it keeps running forever.
# eye_tracking_data_cm2deg_new = funcs.acceleration(eye_tracking_data_cm2deg_new)


# Velocity (◦/s) and acceleration (◦/s2) of the gaze points. calculated using a Savitzky–Golay filter with polynomial
# order 2 and a window size of 12 ms—half the duration of shortest saccade, as suggested by Nystrom ¨ and Holmqvist (2010)
# SAVITZKY-GOLD filter: https://medium.com/pythoneers/introduction-to-the-savitzky-golay-filter-a-comprehensive-guide-using-python-b2dd07a8e2ce#:~:text=The%20Savitzky%2DGolay%20filter%20is,explained%20further%20in%20this%20post).


#calculate velocity - when I add this inside a function it keeps running forever.
preprocessed_et_data["velocity_deg_s"] = preprocessed_et_data['cm_to_deg_inside_VE']/preprocessed_et_data["time_diff"]
preprocessed_et_data["acceler_deg_s"] = preprocessed_et_data['velocity_deg_s']/preprocessed_et_data["time_diff"]


### feature importance !!!


################################# 3 MEAN DIFFF #########################################################################
######################################################################################################################


# 3- mean-diff - Distance (◦) between the mean gaze position in a 100-ms window before the sample and a 100-ms window after the sample.
# Pg. 15 Olsson 2007

# Take the r number of points that are before the sample (so the interval are 100ms) and r points after 
# take the average for both of them. measure the distance (in degrees). this is the mean-diff


xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']
win_4_x_y_z_cx_cy_cz = funcs_prep.calculate_average_window(preprocessed_et_data, xyzcxcycz, window=5) #changed from 4 to 5

preproc_add_mean_dist_m = funcs_feat.calc_mean_dist_m(win_4_x_y_z_cx_cy_cz) # in meters, convert then to degrees. 

# Convert MEAN-DIFF to degrees:
preproc_add_mean_dist_m_view_dist = funcs_feat.apply_viewing_distance_df(preproc_add_mean_dist_m)
preproc_add_mean_diff_degree = funcs_feat.mean_diff_degree_inside_VE(preproc_add_mean_dist_m_view_dist)


######################################################### 4 DISPERSION - DEGREES ##############################
################################################################################################################

# 4- disp - Dispersion (◦). Calculated as (xmax −xmin)+(ymax −ymin) over a 100-ms window. 


preproc_add_disp_m = funcs_feat.calculate_dispersion_meters(preproc_add_mean_diff_degree,'L_x', 'L_y', 'L_z' )
preproc_add_disp_degree = funcs_feat.convert_met_to_degree(preproc_add_disp_m, "disp_degree", 'dispersion_meters',"viewing_distance")




#################################### 5 MED-DIFF ####################################
##################################################################################


# 5 - med-diff - Distance (◦) between the median gaze position in a 100-ms window before the sample and a 100-ms window after the sample.
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden



preproc_add_disp_degree = pd.read_csv(config["prepr_and_features_file_updated"])


xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']
median_win5 = funcs_feat.calculate_median_window(preproc_add_disp_degree,xyzcxcycz, window=5)
median_dist_m = funcs_feat.calc_median_dist_m(median_win5) #function below a bit slow to run

#there is something wrong with calc_dist_function because viewing_dist values only increase.
#I dont think it harm the results because it is a long distance and it only changes slightly. Diff between smalles and bigger is 40cm.

view_dist_median_m = funcs_feat.apply_viewing_distance_df(median_dist_m, 'viewing_distance_median_wind', "L_x_Median_Before_Win5", "L_y_Median_Before_Win5", "L_z_Median_Before_Win5", "C_x_Median_Before_Win5" , "C_y_Median_Before_Win5", "C_z_Median_Before_Win5")
median_deg = funcs_feat.convert_met_to_degree(view_dist_median_m,"med_diff_deg", 'median_dist_m' , 'viewing_distance_median_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


############################### 6 STD ###########################################
##################################################################################

# 6 - std - Standard deviation (◦) of the recorded gaze position in a 100-ms window centered on the sample.  (Holmqvist et al., 2011) # book 152 dollars


median_deg = pd.read_csv(config["prepr_and_features_file_updated"])

std_m = funcs_feat.calculate_std_meters_win(median_deg,"L_x", "L_y", "L_z", window=5)
view_dist_std_m = funcs_feat.apply_viewing_distance_df(std_m, 'viewing_distance_std_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
std_deg = funcs_feat.convert_met_to_degree(view_dist_std_m,"std_deg", 'std_total_meters' , 'viewing_distance_std_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


############################### 7 STD-DIFF ###########################################
##################################################################################


# 14 - std - diff Difference in standard deviation (◦) between two 100-ms windows, 
# one before and one after the sample. Olsson (2007)

# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden


std_deg = pd.read_csv(config["prepr_and_features_file_updated"])

xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']
std_wind = funcs_feat.calculate_std_window(std_deg, xyzcxcycz, window = 5)
std_dist_wind_m = funcs_feat.calc_std_wind_dist_m(std_wind) #vectorized

#there is something wrong with calc_dist_function because viewing_dist values only increase.
#I dont think it harm the results because it is a long distance and it only changes slightly. Diff between smalles and bigger is 40cm.

view_dist_std_m = funcs_feat.apply_viewing_distance_df(std_dist_wind_m, 'viewing_distance_std_wind', "L_x_Std_Before_Win5", "L_y_Std_Before_Win5", "L_z_Std_Before_Win5", "C_x_Std_Before_Win5" , "C_y_Std_Before_Win5", "C_z_Std_Before_Win5") #(df, df_new_col, lx_col, ly_col, lz_col, cx_col, cy_col, cz_col)
std_diff_deg = funcs_feat.convert_met_to_degree(view_dist_std_m,"std_diff_deg", 'std_wind_dist_m' , 'viewing_distance_std_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)

std_diff_deg = std_deg.drop(index=0).reset_index(drop=True)

std_diff_deg = std_diff_deg.drop(columns=['L_x_Std_Before_Win5',
'L_x_Std_After_Win5', 'L_y_Std_Before_Win5', 'L_y_Std_After_Win5',
'L_z_Std_Before_Win5', 'L_z_Std_After_Win5', 'C_x_Std_Before_Win5',
'C_x_Std_After_Win5', 'C_y_Std_Before_Win5', 'C_y_Std_After_Win5',
'C_z_Std_Before_Win5', 'C_z_Std_After_Win5', 'std_wind_dist_m'])


 
############################### 8 RMS-DIFF ###########################################
##################################################################################

#rms-diff - Difference in root mean square (◦) between two 100-ms 
# windows before and after the sample. Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden

std_diff_deg = pd.read_csv(config["prepr_and_features_file_updated"])

xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']
rms_wind = funcs_feat.calculate_rms_window(std_diff_deg, xyzcxcycz, window = 5) # a bit slow to run
rms_dist_wind_m = funcs_feat.calc_feature_wind_dist_m(rms_wind, 'rms_wind_dist_m','L_x_Rms_Before_Win5','L_x_Rms_After_Win5','L_y_Rms_Before_Win5', 'L_y_Rms_After_Win5','L_z_Rms_Before_Win5', 'L_z_Rms_After_Win5' ) #vectorized #calc_feature_wind_dist_m(df, new_col_feature_dist_m, x_col_feature_bef_wind, x_col_feature_aft_wind, y_col_feature_bef_wind, y_col_feature_aft_wind, z_col_feature_bef_wind, z_col_feature_aft_wind) 

#there is something wrong with calc_dist_function because viewing_dist values only increase.
#I dont think it harm the results because it is a long distance and it only changes slightly. Diff between smalles and bigger is 40cm.

view_dist_rms_m = funcs_feat.apply_viewing_distance_df(rms_dist_wind_m, 'viewing_distance_rms_wind', "L_x_Rms_Before_Win5", "L_y_Rms_Before_Win5", "L_z_Rms_Before_Win5", "C_x_Rms_Before_Win5" , "C_y_Rms_Before_Win5", "C_z_Rms_Before_Win5") #(df, df_new_col, lx_col, ly_col, lz_col, cx_col, cy_col, cz_col)
rms_diff_deg = funcs_feat.convert_met_to_degree(view_dist_rms_m,"rms_diff_deg", 'rms_wind_dist_m' , 'viewing_distance_rms_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)
# rms_diff_deg = rms_deg.drop(index=0).reset_index(drop=True)

rms_diff_deg = rms_diff_deg.drop(columns=[
    'viewing_distance_rms_wind',
    'L_x_Rms_Before_Win5', 'L_x_Rms_After_Win5',
    'L_y_Rms_Before_Win5', 'L_y_Rms_After_Win5',
    'L_z_Rms_Before_Win5', 'L_z_Rms_After_Win5',
    'C_x_Rms_Before_Win5', 'C_x_Rms_After_Win5',
    'C_y_Rms_Before_Win5', 'C_y_Rms_After_Win5',
    'C_z_Rms_Before_Win5', 'C_z_Rms_After_Win5',
    'rms_wind_dist_m','rms_wind_dist_m'
])


# Select the columns to move to the end
columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

# Reorder the DataFrame
rms_diff_deg = rms_diff_deg[[col for col in rms_diff_deg.columns if col not in columns_to_move] + columns_to_move]



############################### 9 RMS ###########################################
##################################################################################

#zemblys 2017:
# Next to these,
# we also propose several new features, which we hypothesize are likely to be useful for the detection of the onset and offset
# of saccades: rms-diff, std-diff and bcea-diff. These new features are inspired by Olsson (2007) and are calculated
# by taking the difference in the RMS, STD, and BCEA precision measures calculated for 100-ms windows preceding
# and following the current sample. Obviously, the largest differences (and therefore peaks in the feature) should occur
# around the onset and offset of the saccades.

# Root mean square (◦) of the sample-to-sample displacement in a 100-ms window centered on a sample.  

rms_m = funcs_feat.calculate_rms_meters_win(rms_diff_deg,"L_x", "L_y", "L_z", window=5)
view_dist_rms_m = funcs_feat.apply_viewing_distance_df(rms_m, 'viewing_distance_rms_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
rms_deg = funcs_feat.convert_met_to_degree(view_dist_rms_m,"rms_deg", 'rms_total_meters' , 'viewing_distance_rms_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)

rms_deg = rms_deg.drop(columns=['rms_x_meters', 'rms_y_meters',
'rms_z_meters', 'rms_total_meters', 'viewing_distance_rms_wind'])
                       
# Select the columns to move to the end
columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

# Reorder the DataFrame
rms_deg = rms_deg[[col for col in rms_deg.columns if col not in columns_to_move] + columns_to_move]

                   
### save full file preproc and feature extracted ## CHANGE DF NAME AND JSON CSV FILE!!
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
rms_deg.to_csv(output_file_features_GTs_updated, index=False)

#### Save just features and GTs ## CHANGE DF NAME AND JSON CSV FILE!!
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', "med_diff_deg", "disp_degree",  "std_deg", 'std_diff_deg', 'rms_deg', 'rms_diff_deg','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
only_until_9_rms_and_GTs = rms_deg[columns_to_keep] # CHANGE DF NAME- HERE!

output_file_features_GTs_TEST = os.path.join(script_dir, config["only_extracted_features_and_GTs_TEST_file"])
only_until_9_rms_and_GTs.to_csv(output_file_features_GTs_TEST, index=False)
                      
  
############################### 10  BCEA  ###########################################
##################################################################################

# 10 - BCEA - Bivariate contour ellipse area (◦2). 
# Measures the area in which the recorded gaze position lies 
# within a 100-ms window in (P %) of the time. (Blignaut and Beelders, 2012)

# Blignaut, P., & Beelders, T. (2012). The precision of eye-trackers:
# a case for a new measure. In Proceedings of the symposium on
# eye tracking research and applications, ETRA ’12, (pp. 289–292).
# New York, NY, USA: ACM

# The bivariate contour ellipse (BCEA) is similar to CEP but it
# acknowledges that the geometrical shape of fixations might be
# elliptic. For a given proportion of samples,
 #  P = 1 -e**(-k)., k determines the proportion of samples to include in the covered area.
# If, for example, k=1, P = 63.2%. BCEA is then defined
# as 2 * k*  PI() * stdx * stdy * (1-p^2)**1/2 

#  where sx and sy denote the standard deviations in the x and y directions respectively and p is the
# Pearson correlation coefficient between the samples' x and y coordinates [Crossland and Rubin, 2002].
# The area of an ellipse can be expressed as A = PI()* a* b where a is  one half of the major diameter and b is one half of the minor
# diameter. Therefore, the following equality holds for an ellipse that includes 63.2% of the samples: 
    # ab = 2 *stdx *stdy * (1-p^2)**1/2 




rms_deg = pd.read_csv(config["prepr_and_features_file_updated"])

bcea_xy_m = funcs_feat.calculate_bcea_m_win(rms_deg,"L_x", "L_y", k=1, window=5) # 2D data #(df, dim1, dim2, k=1, window=5)

bcea_yz_m = funcs_feat.calculate_bcea_m_win(rms_deg,"L_y", "L_z", k=1, window=5) # 2D data

bcea_zx_m = funcs_feat.calculate_bcea_m_win(rms_deg,"L_z", "L_x", k=1, window=5)# 2D data

bcea_volume = funcs_feat.calculate_3d_bcea_rolling(rms_deg, "L_x", "L_y", "L_z", k=1, window=5) # 3D data (using x, y and z at the same time)

view_dist_bcea_m = funcs_feat.apply_viewing_distance_df(rms_m, 'viewing_distance_bcea_vol_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
rms_deg = funcs_feat.convert_met_to_degree(view_dist_rms_m,"bcea_vol_deg", 'bcea_vol_total_meters' , 'viewing_distance_bcea_vol_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)

rms_deg = rms_deg.drop(columns=['rms_x_meters', 'rms_y_meters',
'rms_z_meters', 'rms_total_meters', 'viewing_distance_rms_wind'])
                       
# Select the columns to move to the end
columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

# Reorder the DataFrame
rms_deg = rms_deg[[col for col in rms_deg.columns if col not in columns_to_move] + columns_to_move]




############################### 11 BCEA-DIFF  ###########################################
##################################################################################

# BCEA-DIFF Difference in bivariate contour ellipse area (◦2) 
#between two 100ms windows, one before and one after the sample. Olsson (2007)



######################

# 6 - fs - Sampling frequency (Hz). Mean sampling rate: 44.27 Hz  (SMI BeGaze)


#### 7 is not possible because we cannot detectate saccades, Idont have this data
# 7 - i2mc - A feature used to detect saccades in very noisy data. 
# The final weights come from the two-means clustering procedure as 
# per the original implementation by Hessels et al. (2016). 
# A 200-ms window was used, centered on the sample.
# Hessels, R. S., Niehorster, D. C., Kemner, C., & Hooge, I. T. C. (2016).
# Noise-robust fixation detection in eye movement data: Identification by two-means clustering (i2mc). Behavior Research Methods,
# 1–22. doi:10.3758/s13428-016-0822-1




# 10 - Rayleightest - Tests whether the sample-to-sample directions in
# a 22-ms window are uniformly distributed. Larsson et al. (2015)
# Larsson, L., Nystrom, M., & Stridh, M. (2013). Detection of sac- ¨
# cades and postsaccadic oscillations in the presence of smooth
# pursuit. IEEE Transactions on Biomedical Engineering, 60(9),
# 2484–2493.







# Example DataFrame
data2 = {'observer': [1, 2, 3, 4, 5, 6, 7.1, 8,9,10,11,12,13,14]}
median_degTEST = pd.DataFrame(data2)

# Identify rows where the observer value is a float and its decimal part is not 0
rows_to_remove = median_degTEST[median_degTEST['observer'] % 1 != 0].index

# Extend the range to include 5 rows before and 5 rows after
extended_rows_to_remove = set()
for idx in rows_to_remove:
    extended_rows_to_remove.update(range(max(0, idx - 5), min(len(median_degTEST), idx + 6)))

# Create a new DataFrame excluding these rows
median_degTEST_no_float_zero = median_degTEST.drop(index=extended_rows_to_remove).reset_index(drop=True)

print("Original DataFrame:")
print(median_degTEST)
print("\nFiltered DataFrame:")
print(median_degTEST_no_float_zero)


# clean columns not used for other features and save data:
    

 
### save full file preproc and feature extracted ## CHANGE DF NAME AND JSON CSV FILE!!
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
std_diff_deg.to_csv(output_file_features_GTs_updated, index=False)

