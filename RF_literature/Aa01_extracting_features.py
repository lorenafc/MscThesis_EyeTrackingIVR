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
import Aa01_TEST_frequenc_funcs_extracting_features


with open('config.json') as json_file:
    config = json.load(json_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  


preprocessed_et_data = pd.read_csv(config["preprocessed_data_file"])



####################################### VELOCITY AND ACCELERATION #################################################################
###################################################################################################################################


#calculate velocity - when I add this inside a function it keeps running forever.
preprocessed_et_data["velocity_deg_s"] = preprocessed_et_data['cm_to_deg_inside_VE']/preprocessed_et_data["time_diff"]
preprocessed_et_data["acceler_deg_s"] = preprocessed_et_data['velocity_deg_s']/preprocessed_et_data["time_diff"]




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


columns_to_drop = [
    'viewing_distance_rms_wind',
    'L_x_Rms_Before_Win5', 'L_x_Rms_After_Win5',
    'L_y_Rms_Before_Win5', 'L_y_Rms_After_Win5',
    'L_z_Rms_Before_Win5', 'L_z_Rms_After_Win5',
    'C_x_Rms_Before_Win5', 'C_x_Rms_After_Win5',
    'C_y_Rms_Before_Win5', 'C_y_Rms_After_Win5',
    'C_z_Rms_Before_Win5', 'C_z_Rms_After_Win5',
    'rms_wind_dist_m', 'rms_wind_dist_m'
]

rms_diff_deg = drop_and_reorder_columns(rms_diff_deg, columns_to_drop)



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


rms_deg = pd.read_csv(config["prepr_and_features_file_updated"])


#### 2D #####

## all axis: xy, yz,zx:
bcea_xy_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_x", "L_y", k=1, window=5) # 2D data
bcea_yz_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_y", "L_z", k=1, window=5) # 2D data
bcea_zx_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_z", "L_x", k=1, window=5) # 2D data

# view_dist_bcea_m = funcs_feat.apply_viewing_distance_df(bcea_yz_corr, 'viewing_distance_bcea_2d_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
# bcea_deg = funcs_feat.convert_met_to_degree(view_dist_bcea_m,"bcea_2d_deg", 'bcea_LyLz', 'viewing_distance_bcea_2d_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
bcea_xy_yz_zx = bcea_zx_corr[[col for col in bcea_zx_corr.columns if col not in columns_to_move] + columns_to_move]

### save full file preproc and feature extracted ## CHANGE DF NAME AND JSON CSV FILE!!
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated_bcea_complete"])
bcea_xy_yz_zx.to_csv(output_file_features_GTs_updated, index=False)


## only yz:
    
bcea_yz_only = bcea_zx_corr.drop(columns=['bcea_L_xL_y', 'bcea_L_zL_x' ])
columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
bcea_yz_only = bcea_yz_only[[col for col in bcea_yz_only.columns if col not in columns_to_move] + columns_to_move]

### save full file preproc and feature extracted ## CHANGE DF NAME AND JSON CSV FILE!!
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated_bcea_yz"])
bcea_yz_only.to_csv(output_file_features_GTs_updated, index=False)

test_only_yz_rf = bcea_yz_only[['bcea_L_yL_z', 'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]
funcs_feat.save_df(test_only_yz_rf, "data/Aa01_test_only_yz_GTs_rf.csv")

test_only_yz_rf.columns

test_xy_xz_zx_rf = bcea_zx_corr[["bcea_L_xL_y", "bcea_L_yL_z", "bcea_L_zL_x",'bcea_L_yL_z', 'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]
funcs_feat.save_df(test_xy_xz_zx_rf, "data/Aa01_test_xy_yz_zx_rf.csv")

######### 3D ###################

std_xyz = funcs_feat.calculate_std_only_m_win(rms_deg, "L_x" , "L_y","L_z", window=5, center=True)

pearson_xy = funcs_feat.calculate_pearson(rms_deg, "L_x" , "L_y","L_z",  window=5)
pearson_yz = funcs_feat.calculate_pearson(pearson_xy, "L_y","L_z", "L_x" , window=5)
pearson_zx = funcs_feat.calculate_pearson(pearson_yz, "L_z" , "L_x","L_y", window=5)

pearson_xy.columns

bcea_3d = funcs_feat.calculate_bcea_volume(pearson_zx, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x") 

only_bcea_3d_GTs = bcea_3d[['bcea_3d','GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']] 
funcs_feat.save_df(only_bcea_3d_GTs, "data/Aa01_test_only_bcea3d_GTs_rf.csv")

# Example values for k and Pearson correlations
k = 1.0
pearson_xy = 0.5
pearson_yz = 0.6
pearson_zx = 0.7

# Calculate the volume
volume = funcs_feat.calculate_bcea_volume(k, pearson_xy, pearson_yz, pearson_zx)
print(f"The calculated volume is: {volume}")


## combine xy plane with 3D 


bcea_xy = pd.read_csv("data/Aa01_test_only_yz_GTs_rf.csv") #config["prepr_and_features_file_updated"]"")


col_bcea_yz = bcea_xy["bcea_L_yL_z"]

col_bcea_3d = bcea_3d["bcea_3d"]
 

# preprocessing + 3d +yz
bcea_3d_yz = pd.concat([bcea_3d, col_bcea_yz], axis=1)
funcs_feat.save_df(bcea_3d_yz, "data/Aa01_test_preproc_bcea3d_yz.csv")

# only yz 3d and gts for test in the rf model
bcea_yz_3d_GT_only = pd.concat([bcea_xy, col_bcea_3d], axis=1)

second_col = bcea_yz_3d_GT_only.pop('bcea_3d')
bcea_yz_3d_GT_only.insert(1, 'bcea_3d', second_col)

funcs_feat.save_df(bcea_yz_3d_GT_only, "data/Aa01_test_only_bcea_yz_3d_GTs_rf.csv")

# bcea_3d with noise

bcea_3d_noise = funcs_feat.calculate_bcea_volume_noise(bcea_3d, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x") 

# extract only 3d_noise for RF

bcea_3d_noise_rf = bcea_3d_noise[["bcea_3d_noise", 'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]
funcs_feat.save_df(only_bcea_3d_GTs, "data/Aa01_test_only_bcea3d_noise_GTs_rf.csv")







############################### 11 BCEA-DIFF  ###########################################
##################################################################################

# BCEA-DIFF Difference in bivariate contour ellipse area (◦2) 
#between two 100ms windows, one before and one after the sample. Olsson (2007)


bcea_diff_xy = funcs_feat.calculate_bcea2d_window(bcea_xy_yz_zx, "L_x", "L_y", k=1, window=5) #
bcea_diff_yz = funcs_feat.calculate_bcea2d_window(bcea_diff_xy, "L_y", "L_z", k=1, window=5)
bcea_diff_zx = funcs_feat.calculate_bcea2d_window(bcea_diff_yz, "L_z", "L_x", k=1, window=5)

non_zero_count = (bcea_diff_zx["bcea_L_xL_y"] != 0).sum()
print(f"Number of non-zero values: {non_zero_count}") #Number of non-zero values: 8711 = 8.2%

#calc distance before and after
bcea_dist_wind_m_xy_yz_zx = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx, 'bcea_diff_wind_dist_m',  'bcea_L_xL_y_Before_Win5',
'bcea_L_xL_y_After_Win5', 'bcea_L_yL_z_Before_Win5',
'bcea_L_yL_z_After_Win5', 'bcea_L_zL_x_Before_Win5',
'bcea_L_zL_x_After_Win5')


# view_dist_bcea_diff_m = funcs_feat.apply_viewing_distance_df(rms_m, 'viewing_distance_bcea_diff_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
bcea_diff_deg = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx,"bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


bcea_diff_xy_xz_zx_deg_m_clean = funcs_feat.drop_and_reorder_columns(bcea_diff_deg, ['bcea_L_xL_y_Before_Win5',
'bcea_L_xL_y_After_Win5', 'bcea_L_yL_z_Before_Win5',
'bcea_L_yL_z_After_Win5', 'bcea_L_zL_x_Before_Win5',
'bcea_L_zL_x_After_Win5'])

bcea_diff_xy_xz_zx_deg_clean = bcea_diff_xy_xz_zx_deg_m_clean.drop(columns=['bcea_diff_wind_dist_m'])




#### Save just features and GTs ## CHANGE DF NAME AND JSON CSV FILE!!
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', "med_diff_deg", "disp_degree",  "std_deg", 'std_diff_deg', 'rms_deg', 'rms_diff_deg','bcea_L_xL_y', 'bcea_L_yL_z', 'bcea_L_zL_x', 'bcea_diff_deg', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7' ]
only_until_11_bceadiff_and_GTs = bcea_diff_xy_xz_zx_deg_clean[columns_to_keep] 






######################## # 12 - Rayleightest - Tests whether the sample-to-sample directions in
# a 22-ms window are uniformly distributed. Larsson et al. (2015)
# Larsson, L., Nystrom, M., & Stridh, M. (2013). Detection of sac- ¨
# cades and postsaccadic oscillations in the presence of smooth
# pursuit. IEEE Transactions on Biomedical Engineering, 60(9),
# 2484–2493.

## A novel algorithm for detection of saccades and postsaccadic oscillations in the presence of 
# smooth pursuit movements is proposed. The method combines saccade detection in the acceleration domain 
# with specialized on- and offset criteria for saccades and postsaccadic oscillations. 


# 6 - fs - Sampling frequency (Hz). Mean sampling rate: 44.27 Hz  (SMI BeGaze)



#### 7 is not possible because we cannot detectate saccades, Idont have this data
# 7 - i2mc - A feature used to detect saccades in very noisy data. 
# The final weights come from the two-means clustering procedure as 
# per the original implementation by Hessels et al. (2016). 
# A 200-ms window was used, centered on the sample.
# Hessels, R. S., Niehorster, D. C., Kemner, C., & Hooge, I. T. C. (2016).
# Noise-robust fixation detection in eye movement data: Identification by two-means clustering (i2mc). Behavior Research Methods,
# 1–22. doi:10.3758/s13428-016-0822-1




# clean columns not used for other features and save data:
    

 
### save full file preproc and feature extracted ## CHANGE DF NAME AND JSON CSV FILE!!
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
std_diff_deg.to_csv(output_file_features_GTs_updated, index=False)

