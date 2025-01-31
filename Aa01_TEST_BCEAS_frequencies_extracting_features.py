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


preprocessed_et_data = pd.read_csv("data/Aa00_preprocessed_eye_tracking_freq.csv")




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

funcs_feat.save_df(preprocessed_et_data, "data/Aa01_et_data_all_freq_vel_acc.csv") 


### DO IT FOR ALL 5 FREQUENCIES




############################### 9 RMS ###########################################
##################################################################################



#### SAVE

funcs_feat.save_df(rms_deg_44, "data/Aa00_preprocessed_eye_tracking_44Hz_9rms_deg.csv")
funcs_feat.save_df(rms_deg_87, "data/Aa00_preprocessed_eye_tracking_87Hz_9rms_deg.csv")
funcs_feat.save_df(rms_deg_130, "data/Aa00_preprocessed_eye_tracking_130Hz_9rms_deg.csv")
funcs_feat.save_df(rms_deg_174, "data/Aa00_preprocessed_eye_tracking_174Hz_9rms_deg.csv")
funcs_feat.save_df(rms_deg_217, "data/Aa00_preprocessed_eye_tracking_217Hz_9rms_deg.csv")   

############################### 10  BCEA  ###########################################
##################################################################################

# 10 - BCEA - Bivariate contour ellipse area (◦2). 
# Measures the area in which the recorded gaze position lies 
# within a 100-ms window in (P %) of the time. (Blignaut and Beelders, 2012)


rms_deg = pd.read_csv(config["prepr_and_features_file_updated"])


#### 2D #####


## all axis: xy, yz,zx:
bcea_xy_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_x", "L_y", 5, k=1) # 2D data
bcea_yz_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_y", "L_z", 5, k=1) # 2D data
bcea_zx_corr = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg, "L_z", "L_x", 5, k=1) # 2D data

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

bcea_diff = pd.read_csv("data/Aa01_extracted_features_eye_tracking_updated_cleaned_bcea_yz.csv")

std_xyz = funcs_feat.calculate_std_only_m_win(bcea_diff, "L_x" , "L_y","L_z", window=5, center=True)


pearson_xy = funcs_feat.calculate_pearson(bcea_diff, "L_x" , "L_y","L_z",  window=5)
pearson_yz = funcs_feat.calculate_pearson(pearson_xy, "L_y","L_z", "L_x" , window=5)
pearson_zx = funcs_feat.calculate_pearson(pearson_yz, "L_z" , "L_x","L_y", window=5)

pearson_xy.columns

bcea_3d = funcs_feat.calculate_bcea_volume(pearson_zx, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x") 

only_bcea_3d_GTs = bcea_3d[['bcea_3d','GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']] 
# funcs_feat.save_df(only_bcea_3d_GTs, "data/Aa01_test_only_bcea3d_GTs_rf.csv")


# bcea_3d with noise


bcea_3d_noise = funcs_feat.calculate_bcea_volume_noise(bcea_3d, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x") 

# extract only 3d_noise for RF

bcea_3d_noise_rf = bcea_3d_noise[["bcea_3d_noise", 'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]
funcs_feat.save_df(bcea_3d_noise_rf, "data/Aa01_test_only_bcea3d_noise_GTs_rf.csv")



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

####### NOW FOR ALL FREQUENCIES:

rms_deg_44 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_44Hz_9rms_deg.csv")
rms_deg_87 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_87Hz_9rms_deg.csv")
rms_deg_130 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_130Hz_9rms_deg.csv")
rms_deg_174 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_174Hz_9rms_deg.csv")
rms_deg_217 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_217Hz_9rms_deg.csv")



#### CODE

### 2d

columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']

bcea_yz_corr_44 = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg_44, "L_y", "L_z", 5, k=1) # 2D data
bcea_yz_44 = bcea_yz_corr_44[[col for col in bcea_yz_corr_44.columns if col not in columns_to_move] + columns_to_move]

# bcea_deg = funcs_feat.convert_met_to_degree(view_dist_bcea_m,"bcea_2d_deg", 'bcea_LyLz', 'viewing_distance' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


# view_dist_bcea_m = funcs_feat.apply_viewing_distance_df(bcea_yz_corr, 'viewing_distance_bcea_2d_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
# bcea_deg = funcs_feat.convert_met_to_degree(view_dist_bcea_m,"bcea_2d_deg", 'bcea_LyLz', 'viewing_distance_bcea_2d_wind' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)

# For 87Hz, window 10
bcea_yz_corr_87 = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg_87, "L_y", "L_z", 10, k=1)
bcea_yz_87 = bcea_yz_corr_87[[col for col in bcea_yz_corr_87.columns if col not in columns_to_move] + columns_to_move]

# For 130Hz, window 14
bcea_yz_corr_130 = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg_130, "L_y", "L_z", 14, k=1)
bcea_yz_130 = bcea_yz_corr_130[[col for col in bcea_yz_corr_130.columns if col not in columns_to_move] + columns_to_move]

# For 174Hz, window 18
bcea_yz_corr_174 = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg_174, "L_y", "L_z", 18, k=1)
bcea_yz_174 = bcea_yz_corr_174[[col for col in bcea_yz_corr_174.columns if col not in columns_to_move] + columns_to_move]

# For 217Hz, window 23
bcea_yz_corr_217 = funcs_feat.calculate_bcea2d_m_win_corrcoef(rms_deg_217, "L_y", "L_z", 23, k=1)
bcea_yz_217 = bcea_yz_corr_217[[col for col in bcea_yz_corr_217.columns if col not in columns_to_move] + columns_to_move]

###

#SAVE YZ:
    
funcs_feat.save_df(bcea_yz_44, "data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_yz.csv")
funcs_feat.save_df(bcea_yz_87, "data/Aa00_preprocessed_eye_tracking_87Hz_10_bcea_yz.csv")
funcs_feat.save_df(bcea_yz_130, "data/Aa00_preprocessed_eye_tracking_130Hz_10_bcea_yz.csv")
funcs_feat.save_df(bcea_yz_174, "data/Aa00_preprocessed_eye_tracking_174Hz_10_bcea_yz.csv")
funcs_feat.save_df(bcea_yz_217, "data/Aa00_preprocessed_eye_tracking_217Hz_10_bcea_yz.csv")

#### 3D with noise

## 44Hz
std_xyz_44 = funcs_feat.calculate_std_only_m_win(bcea_yz_44, "L_x" , "L_y","L_z", 5, center=True)

pearson_xy_44 = funcs_feat.calculate_pearson(std_xyz_44, "L_x" , "L_y","L_z",  5)
pearson_yz_44 = funcs_feat.calculate_pearson(pearson_xy_44, "L_y","L_z", "L_x" , 5)
pearson_zx_44 = funcs_feat.calculate_pearson(pearson_yz_44, "L_z" , "L_x","L_y", 5)

bcea_3d_noise_44 = funcs_feat.calculate_bcea_volume_noise(pearson_zx_44, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x") 

# For 87Hz, window 10
std_xyz_87 = funcs_feat.calculate_std_only_m_win(bcea_yz_87, "L_x", "L_y", "L_z", 10, center=True)
pearson_xy_87 = funcs_feat.calculate_pearson(std_xyz_87, "L_x", "L_y", "L_z", 10)
pearson_yz_87 = funcs_feat.calculate_pearson(pearson_xy_87, "L_y", "L_z", "L_x", 10)
pearson_zx_87 = funcs_feat.calculate_pearson(pearson_yz_87, "L_z", "L_x", "L_y", 10)
bcea_3d_noise_87 = funcs_feat.calculate_bcea_volume_noise(pearson_zx_87, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x")


# For 130Hz, window 14
std_xyz_130 = funcs_feat.calculate_std_only_m_win(bcea_yz_130, "L_x", "L_y", "L_z", 14, center=True)
pearson_xy_130 = funcs_feat.calculate_pearson(std_xyz_130, "L_x", "L_y", "L_z", 14)
pearson_yz_130 = funcs_feat.calculate_pearson(pearson_xy_130, "L_y", "L_z", "L_x", 14)
pearson_zx_130 = funcs_feat.calculate_pearson(pearson_yz_130, "L_z", "L_x", "L_y", 14)
bcea_3d_noise_130 = funcs_feat.calculate_bcea_volume_noise(pearson_zx_130, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x")


# For 174Hz, window 18
std_xyz_174 = funcs_feat.calculate_std_only_m_win(bcea_yz_174, "L_x", "L_y", "L_z", 18, center=True)
pearson_xy_174 = funcs_feat.calculate_pearson(std_xyz_174, "L_x", "L_y", "L_z", 18)
pearson_yz_174 = funcs_feat.calculate_pearson(pearson_xy_174, "L_y", "L_z", "L_x", 18)
pearson_zx_174 = funcs_feat.calculate_pearson(pearson_yz_174, "L_z", "L_x", "L_y", 18)
bcea_3d_noise_174 = funcs_feat.calculate_bcea_volume_noise(pearson_zx_174, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x")


# For 217Hz, window 23
std_xyz_217 = funcs_feat.calculate_std_only_m_win(bcea_yz_217, "L_x", "L_y", "L_z", 23, center=True)
pearson_xy_217 = funcs_feat.calculate_pearson(std_xyz_217, "L_x", "L_y", "L_z", 23)
pearson_yz_217 = funcs_feat.calculate_pearson(pearson_xy_217, "L_y", "L_z", "L_x", 23)
pearson_zx_217 = funcs_feat.calculate_pearson(pearson_yz_217, "L_z", "L_x", "L_y", 23)
bcea_3d_noise_217 = funcs_feat.calculate_bcea_volume_noise(pearson_zx_217, "std_x_m", "std_y_m", "std_z_m", "pearson_L_x_L_y", "pearson_L_y_L_z", "pearson_L_z_L_x")


## CLEAN



###SAVE
funcs_feat.save_df(bcea_3d_noise_44, "data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_xyyzzx_3d_noise.csv")
funcs_feat.save_df(bcea_3d_noise_87, "data/Aa00_preprocessed_eye_tracking_87Hz_10_bcea_xyyzzx_3d_noise.csv")
funcs_feat.save_df(bcea_3d_noise_130, "data/Aa00_preprocessed_eye_tracking_130Hz_10_bcea_xyyzzx_3d_noise.csv")
funcs_feat.save_df(bcea_3d_noise_174, "data/Aa00_preprocessed_eye_tracking_174Hz_10_bcea_xyyzzx_3d_noise.csv")
funcs_feat.save_df(bcea_3d_noise_217, "data/Aa00_preprocessed_eye_tracking_217Hz_10_bcea_xyyzzx_3d_noise.csv")


################ 

########## ADD VELOCITY AND ACCELERATION M/S

bcea_3d_noise_44["velocity_m_s"] = bcea_3d_noise_44['coordinates_dist']/bcea_3d_noise_44["time_diff"]
bcea_3d_noise_44["acceler_m_s"] = bcea_3d_noise_44['velocity_m_s']/bcea_3d_noise_44["time_diff"]

# For 87Hz
bcea_3d_noise_87["velocity_m_s"] = bcea_3d_noise_87['coordinates_dist'] / bcea_3d_noise_87["time_diff"]
bcea_3d_noise_87["acceler_m_s"] = bcea_3d_noise_87['velocity_m_s'] / bcea_3d_noise_87["time_diff"]

# For 130Hz
bcea_3d_noise_130["velocity_m_s"] = bcea_3d_noise_130['coordinates_dist'] / bcea_3d_noise_130["time_diff"]
bcea_3d_noise_130["acceler_m_s"] = bcea_3d_noise_130['velocity_m_s'] / bcea_3d_noise_130["time_diff"]

# For 174Hz
bcea_3d_noise_174["velocity_m_s"] = bcea_3d_noise_174['coordinates_dist'] / bcea_3d_noise_174["time_diff"]
bcea_3d_noise_174["acceler_m_s"] = bcea_3d_noise_174['velocity_m_s'] / bcea_3d_noise_174["time_diff"]

# For 217Hz
bcea_3d_noise_217["velocity_m_s"] = bcea_3d_noise_217['coordinates_dist'] / bcea_3d_noise_217["time_diff"]
bcea_3d_noise_217["acceler_m_s"] = bcea_3d_noise_217['velocity_m_s'] / bcea_3d_noise_217["time_diff"]

funcs_feat.save_df(bcea_3d_noise_44, "data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_xyyzzx_3d_noise_vel_ms.csv")
funcs_feat.save_df(bcea_3d_noise_87, "data/Aa00_preprocessed_eye_tracking_87Hz_10_bcea_xyyzzx_3d_noise_vel_ms.csv")
funcs_feat.save_df(bcea_3d_noise_130, "data/Aa00_preprocessed_eye_tracking_130Hz_10_bcea_xyyzzx_3d_noise_vel_ms.csv")
funcs_feat.save_df(bcea_3d_noise_174, "data/Aa00_preprocessed_eye_tracking_174Hz_10_bcea_xyyzzx_3d_noise_vel_ms.csv")
funcs_feat.save_df(bcea_3d_noise_217, "data/Aa00_preprocessed_eye_tracking_217Hz_10_bcea_xyyzzx_3d_noise_vel_ms.csv")

################ convert BCEA TO DEGREE

bcea_deg_44 = funcs_feat.convert_met_to_degree(bcea_3d_noise_44,"bcea_2d_yz_deg", 'bcea_L_yL_z', 'viewing_distance' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)
bcea_deg_3d_44 = funcs_feat.convert_met_to_degree(bcea_deg_44,"bcea_3d_deg", 'bcea_3d_noise', 'viewing_distance' )

bcea_deg_87 = funcs_feat.convert_met_to_degree(bcea_3d_noise_87, "bcea_2d_yz_deg", 'bcea_L_yL_z', 'viewing_distance')
bcea_deg_3d_87 = funcs_feat.convert_met_to_degree(bcea_deg_87, "bcea_3d_deg", 'bcea_3d_noise', 'viewing_distance')


bcea_deg_130 = funcs_feat.convert_met_to_degree(bcea_3d_noise_130, "bcea_2d_yz_deg", 'bcea_L_yL_z', 'viewing_distance')
bcea_deg_3d_130 = funcs_feat.convert_met_to_degree(bcea_deg_130, "bcea_3d_deg", 'bcea_3d_noise', 'viewing_distance')

bcea_deg_174 = funcs_feat.convert_met_to_degree(bcea_3d_noise_174, "bcea_2d_yz_deg", 'bcea_L_yL_z', 'viewing_distance')
bcea_deg_3d_174 = funcs_feat.convert_met_to_degree(bcea_deg_174, "bcea_3d_deg", 'bcea_3d_noise', 'viewing_distance')


bcea_deg_217 = funcs_feat.convert_met_to_degree(bcea_3d_noise_217, "bcea_2d_yz_deg", 'bcea_L_yL_z', 'viewing_distance')
bcea_deg_3d_217 = funcs_feat.convert_met_to_degree(bcea_deg_217, "bcea_3d_deg", 'bcea_3d_noise', 'viewing_distance')


funcs_feat.save_df(bcea_deg_3d_44, "data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
funcs_feat.save_df(bcea_deg_3d_87, "data/Aa00_preprocessed_eye_tracking_87Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
funcs_feat.save_df(bcea_deg_3d_130, "data/Aa00_preprocessed_eye_tracking_130Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
funcs_feat.save_df(bcea_deg_3d_174, "data/Aa00_preprocessed_eye_tracking_174Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
funcs_feat.save_df(bcea_deg_3d_217, "data/Aa00_preprocessed_eye_tracking_217Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")

############################### 11 BCEA-DIFF  ###########################################
##################################################################################

# BCEA-DIFF Difference in bivariate contour ellipse area (◦2) 
#between two 100ms windows, one before and one after the sample. Olsson (2007)


bcea_diff_xy = funcs_feat.calculate_bcea2d_window(bcea_xy_yz_zx, "L_x", "L_y", k=1, window=5)
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

## only beca m and deg with GTs

# bcea_diff_xy_xz_zx_m_GTs = bcea_diff_xy_xz_zx_deg_m_clean[["bcea_diff_deg",'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]
# bcea_diff_xy_xz_zx_deg_GTs = bcea_diff_xy_xz_zx_deg_m_clean[['bcea_diff_wind_dist_m', 'GT1','GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]

# funcs_feat.save_df(bcea_diff_xy_xz_zx_m_GTs, "data/Aa01_test_bcea_diff_xy_xz_zx_m_GTs.csv")
# funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_clean, "data/Aa01_bcea_diff_xy_xz_zx_deg_GTs.csv")

####

funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_clean, config["prepr_and_features_file_updated"])

#### Save just features and GTs ## CHANGE DF NAME AND JSON CSV FILE!!
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', "med_diff_deg", "disp_degree",  "std_deg", 'std_diff_deg', 'rms_deg', 'rms_diff_deg','bcea_L_xL_y', 'bcea_L_yL_z', 'bcea_L_zL_x', 'bcea_diff_deg', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7' ]
only_until_11_bceadiff_and_GTs = bcea_diff_xy_xz_zx_deg_clean[columns_to_keep] # CHANGE DF NAME- HERE!

# bcea_diff_xy_xz_zx_deg_feat_GTs = funcs_feat.df_features_GTs(bcea_diff_xy_xz_zx_deg_clean,['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', "med_diff_deg", "disp_degree",  "std_deg", 'std_diff_deg', 'rms_deg', 'rms_diff_deg','bcea_L_xL_y', 'bcea_L_yL_z', 'bcea_L_zL_x', 'bcea_diff_deg', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7'])

funcs_feat.save_df(only_until_11_bceadiff_and_GTs, config["only_extracted_features_and_GTs_TEST_file"])



########## WITH 5 FREQUENCIES: ### bcea in deg and vel and acc m/s
    
bcea_deg_3d_44 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv") "data/Aa00_preprocessed_eye_tracking_44Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv"
bcea_deg_3d_87 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_87Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
bcea_deg_3d_130 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_130Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
bcea_deg_3d_174 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_174Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")
bcea_deg_3d_217 = pd.read_csv("data/Aa00_preprocessed_eye_tracking_217Hz_10_bcea_xyyzzx_3d_noise_deg_vel_ms.csv")    
    
    
### CODE


# 44Hz

bcea_diff_xy_44 = funcs_feat.calculate_bcea2d_window(bcea_deg_3d_44, "L_x", "L_y", 5, k=1)
bcea_diff_yz_44 = funcs_feat.calculate_bcea2d_window(bcea_diff_xy_44, "L_y", "L_z", 5, k=1)
bcea_diff_zx_44 = funcs_feat.calculate_bcea2d_window(bcea_diff_yz_44, "L_z", "L_x", 5, k=1)


#calc distance before and after
bcea_dist_wind_m_xy_yz_zx_44 = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx_44, 'bcea_diff_wind_dist_m',  'bcea_L_xL_y_Before_Win5',
'bcea_L_xL_y_After_Win5', 'bcea_L_yL_z_Before_Win5',
'bcea_L_yL_z_After_Win5', 'bcea_L_zL_x_Before_Win5',
'bcea_L_zL_x_After_Win5')


# view_dist_bcea_diff_m = funcs_feat.apply_viewing_distance_df(rms_m, 'viewing_distance_bcea_diff_wind', "L_x", "L_y", "L_z",'C_x','C_y','C_z')
bcea_diff_deg_44 = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx_44,"bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance' ) #(df, df_new_col_deg, col_dist_meters, col_view_dist)


## clean

bcea_diff_xy_xz_zx_deg_m_clean_44 = funcs_feat.drop_and_reorder_columns(bcea_diff_deg_44, ['bcea_L_xL_y_Before_Win5',
'bcea_L_xL_y_After_Win5', 'bcea_L_yL_z_Before_Win5',
'bcea_L_yL_z_After_Win5', 'bcea_L_zL_x_Before_Win5',
'bcea_L_zL_x_After_Win5'])


# 87Hz, window = 10
bcea_diff_xy_87 = funcs_feat.calculate_bcea2d_window(bcea_deg_3d_87, "L_x", "L_y", 10, k=1)
bcea_diff_yz_87 = funcs_feat.calculate_bcea2d_window(bcea_diff_xy_87, "L_y", "L_z", 10, k=1)
bcea_diff_zx_87 = funcs_feat.calculate_bcea2d_window(bcea_diff_yz_87, "L_z", "L_x", 10, k=1)

bcea_dist_wind_m_xy_yz_zx_87 = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx_87, 'bcea_diff_wind_dist_m', 'bcea_L_xL_y_Before_Win10',
'bcea_L_xL_y_After_Win10', 'bcea_L_yL_z_Before_Win10',
'bcea_L_yL_z_After_Win10', 'bcea_L_zL_x_Before_Win10',
'bcea_L_zL_x_After_Win10')

bcea_diff_deg_87 = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx_87, "bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance')

bcea_diff_xy_xz_zx_deg_m_clean_87 = funcs_feat.drop_and_reorder_columns(bcea_diff_deg_87, ['bcea_L_xL_y_Before_Win10',
'bcea_L_xL_y_After_Win10', 'bcea_L_yL_z_Before_Win10',
'bcea_L_yL_z_After_Win10', 'bcea_L_zL_x_Before_Win10',
'bcea_L_zL_x_After_Win10'])

# 130Hz, window = 14
bcea_diff_xy_130 = funcs_feat.calculate_bcea2d_window(bcea_deg_3d_130, "L_x", "L_y", 14, k=1)
bcea_diff_yz_130 = funcs_feat.calculate_bcea2d_window(bcea_diff_xy_130, "L_y", "L_z", 14, k=1)
bcea_diff_zx_130 = funcs_feat.calculate_bcea2d_window(bcea_diff_yz_130, "L_z", "L_x", 14, k=1)

bcea_dist_wind_m_xy_yz_zx_130 = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx_130, 'bcea_diff_wind_dist_m', 'bcea_L_xL_y_Before_Win14',
'bcea_L_xL_y_After_Win14', 'bcea_L_yL_z_Before_Win14',
'bcea_L_yL_z_After_Win14', 'bcea_L_zL_x_Before_Win14',
'bcea_L_zL_x_After_Win14')

bcea_diff_deg_130 = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx_130, "bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance')

bcea_diff_xy_xz_zx_deg_m_clean_130 = funcs_feat.drop_and_reorder_columns(bcea_diff_deg_130, ['bcea_L_xL_y_Before_Win14',
'bcea_L_xL_y_After_Win14', 'bcea_L_yL_z_Before_Win14',
'bcea_L_yL_z_After_Win14', 'bcea_L_zL_x_Before_Win14',
'bcea_L_zL_x_After_Win14'])

# 174Hz, window = 18
bcea_diff_xy_174 = funcs_feat.calculate_bcea2d_window(bcea_deg_3d_174, "L_x", "L_y", 18, k=1)
bcea_diff_yz_174 = funcs_feat.calculate_bcea2d_window(bcea_diff_xy_174, "L_y", "L_z", 18, k=1)
bcea_diff_zx_174 = funcs_feat.calculate_bcea2d_window(bcea_diff_yz_174, "L_z", "L_x", 18, k=1)

bcea_dist_wind_m_xy_yz_zx_174 = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx_174, 'bcea_diff_wind_dist_m', 'bcea_L_xL_y_Before_Win18',
'bcea_L_xL_y_After_Win18', 'bcea_L_yL_z_Before_Win18',
'bcea_L_yL_z_After_Win18', 'bcea_L_zL_x_Before_Win18',
'bcea_L_zL_x_After_Win18')

bcea_diff_deg_174 = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx_174, "bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance')

bcea_diff_xy_xz_zx_deg_m_clean_174 = funcs_feat.drop_and_reorder_columns(bcea_diff_deg_174, ['bcea_L_xL_y_Before_Win18',
'bcea_L_xL_y_After_Win18', 'bcea_L_yL_z_Before_Win18',
'bcea_L_yL_z_After_Win18', 'bcea_L_zL_x_Before_Win18',
'bcea_L_zL_x_After_Win18'])

# 217Hz, window = 23
bcea_diff_xy_217 = funcs_feat.calculate_bcea2d_window(bcea_deg_3d_217, "L_x", "L_y", 23, k=1)
bcea_diff_yz_217 = funcs_feat.calculate_bcea2d_window(bcea_diff_xy_217, "L_y", "L_z", 23, k=1)
bcea_diff_zx_217 = funcs_feat.calculate_bcea2d_window(bcea_diff_yz_217, "L_z", "L_x", 23, k=1)

bcea_dist_wind_m_xy_yz_zx_217 = funcs_feat.calc_feature_wind_dist_m(bcea_diff_zx_217, 'bcea_diff_wind_dist_m', 'bcea_L_xL_y_Before_Win23',
'bcea_L_xL_y_After_Win23', 'bcea_L_yL_z_Before_Win23',
'bcea_L_yL_z_After_Win23', 'bcea_L_zL_x_Before_Win23',
'bcea_L_zL_x_After_Win23')

bcea_diff_deg_217 = funcs_feat.convert_met_to_degree(bcea_dist_wind_m_xy_yz_zx_217, "bcea_diff_deg", 'bcea_diff_wind_dist_m', 'viewing_distance')

bcea_diff_xy_xz_zx_deg_m_clean_217 = funcs_feat.drop_and_reorder_columns(bcea_diff_deg_217, ['bcea_L_xL_y_Before_Win23',
'bcea_L_xL_y_After_Win23', 'bcea_L_yL_z_Before_Win23',
'bcea_L_yL_z_After_Win23', 'bcea_L_zL_x_Before_Win23',
'bcea_L_zL_x_After_Win23'])


## SAVE:
    

funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_m_clean_44, "data/Aa00_preproc_44Hz_all_features_extr.csv")
funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_m_clean_87, "data/Aa00_preproc_87Hz_all_features_extr.csv")
funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_m_clean_130, "data/Aa00_preproc_130Hz_all_features_extr.csv")
funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_m_clean_174, "data/Aa00_preproc_174Hz_all_features_extr.csv")
funcs_feat.save_df(bcea_diff_xy_xz_zx_deg_m_clean_217, "data/Aa00_preproc_217Hz_all_features_extr.csv")



non_zero_count = (bcea_diff_zx["bcea_L_xL_y"] != 0).sum()
print(f"Number of non-zero values: {non_zero_count}") #Number of non-zero values: 8711 = 8.2%



################### AFTER EXTRACTING ALL FEATURES OF THE DATA OF THE 5 FREQUENCIES: 

    
## CONCAT THEM ALL TOGETHER - needs to be done after extracting all features - because of the windows.

### concat after all the columns have the same name:
bcea_diff_xy_xz_zx_deg_m_clean_44 = bcea_diff_xy_xz_zx_deg_m_clean_44.rename(columns={'med_diff_deg_44Hz': 'med_diff_deg'})
bcea_diff_xy_xz_zx_deg_m_clean_87 = bcea_diff_xy_xz_zx_deg_m_clean_87.rename(columns={'med_diff_deg_87Hz': 'med_diff_deg'})    
bcea_diff_xy_xz_zx_deg_m_clean_130 = bcea_diff_xy_xz_zx_deg_m_clean_130.rename(columns={'med_diff_deg_130Hz': 'med_diff_deg'})
bcea_diff_xy_xz_zx_deg_m_clean_174 = bcea_diff_xy_xz_zx_deg_m_clean_174.rename(columns={'med_diff_deg_174Hz': 'med_diff_deg'})
bcea_diff_xy_xz_zx_deg_m_clean_217 = bcea_diff_xy_xz_zx_deg_m_clean_217.rename(columns={'med_diff_deg_217Hz': 'med_diff_deg'})
  
bcea_diff_xy_xz_zx_deg_m_clean_44 = bcea_diff_xy_xz_zx_deg_m_clean_44.drop(columns=['bcea_2d_deg'])
   
vertical_concat_all_wrong_order_observ = pd.concat([bcea_diff_xy_xz_zx_deg_m_clean_44, bcea_diff_xy_xz_zx_deg_m_clean_87, bcea_diff_xy_xz_zx_deg_m_clean_130, bcea_diff_xy_xz_zx_deg_m_clean_174, bcea_diff_xy_xz_zx_deg_m_clean_217], axis=0, ignore_index=True)   
# et_data_concat_observer_right_order = vertical_concat_all_wrong_order_observ.sort_values(by=['observer'])



vertical_concat_all_wrong_reset = vertical_concat_all_wrong_order_observ.reset_index()
et_data_concat_observer_right_order = vertical_concat_all_wrong_reset.sort_values(['observer', 'index'], ascending=[True, True]) # https://stackoverflow.com/questions/17141558/how-to-sort-a-pandas-dataframe-by-two-or-more-columns


funcs_feat.save_df(et_data_concat_observer_right_order, "data/prA01_et_data_all_features_extracted_all_freq_CONCAT.csv")

####  select only the features for the RF model:

#degree
et_deg_all_feat_GT_concat_RF = et_data_concat_observer_right_order[['velocity_deg_s','acceler_deg_s',  'mean_diff_deg', 'disp_degree', 'med_diff_deg', 'std_deg', 'std_diff_deg', 'rms_diff_deg', 'rms_deg','bcea_3d_noise', 'bcea_2d_yz_deg', 'bcea_diff_deg','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']] #bcea_diff2d

funcs_feat.save_df(et_deg_all_feat_GT_concat_RF, "data/RF/prA01_et_data_all_features_extracted_all_freq_CONCAT_RF.csv")

### m/s - only std_diff is in degree
et_m_s_all_feat_GT_concat_RF = et_data_concat_observer_right_order[['velocity_m_s', 'acceler_m_s','mean_dist_m','dispersion_meters','median_dist_m', 'std_wind_dist_m','std_diff_deg','rms_wind_dist_m','rms_total_meters','bcea_L_yL_z','bcea_3d_noise','bcea_diff_wind_dist_m','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']]

et_m_s_all_feat_GT_concat_RF = et_m_s_all_feat_GT_concat_RF.rename(columns={ 'mean_dist_m':'mean_diff_m', 'median_dist_m':'median_diff_m', 'std_total_meters':'std_m', 'std_wind_dist_m' :'std_diff_m', 'rms_wind_dist_m':'rms_diff_m','rms_total_meters':'rms_m', 'bcea_3d_noise':'bcea_3d_noise_m', 'bcea_diff_wind_dist_m':'bcea_diff_m' })

funcs_feat.save_df(et_m_s_all_feat_GT_concat_RF, "data/RF/prA01_et_data_all_features_extracted_all_freq_CONCAT_m_s_RF.csv")





#### After adding different frequencies to the original dataset:

    
eye_tracking_data = pd.read_csv("data/Aa01_et_data_concat_observer_right_order_count_freq_checked.csv")
 
eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)
funcs_feat.save_df(eye_tracking_data_cm2deg_new, "data/Aa00_preprocessed_eye_tracking_freq.csv")
funcs_feat.save_df(eye_tracking_data_cm2deg_new, config["preprocessed_data_file"])



###### remove windows lines from begin and end of each observer.

### WHEN YOURE DONE CALCULATE AGAIN VELOCITY AND ACCELERATION IN METERS AND THEN SPLIT 2 DFS FOR RF: 1 METERS AND OTHER DEGREE






## CONCAT THEM ALL TOGETHER
vertical_concat_all_wrong_order_observ = pd.concat([et_data_44Hz_original, et_data_87Hz_reset_index, et_data_130Hz_reset_index, et_data_174Hz_reset_index, et_data_217Hz_reset_index], axis=0, ignore_index=True)   
# et_data_concat_observer_right_order = vertical_concat_all_wrong_order_observ.sort_values(by=['observer'])

vertical_concat_all_wrong_reset = vertical_concat_all_wrong_order_observ.reset_index()
et_data_concat_observer_right_order = vertical_concat_all_wrong_reset.sort_values(['observer', 'index'], ascending=[True, True]) # https://stackoverflow.com/questions/17141558/how-to-sort-a-pandas-dataframe-by-two-or-more-columns

funcs_feat.save_df(et_data_concat_observer_right_order, "data/Aa01_et_data_concat_12ALL_FREQUENCIES_and_features.csv")
 
  
 
    
 
    

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




### 



