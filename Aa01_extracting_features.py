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

# path_file = os.path.join(script_dir, config["data_path"])

preprocessed_et_data = pd.read_csv(config["preprocessed_data_file"])



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



### feature importance !!!



################################# MEAN DIFFF ###################################
##################################################################################


# 8 - mean-diff - Distance (◦) between the mean gaze position in a 
# 100-ms window before the sample and a 100-ms window after the sample.
# Pg. 15 Olsson 2007

# Take the r number of points that are before the sample (so the interval are 100ms) and r points after 
# take the average for both of them. measure the distance (in degrees). this is the mean-diff

preprocessed_and_features = pd.read_csv(config["prepr_and_features_file"])

xyzcxcycz = ['L_x', 'L_y', 'L_z','C_x', 'C_y', 'C_z']
win_4_x_y_z_cx_cy_cz = funcs_prep.calculate_average_window(preprocessed_and_features, xyzcxcycz, window=4)

preproc_add_mean_dist_m = funcs_feat.calc_mean_dist_m(win_4_x_y_z_cx_cy_cz) # in meters, convert then to degrees. 


# Convert MEAN-DIFF to degrees:

preproc_add_mean_dist_m_view_dist = funcs_feat.apply_viewing_distance_df(preproc_add_mean_dist_m)
preproc_add_mean_diff_degree = funcs_feat.mean_diff_degree_inside_VE(preproc_add_mean_dist_m_view_dist)

output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
preproc_add_mean_diff_degree.to_csv(output_file_features_GTs_updated, index=False)

columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg','GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
only_vel_acc_mean_diff_and_GTs = preproc_add_mean_diff_degree[columns_to_keep]

output_file_features_GTs_TEST = os.path.join(script_dir, config["only_extracted_features_and_GTs_TEST_file"])
only_vel_acc_mean_diff_and_GTs.to_csv(output_file_features_GTs_TEST, index=False)


preprocessed_and_features.columns 



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

preproc_and_features_meandiff = pd.read_csv(config["prepr_and_features_file_updated"])


preproc_add_disp_m = funcs_feat.calculate_dispersion_meters(preproc_and_features_meandiff,'L_x', 'L_y', 'L_z' )

preproc_add_disp_degree = funcs_feat.convert_met_to_degree(preproc_add_disp_m, "disp_degree", 'dispersion_meters',"viewing_distance")


### save full file preproc and feature extracted 
output_file_features_GTs_updated = os.path.join(script_dir, config["prepr_and_features_file_updated"])
preproc_add_disp_m.to_csv(output_file_features_GTs_updated, index=False)

#### Save just features and GTs ## CHANGE DF NAME AND JSON CSV FILE!!
columns_to_keep = ['velocity_deg_s', 'acceler_deg_s', 'mean_diff_deg', "disp_degree",  'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
only_vel_acc_mean_diff_disp_and_GTs = preproc_add_disp_m[columns_to_keep]

output_file_features_GTs_TEST = os.path.join(script_dir, config["only_extracted_features_and_GTs_TEST_file"])
only_vel_acc_mean_diff_disp_and_GTs.to_csv(output_file_features_GTs_TEST, index=False)

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






