# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes


This scrip was made based on GazeParser library, by Hiroyuki Sogo. 

.. Part of GazeParser package.
.. Copyright (C) 2012-2015 Hiroyuki Sogo.
.. Distributed under the terms of the GNU General Public License (GPL).

website: https://gazeparser.sourceforge.net/#

Sogo, H. (2013) GazeParser: an open-source and multiplatform library for low-cost eye tracking and analysis. 
Behavior Reserch Methods 45, pp 684-695, doi:10.3758/s13428-012-0286-x

"""

import numpy as np
import os
import pandas as pd

import funcs_random_forest_literature as funcs

# set up environment
# gets the current working directory
script_dir =  r"\path\to\your\script" #os.getcwd()

# change to the script's working directory
os.chdir(script_dir)

path_file = ".\path\to\your\file.csv"
eye_tracking_data = pd.read_csv(path_file)


eye_tracking_data_view_dist = apply_viewing_distance_df(eye_tracking_data)   
eye_tracking_data_time_diff = calc_time_diff(eye_tracking_data_view_dist)
eye_tracking_data_coord = calc_coordinates(eye_tracking_data_time_diff)
eye_tracking_data_coord_dist = calc_coordinates_dist(eye_tracking_data_coord)  # function a bit slow to run

eye_tracking_data_coord_dist['coordinates_dist'].head()

eye_tracking_data_cm2deg = convert_cm_to_degree_inside_VE(eye_tracking_data_coord_dist)

eye_tracking_data_cm2deg[['cm_to_deg_inside_VE', "time_diff"]].head()

print(eye_tracking_data_cm2deg[['cm_to_deg_inside_VE', 'time_diff']].dtypes)
print(eye_tracking_data_cm2deg[["cm_to_deg_inside_VE", "time_diff"]].isnull().sum())

print((eye_tracking_data_cm2deg["cm_to_deg_inside_VE"] == 0).sum())  
print((eye_tracking_data_cm2deg["time_diff"] == 0).sum())            

eye_tracking_data_cm2deg = calc_velocity_deg_s(eye_tracking_data_coord_dist)

eye_tracking_data_cm2deg = eye_tracking_data_cm2deg[(eye_tracking_data_cm2deg["cm_to_deg_inside_VE"] != 0) & (eye_tracking_data_cm2deg["time_diff"] != 0)].copy()

print("Number of rows with zero values in 'cm_to_deg_inside_VE':", (eye_tracking_data_cm2deg["cm_to_deg_inside_VE"] == 0).sum())
print("Number of rows with zero values in 'time_diff':", (eye_tracking_data_cm2deg["time_diff"] == 0).sum())


eye_tracking_data_cm2deg["velocity"].head()
eye_tracking_data_cm2deg[['cm_to_deg_inside_VE',"time_diff" ]].head()

######### save the df with this new feature

# eye_tracking_data_path_csv = (os.path.join(script_dir, "output", "LLA2020_labeled_feature_eng.csv"))
# eye_tracking_data_path_excel = (os.path.join(script_dir, "output", "LLA2020_labeled_feature_eng.xlsx"))

# eye_tracking_data.to_csv(eye_tracking_data_path_csv, index=False)
# eye_tracking_data.to_excel(eye_tracking_data_path_excel, index=False)

#######


# maybe use viewing distance for the autoencoders also
# gaze_viewing_distance = []


# Convert the relevant columns to NumPy arrays
# elapsed_time = eye_tracking_data['time'].values  # Converts the 'time' column to a NumPy array
# gaze_positions = eye_tracking_data[['L_x', 'L_y', 'L_z']].values  
# observer_positions = eye_tracking_data[['C_x', 'C_y','C_z']].values 

# # Check the shapes to verify
# print("Elapsed Time Array Shape:", elapsed_time.shape)
# print("Gaze Positions Array Shape:", gaze_positions.shape)
# print("Observer Positions Array Shape:", observer_positions.shape)

# Velocity (◦/s) and acceleration (◦/s2) of the gaze points.
# calculated using a Savitzky–Golay filter with polynomial
# order 2 and a window size of 12 ms—half the duration of shortest saccade, as suggested by Nystrom ¨
# and Holmqvist (2010)

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
# 100-ms window centered on a sample.  (Holmqvist et al., 2011)
# Holmqvist, K., Andersson, R., Jarodzka, H., Kok, E., Nystrom, M., ¨
# & Dewhurst, R. (2016). Eye tracking. A comprehensive guide to
# methods and measures. Oxford: Oxford University Press


# 12 - rms-diff - Difference in root mean square (◦) between two 100-ms 
# windows before and after the sample. Olsson (2007)
# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden


# 13 - std - Standard deviation (◦) of the recorded gaze position in a 
# 100-ms window centered on the sample.  (Holmqvist et al., 2011)
# Holmqvist, K., Andersson, R., Jarodzka, H., Kok, E., Nystrom, M., ¨
# & Dewhurst, R. (2016). Eye tracking. A comprehensive guide to
# methods and measures. Oxford: Oxford University Press



# 14 - std - diff Difference in standard deviation (◦) between two 100-ms windows, 
# one before and one after the sample. Olsson (2007)

# Olsson, P. (2007). Real-time and offline filters for eye tracking. Master’s thesis, Royal Institute of Technology, Stockholm, Sweden






