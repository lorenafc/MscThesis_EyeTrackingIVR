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

print(preprocessed_et_data.head())

print("teste A0a01")


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

output_file = os.path.join(script_dir, config["extracted_features_file"])
extracted_features.to_csv(output_file, index=False)





 
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




