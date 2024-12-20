# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""

from Aa00_preprocessing import eye_tracking_data_cm2deg_new #preprocessed gaze data


#calculate velocity deg/s
eye_tracking_data_cm2deg_new ["velocity_deg_s"] = eye_tracking_data_cm2deg_new ['cm_to_deg_inside_VE']/eye_tracking_data_cm2deg_new ["time_diff"]

#calculate acceleration deg/s 
eye_tracking_data_cm2deg_new ["acceler_deg_s"] = eye_tracking_data_cm2deg_new ['velocity_deg_s']/eye_tracking_data_cm2deg_new ["time_diff"]


