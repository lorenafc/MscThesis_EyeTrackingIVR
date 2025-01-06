# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""


from Aa00_preprocessing import eye_tracking_data_cm2deg_new #preprocessed gaze data


def velocity(df):

    df["velocity_deg_s"] = df["cm_to_deg_inside_VE"] / df["time_diff"]
    return df

def acceleration(df):

    df["acceler_deg_s"] = df["velocity_deg_s"] / df["time_diff"]
    return df

