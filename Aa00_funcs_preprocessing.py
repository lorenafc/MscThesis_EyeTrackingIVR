# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 

@author: Lorena Carpes
"""

import math
import pandas as pd
import numpy as np

def calc_viewing_distance(lx,ly,lz,cx,cy,cz):   
    
    sqr_dist_x = (cx-lx)**2
    sqr_dist_y = (cy-ly)**2
    sqr_dist_z = (cz-lz)**2

    sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
    viewing_distance = round(math.sqrt(sum_square_distances),4)
    
    return viewing_distance

def apply_viewing_distance_df(df):
    
    if "viewing_distance" not in df.columns:
        df["viewing_distance"] = ""
    
    for gaze, row in df.iterrows():
        viewing_distance = calc_viewing_distance(row["L_x"], row["L_y"],row["L_z"], row["C_x"],row["C_y"], row["C_z"])    
     
        df.at[gaze,'viewing_distance'] = viewing_distance
        
    return df

def calc_time_diff(df):
    
    if "time_diff" not in df.columns:
        df["time_diff"] = ""         
        df["time_diff"] = df["time"].diff().fillna(0)
        df["time_diff"] = pd.to_numeric(df["time_diff"], errors='coerce')
        
    return df
   
def calc_coordinates(df):
    
    if "coordinates" not in df.columns:        
        df["coordinates"] = ""
            
    df["coordinates"] = df.apply(lambda row: (row["L_x"], row["L_y"], row["L_z"]), axis=1)
 
    # df["coordinates"][0], df["coordinates"][1], df["coordinates"][2] = df["L_x"], df["L_y"],df["L_z"]
        
    # for gaze in range(1, len(df)):       
    #     x, y, z  = df.iloc[gaze]["L_x"], df.iloc[gaze]["L_y"], df.iloc[gaze]["L_z"]
    #     df.at[gaze, "coordinates"] = (x, y, z)
        
    return df   

def calc_coordinates_dist(df): # a bit slow to run
    
    if "coordinates_dist" not in df.columns:
        df["coordinates_dist"] = ""
        
    for gaze in range(1, len(df)):
        
        x, y, z  = df.iloc[gaze]["L_x"], df.iloc[gaze]["L_y"], df.iloc[gaze]["L_z"]
        prev_x, prev_y, prev_z = df.iloc[gaze-1]["L_x"], df.iloc[gaze-1]["L_y"], df.iloc[gaze-1]["L_z"]
    
        sqr_dist_x = (x - prev_x)**2
        sqr_dist_y = (y - prev_y)**2
        sqr_dist_z = (z - prev_z)**2

        sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
        coordinates_distance = round(math.sqrt(sum_square_distances), 4)
       
        df.at[gaze,'coordinates_dist'] = coordinates_distance   
        df['coordinates_dist'] = pd.to_numeric(df['coordinates_dist'], errors='coerce')
      
    return df
 
def convert_cm_to_degree_inside_VE(df): # based on GazeParser library, by Hiroyuki Sogo.
    
    if "cm_to_deg_inside_VE" not in df.columns:
        df["cm_to_deg_inside_VE"] = np.nan   
        
    df['coordinates_dist'] = pd.to_numeric(df['coordinates_dist'], errors='coerce')
    df["viewing_distance"] = pd.to_numeric(df["viewing_distance"], errors='coerce')


    df = df.dropna(subset=['coordinates_dist', 'viewing_distance']).reset_index(drop=True)
    
    # Calculate cm_to_deg_inside_VE using vectorized operations
    df["cm_to_deg_inside_VE"] = (180 / np.pi * np.arctan(df["coordinates_dist"] / (2 * df["viewing_distance"])))
    df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
        
    return df


def reorder_columns(df):

    columns_to_move = ['observer', 'GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
    new_column_order = [col for col in df.columns if col not in columns_to_move] + columns_to_move
    df = df[new_column_order]
    return df


def process_eye_tracking_data(df):

    eye_tracking_data_view_dist = apply_viewing_distance_df(df)
    eye_tracking_data_time_diff = calc_time_diff(eye_tracking_data_view_dist)
    eye_tracking_data_coord = calc_coordinates(eye_tracking_data_time_diff)
    eye_tracking_data_coord_dist = calc_coordinates_dist(eye_tracking_data_coord)  # slow function

    # Convert distance from cm to degrees inside the VE
    eye_tracking_data_cm2deg = convert_cm_to_degree_inside_VE(eye_tracking_data_coord_dist)    
    eye_tracking_data_cm2deg_ordered = reorder_columns(eye_tracking_data_cm2deg)
    
    return eye_tracking_data_cm2deg_ordered

#convert data to 100Hz 