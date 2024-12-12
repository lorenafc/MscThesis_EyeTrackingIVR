# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:33:02 2024

@author: Lorena Carpes
"""

def calc_viewing_distance(lx,ly,lz,cx,cy,cz):
    
    import math
    
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
    
    import pandas as pd
    
    if "time_diff" not in df.columns:
        df["time_diff"] = ""         
        df["time_diff"] = df["time"].diff().fillna(0)
        df["time_diff"] = pd.to_numeric(df["time_diff"], errors='coerce')
        
    return df
   
def calc_coordinates(df):
    
    import math
    import pandas as pd
    
    if "coordinates" not in df.columns:        
        df["coordinates"] = ""
            
    df["coordinates"] = df.apply(lambda row: (row["L_x"], row["L_y"], row["L_z"]), axis=1)
 
    # df["coordinates"][0], df["coordinates"][1], df["coordinates"][2] = df["L_x"], df["L_y"],df["L_z"]
        
    # for gaze in range(1, len(df)):       
    #     x, y, z  = df.iloc[gaze]["L_x"], df.iloc[gaze]["L_y"], df.iloc[gaze]["L_z"]
    #     df.at[gaze, "coordinates"] = (x, y, z)
        
    return df   

def calc_coordinates_dist(df): # a bit slow to run
    
    import math
    import pandas as pd
    
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
 
def convert_cm_to_degree_inside_VE(df):
    
    
    if "cm_to_deg_inside_VE" not in df.columns:
        df["cm_to_deg_inside_VE"] = np.nan   
        
    df['coordinates_dist'] = pd.to_numeric(df['coordinates_dist'], errors='coerce')
    df["viewing_distance"] = pd.to_numeric(df["viewing_distance"], errors='coerce')

        # Drop rows where 'coordinates_dist' or 'viewing_distance' is NaN
    df = df.dropna(subset=['coordinates_dist', 'viewing_distance']).reset_index(drop=True)
    
    # Calculate cm_to_deg_inside_VE using vectorized operations
    df["cm_to_deg_inside_VE"] = (180 / np.pi * np.arctan(df["coordinates_dist"] / (2 * df["viewing_distance"])))
    df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
        
    return df
    
def calc_velocity_deg_s(df):
    
    import numpy as np
    
    # if "velocity" not in df.columns:
    #     df["velocity"] = np.nan  
    
    df = df[df["time_diff"] != 0].reset_index(drop=True)
    
    # df['cm_to_deg_inside_VE'] = pd.to_numeric(df['cm_to_deg_inside_VE'], errors='coerce')
    # df["time_diff"] = pd.to_numeric(df["time_diff"], errors='coerce')
       
    df["velocity"] = df['cm_to_deg_inside_VE']/df["time_diff"]
           
    # for gaze in range(0, len(df)):
    #     velocity = df.iloc[gaze]['cm_to_deg_inside_VE']/df.iloc[gaze]["time_diff"]
    #     df.at[gaze,'velocity'] = velocity  
    
    return df