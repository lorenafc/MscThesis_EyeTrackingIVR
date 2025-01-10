# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""
import math
import pandas as pd
import numpy as np

######## VELOCITY AND ACCELERATION ##########

# def velocity(df): # for some reason this function is not working and the code keeps running forever

#     df["velocity_deg_s"] = df["cm_to_deg_inside_VE"] / df["time_diff"]
#     return df

# def acceleration(df):  # for some reason this function is not working and the code keeps running forever


#     df["acceler_deg_s"] = df["velocity_deg_s"] / df["time_diff"]
#     return df



######### MEAN DIFF ##############




def calc_mean_dist_m(df): # a bit slow to run
    
    if "mean_dist_m" not in df.columns:
        df["mean_dist_m"] = ""
        
    for gaze in range(1, len(df)):
        
        x, y, z  = df.iloc[gaze]["L_x_Avg_After_Win4"], df.iloc[gaze]["L_y_Avg_After_Win4"], df.iloc[gaze]["L_z_Avg_After_Win4"]
        prev_x, prev_y, prev_z = df.iloc[gaze]["L_x_Avg_Before_Win4"], df.iloc[gaze]["L_y_Avg_Before_Win4"], df.iloc[gaze]["L_z_Avg_Before_Win4"]
    
        sqr_dist_x = (x - prev_x)**2
        sqr_dist_y = (y - prev_y)**2
        sqr_dist_z = (z - prev_z)**2

        sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
        mean_distance_meters = round(math.sqrt(sum_square_distances), 4)
       
        df.at[gaze,'mean_dist_m'] = mean_distance_meters   
        df['mean_dist_m'] = pd.to_numeric(df['mean_dist_m'], errors='coerce')
      
    return df    


def calc_viewing_distance(lx,ly,lz,cx,cy,cz):    # Calculate only before
    
    sqr_dist_x = (cx-lx)**2
    sqr_dist_y = (cy-ly)**2
    sqr_dist_z = (cz-lz)**2

    sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
    viewing_distance = round(math.sqrt(sum_square_distances),4)
    
    return viewing_distance

def apply_viewing_distance_df(df): # calculate only viewing dist before to use in the formula to convert the mean_dist_m to mean_diff in degrees.
    
    if "viewing_distance_avg_wind" not in df.columns:
        df["viewing_distance_avg_wind"] = ""
    
    for gaze, row in df.iterrows():
        viewing_distance = calc_viewing_distance(row["L_x_Avg_Before_Win4"], row["L_y_Avg_Before_Win4"],row["L_z_Avg_Before_Win4"], row["C_x_Avg_Before_Win4"],row["C_y_Avg_Before_Win4"], row["C_z_Avg_Before_Win4"])    
     
        df.at[gaze,'viewing_distance_avg_wind'] = viewing_distance
    df['viewing_distance_avg_wind'] = df['viewing_distance_avg_wind'].round(4)
        
    return df

    
def mean_diff_degree_inside_VE(df): # based on GazeParser library, by Hiroyuki Sogo.
    
    if "mean_diff_deg" not in df.columns:
        df["mean_diff_deg"] = np.nan   
        
    df['mean_dist_m'] = pd.to_numeric(df['mean_dist_m'], errors='coerce')
    df["viewing_distance_avg_wind"] = pd.to_numeric(df["viewing_distance_avg_wind"], errors='coerce')


    df = df.dropna(subset=['mean_dist_m', "viewing_distance_avg_wind"]).reset_index(drop=True)
    
    # Calculate cm_to_deg_inside_VE using vectorized operations
    df["mean_diff_deg"] = (180 / np.pi * np.arctan(df['mean_dist_m'] / (2 * df["viewing_distance_avg_wind"])))
    df["mean_diff_deg"] = pd.to_numeric(df["mean_diff_deg"], errors='coerce')
    
    # Round mean_diff_deg to 4 decimal places
    df["mean_diff_deg"] = df["mean_diff_deg"].round(4)
        
    return df


######## DISPERSION #################

def calculate_dispersion_meters(df, x_column, y_column, z_column, window=5, center = True): # window = 5 

    df['dispersion_meters'] = (
        df[x_column].rolling(window, center=center).max() - df[x_column].rolling(window, center=center).min() +
        df[y_column].rolling(window, center=center).max() - df[y_column].rolling(window, center=center).min() 
        +  df[z_column].rolling(window, center=center).max() - df[z_column].rolling(window, center=center).min()# 
       
    )
    
    return df


def apply_viewing_distance_df(df, df_new_col, lx_col, ly_col, lz_col, cx_col, cy_col, cz_col):

    if df_new_col not in df.columns:
        df[df_new_col] = ""

    for gaze, row in df.iterrows():
        viewing_distance = calc_viewing_distance(
            row[lx_col],
            row[ly_col],
            row[lz_col],
            row[cx_col],
            row[cy_col],
            row[cz_col]
        )
        df.at[gaze, df_new_col] = viewing_distance

    # Round the entire column after the loop
    df[df_new_col] = pd.to_numeric(df[df_new_col], errors='coerce').round(4)

    return df


def convert_met_to_degree(df, df_new_col_deg, col_dist_meters, col_view_dist): # based on GazeParser library, by Hiroyuki Sogo.
    
    if df_new_col_deg not in df.columns:
        df[df_new_col_deg] = np.nan   
        
    df[col_dist_meters] = pd.to_numeric(df[col_dist_meters], errors='coerce')
    df[col_view_dist] = pd.to_numeric(df[col_view_dist], errors='coerce')


    df = df.dropna(subset=[col_dist_meters, col_view_dist]).reset_index(drop=True)
    
    # Calculate cm_to_deg_inside_VE using vectorized operations
    df[df_new_col_deg] = (180 / np.pi * np.arctan(df[col_dist_meters] / (2 * df[col_view_dist])))
    df[df_new_col_deg] = pd.to_numeric(df[df_new_col_deg], errors='coerce')
    
    # Round mean_diff_deg to 4 decimal places
    df[df_new_col_deg] = df[df_new_col_deg].round(4)
        
    return df
