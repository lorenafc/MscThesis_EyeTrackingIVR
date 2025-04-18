# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""
import math
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


##### GENERIC FUNCTIONS

def drop_and_reorder_columns(df, columns_to_drop):
  
    # Drop the specified columns
    df = df.drop(columns=columns_to_drop)
    
    columns_to_move = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
    
    # Reorder the DataFrame
    df = df[[col for col in df.columns if col not in columns_to_move] + columns_to_move]
    
    return df

def df_features_GTs(df,columns_to_keep):

    featuers_and_GTs = df[columns_to_keep]
    
    return df
 

# def df_features_GTs(df, columns_to_keep):

#     GTs = ['GT1', 'GT2', 'GT3', 'GT4', 'GT5', 'GT6', 'GT7']
#     all_columns_to_keep = list(set(columns_to_keep + GTs))
#     features_and_GTs = df[[all_columns_to_keep]]

#     return features_and_GTs


def save_df(df, file_path):

    df.to_csv(file_path, index=False)


######## VELOCITY AND ACCELERATION ##########

# def velocity(df): # for some reason this function is not working and the code keeps running forever,
# so it is in the main script instead of the function script

#     df["velocity_deg_s"] = df["cm_to_deg_inside_VE"] / df["time_diff"]
#     return df

# def acceleration(df):  # for some reason this function is not working and the code keeps running forever

#     df["acceler_deg_s"] = df["velocity_deg_s"] / df["time_diff"]
#     return df



######### MEAN DIFF ##############


def calculate_average_window(df, columns, window):  # win = for the others changed from 4 to 5 to be close to 100ms (approx 90ms)

    for column in columns:
        results = []
        for row_index in range(len(df)):
            # Get the range before and after the row
            start_before = max(0, row_index - window)
            end_after = min(len(df), row_index + window + 1)

            # Samples before
            samples_before = df[start_before:row_index]
            avg_before = np.nanmean(samples_before[column]) if len(samples_before) > 0 else np.nan

            # Samples after
            samples_after = df[row_index + 1:end_after]
            avg_after = np.nanmean(samples_after[column]) if len(samples_after) > 0 else np.nan

            # Append the result as a tuple (avg_before, avg_after)
            results.append((avg_before, avg_after))

        # Add the results to the DataFrame
        df[f'{column}_Avg_Before_Win{window}'] = [result[0] for result in results]
        df[f'{column}_Avg_After_Win{window}'] = [result[1] for result in results]

    return df


def calc_mean_dist_m(df, window): # a bit slow to run
    
    if "mean_dist_m" not in df.columns:
        df["mean_dist_m"] = ""
        
    for gaze in range(1, len(df)):
        
        x, y, z  = df.iloc[gaze][f"L_x_Avg_After_Win{window}"], df.iloc[gaze][f"L_y_Avg_After_Win{window}"], df.iloc[gaze][f"L_z_Avg_After_Win{window}"]
        prev_x, prev_y, prev_z = df.iloc[gaze][f"L_x_Avg_Before_Win{window}"], df.iloc[gaze][f"L_y_Avg_Before_Win{window}"], df.iloc[gaze][f"L_z_Avg_Before_Win{window}"]
    
        sqr_dist_x = (x - prev_x)**2
        sqr_dist_y = (y - prev_y)**2
        sqr_dist_z = (z - prev_z)**2

        sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
        mean_distance_meters = round(math.sqrt(sum_square_distances), 4)
       
        df.at[gaze,'mean_dist_m'] = mean_distance_meters   
        df['mean_dist_m'] = pd.to_numeric(df['mean_dist_m'], errors='coerce')
      
    return df    


def calc_viewing_distance(lx,ly,lz,cx,cy,cz):    
# Best to always use this function to calculate viewing distance. 
# Instead of using the values of the window before
    
    sqr_dist_x = (cx-lx)**2
    sqr_dist_y = (cy-ly)**2
    sqr_dist_z = (cz-lz)**2

    sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
    viewing_distance = round(math.sqrt(sum_square_distances),4)
    
    #df["viewing_distance"] = viewing_distance
    
    # return df
    
    return viewing_distance

def apply_viewing_distance_df_mean_diff(df, window): # calculate only viewing dist before to use in the formula to convert the mean_dist_m to mean_diff in degrees.
    
    if "viewing_distance_avg_wind" not in df.columns:
        df["viewing_distance_avg_wind"] = ""
        
#try the vectorized approach below instead of the for loop with iterrows.
#     df["viewing_distance_avg_wind"] = np.sqrt(
#     (df["C_x_Avg_Before_Win4"] - df["L_x_Avg_Before_Win4"])**2 +
#     (df["C_y_Avg_Before_Win4"] - df["L_y_Avg_Before_Win4"])**2 +
#     (df["C_z_Avg_Before_Win4"] - df["L_z_Avg_Before_Win4"])**2
# ).round(4)

    
    for gaze, row in df.iterrows(): # altered this part of the function to consider only L_X y anz y, and not the values before win4
        viewing_distance = calc_viewing_distance(row[f"L_x_Avg_Before_Win{window}"], row[f"L_y_Avg_Before_Win{window}"],row[f"L_z_Avg_Before_Win{window}"], row[f"C_x_Avg_Before_Win{window}"],row[f"C_y_Avg_Before_Win{window}"], row[f"C_z_Avg_Before_Win{window}"])    
     
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

def calculate_dispersion_meters(df, x_column, y_column, z_column, window, center = True): # window = 5 

    df['dispersion_meters'] = (
        df[x_column].rolling(window, center=center).max() - df[x_column].rolling(window, center=center).min() +
        df[y_column].rolling(window, center=center).max() - df[y_column].rolling(window, center=center).min() 
        +  df[z_column].rolling(window, center=center).max() - df[z_column].rolling(window, center=center).min() 
       
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



############## MED-DIFF #####################


def calculate_median_window(df, columns, window): #win = changed from 4 to 5 to be close to 100ms (approx 90ms)

    for column in columns:
        results = []
        for row_index in range(len(df)):
            # Get the range before and after the row
            start_before = max(0, row_index - window)
            end_after = min(len(df), row_index + window + 1)

            # Samples before
            samples_before = df[start_before:row_index]
            median_before = np.nanmedian(samples_before[column]) # if len(samples_before) > 0 else np.nan 

            # Samples after
            samples_after = df[row_index + 1:end_after]
            median_after = np.nanmedian(samples_after[column]) # if len(samples_after) > 0 else np.nan 

            # Append the result as a tuple 
            results.append((median_before, median_after))

        # Add the results to the DataFrame
        df[f'{column}_Median_Before_Win{window}'] = [result[0] for result in results]
        df[f'{column}_Median_After_Win{window}'] = [result[1] for result in results]

    return df

def calc_median_dist_m(df, window): # a bit slow to run
    
    if "median_dist_m" not in df.columns:
        df["median_dist_m"] = ""
        
    for gaze in range(1, len(df)):
        
        x, y, z  = df.iloc[gaze][f"L_x_Median_After_Win{window}"], df.iloc[gaze][f"L_y_Median_After_Win{window}"], df.iloc[gaze][f"L_z_Median_After_Win{window}"]
        prev_x, prev_y, prev_z = df.iloc[gaze][f"L_x_Median_Before_Win{window}"], df.iloc[gaze][f"L_y_Median_Before_Win{window}"], df.iloc[gaze][f"L_z_Median_Before_Win{window}"]
    
        sqr_dist_x = (x - prev_x)**2
        sqr_dist_y = (y - prev_y)**2
        sqr_dist_z = (z - prev_z)**2

        sum_square_distances = sqr_dist_x + sqr_dist_y + sqr_dist_z
        median_distance_meters = round(math.sqrt(sum_square_distances), 4) 
       
        df.at[gaze,'median_dist_m'] = median_distance_meters   
        df['median_dist_m'] = pd.to_numeric(df['median_dist_m'], errors='coerce')
      
    return df    

# apply_viewing_distance_df
#  convert_met_to_degree

############################### 6 STD ###########################################


# Function to calculate rolling standard deviations
def calculate_std_meters_win_orig(df, x_column, y_column, z_column, window, center=True):
    
    df['std_x_meters'] = df[x_column].rolling(window, center=center).std()
    df['std_y_meters'] = df[y_column].rolling(window, center=center).std()
    df['std_z_meters'] = df[z_column].rolling(window, center=center).std()
    
        # Calculate the total standard deviation (Euclidean norm of std components)
    df['std_total_meters'] = (
        df['std_x_meters']**2 + df['std_y_meters']**2 + df['std_z_meters']**2
    )**0.5
        
    return df



# apply_viewing_distance_df
#  convert_met_to_degree

############################## 7 STD - DIFF ############################################


def calculate_std_window(df, columns, window):
    for column in columns:
        results = []
        for row_index in range(len(df)):
            # Get the range before and after the row
            start_before = max(0, row_index - window)
            end_after = min(len(df), row_index + window + 1)

            # Samples before
            samples_before = df[start_before:row_index]
            std_before = np.nanstd(samples_before[column]) #if len(samples_before) > 0 else np.nan 

            # Samples after
            samples_after = df[row_index + 1:end_after]
            std_after = np.nanstd(samples_after[column]) #if len(samples_after) > 0 else np.nan 

            # Append the result as a tuple
            results.append((std_before, std_after))

        # Add the results to the DataFrame
        df[f'{column}_Std_Before_Win{window}'] = [result[0] for result in results]
        df[f'{column}_Std_After_Win{window}'] = [result[1] for result in results]

    return df




def calc_std_wind_dist_m(df, window):  #its the calc_dist_m finction vectorized
    
    if "std_wind_dist_m" not in df.columns:
        df["std_wind_dist_m"] = ""
        
    df['std_wind_dist_m'] = round(
        ((df[f"L_x_Std_After_Win{window}"]-df[f"L_x_Std_Before_Win{window}"])**2 +  (df[f"L_y_Std_After_Win{window}"]-df[f"L_y_Std_Before_Win{window}"])**2 +  (df[f"L_z_Std_After_Win{window}"]-df[f"L_z_Std_Before_Win{window}"])**2
    )**0.5,4)
           
    return df

#the function below is actually not necessary because the values are too similar from the column "viewing_distance"
def calc_feature_wind_dist_m(df, new_col_feature_dist_m, x_col_feature_bef_wind, x_col_feature_aft_wind, y_col_feature_bef_wind, y_col_feature_aft_wind, z_col_feature_bef_wind, z_col_feature_aft_wind):  # its a generic function for every column name
    
    if new_col_feature_dist_m not in df.columns:
        df[new_col_feature_dist_m] = ""
        
    df[new_col_feature_dist_m] = (
        (df[x_col_feature_aft_wind]-df[x_col_feature_bef_wind])**2 +  (df[y_col_feature_aft_wind]-df[y_col_feature_bef_wind])**2 +  (df[z_col_feature_aft_wind]-df[z_col_feature_bef_wind])**2
    )**0.5
           
    return df 

# apply_viewing_distance_df
#  convert_met_to_degree

############################## 8 RMS - DIFF ############################################


def calculate_rms_window(df, columns, window): # a bit slow to run
    for column in columns:
        results = []
        for row_index in range(len(df)):
            # Get the range before and after the row
            start_before = max(0, row_index - window)
            end_after = min(len(df), row_index + window + 1)

            # Samples before
            samples_before = df[start_before:row_index]
            rms_before = np.sqrt(np.nanmean(samples_before[column]**2)) #if len(samples_before) > 0 else np.nan 

            # Samples after
            samples_after = df[row_index + 1:end_after]
            rms_after = np.sqrt(np.nanmean(samples_after[column]**2)) #if len(samples_after) > 0 else np.nan 

            # Append the result as a tuple
            results.append((rms_before, rms_after))

        # Add the results to the DataFrame
        df[f'{column}_Rms_Before_Win{window}'] = [result[0] for result in results]
        df[f'{column}_Rms_After_Win{window}'] = [result[1] for result in results]

    return df


############################### 9 RMS ###########################################


def calculate_rms_meters_win(df, x_column, y_column, z_column, window, center=True):

    df['rms_x_meters'] = df[x_column].rolling(window, center=center).apply(
        lambda x: np.sqrt(np.nanmean(x**2)), raw=True
    )
    df['rms_y_meters'] = df[y_column].rolling(window, center=center).apply(
        lambda x: np.sqrt(np.nanmean(x**2)), raw=True
    )
    df['rms_z_meters'] = df[z_column].rolling(window, center=center).apply(
        lambda x: np.sqrt(np.nanmean(x**2)), raw=True
    )
    

    df['rms_total_meters'] = (
        df['rms_x_meters']**2 + df['rms_y_meters']**2 + df['rms_z_meters']**2
    )**0.5

    return df

############################### 10 BCEA #################################################


def calculate_bcea2d_m_win_corrcoef(df, dim1, dim2, window, k=1) :

    std_x = df[dim1].rolling(window, center=True).std()
    std_y = df[dim2].rolling(window, center=True).std()
    
    # Pearson correlation
    def rolling_corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    corr = df[dim1].rolling(window, center=True).apply(
        lambda x: rolling_corr(x, df[dim2][x.index]), raw=False
    )

    # BCEA formula
    bcea = 2 * k * np.pi * std_x * std_y * np.sqrt(1 - corr**2)
    
    bcea = bcea.fillna(0)
        
    # Add BCEA as a column in the DataFrame
    df[f'bcea_{dim1}{dim2}'] = bcea
        
    return df


# Function to calculate rolling standard deviations
def calculate_std_meters_win(df, dim_columns, std_columns, window, center=True):
  
    for dim_col, std_col in zip(dim_columns, std_columns):
        if std_col not in df.columns:
            df[std_col] = df[dim_col].rolling(window, center=center).std()
    return df

# def calculate_std_meters_win(df, dim1_column, dim2_column, dim1_std_col, dim2_std_col, z_column=None, window=5, center=True):
#     df[dim1_std_col] = df[dim1_column].rolling(window, center=center).std()
#     df[dim2_std_col] = df[dim2_column].rolling(window, center=center).std()

#     return df


# # BCEA Calculation = calculate_bcea2d_m_win_corrcoef
# def calculate_bcea2d_m_win_cov(df, dim1, dim2, k=1, window=5):
    
#     dim1_std_col = f"{dim1}_std"
#     dim2_std_col = f"{dim2}_std"
    
    
#     # Calculate rolling covariance
#     def cov(x, y):
#         if len(x) < 2:
#             return np.nan
#         return np.cov(x, y)[0][1]  # Extract covariance Source: https://stackoverflow.com/questions/15317822/calculating-covariance-with-python-and-numpy
    
#     cov_xy = df[dim1].rolling(window, center=True).apply(
#         lambda x: cov(x, df[dim2][x.index]),
#         raw=False
#     )

    
#     # Use rolling std calculation for 2D
#     df = calculate_std_meters_win(
#         df, 
#         dim_columns=[dim1, dim2], 
#         std_columns=[dim1_std_col, dim2_std_col], 
#         window=window
#     )
    
   
#     # Use rolling std calculation
#     # df = calculate_std_meters_win(df, dim1, dim2, dim1_std_col, dim2_std_col, z_column=None, window=window)

#     std_x = df[dim1_std_col]
#     std_y = df[dim2_std_col]

#     # Pearson correlation
#     corr = cov_xy / (std_x * std_y)

#     # BCEA formula
#     bcea = 2 * k * np.pi * std_x * std_y * np.sqrt(1 - corr**2)

#     # Replace NaN with 0
#     bcea = bcea.fillna(0)

#     # Add BCEA as a column in the DataFrame
#     df[f'bcea_{dim1}{dim2}_cov'] = bcea
        
#     return df


def calculate_std_only_m_win(df, x_column, y_column, z_column, window, center=True):
    
    df['std_x_m'] = df[x_column].rolling(window, center=center).std()
    df['std_y_m'] = df[y_column].rolling(window, center=center).std()
    df['std_z_m'] = df[z_column].rolling(window, center=center).std()
    
    return df


def calculate_pearson(df, dim1, dim2, dim3, window): # std_x_col, std_y_col, std_z_col

    # std_x = df[std_x_col]
    # std_y = df[std_y_col]
    # std_z = df[std_z_col]   

    # Pearson correlation
    def rolling_corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    corr = df[dim1].rolling(window, center=True).apply(
        lambda x: rolling_corr(x, df[dim2][x.index]), raw=False
    )
    
    corr = corr.fillna(0)
    
    df[f"pearson_{dim1}_{dim2}"] = corr
    
    
    return df


def calculate_bcea_volume(df, std_x_col, std_y_col, std_z_col, pearson_xy_col, pearson_yz_col, pearson_zx_col, k=1):
  
    volume = (4/3) * 2 * k * math.pi**2 * df[std_x_col] * df[std_y_col] * df[std_z_col] * (
        2 * k * math.pi * 
        (1 - df[pearson_xy_col]**2).apply(math.sqrt) * 
        (1 - df[pearson_yz_col]**2).apply(math.sqrt) * 
        (1 - df[pearson_zx_col]**2).apply(math.sqrt)
    ).apply(math.sqrt)
    
    volume = volume.fillna(0)
    df["bcea_3d"] = volume
    
    return df

# 3 to include a small epsilon (e.g. (std_x+0.01),(std_y+0.01) and (std_z+0.1)) 
def calculate_bcea_volume_noise(df, std_x_col, std_y_col, std_z_col, pearson_xy_col, pearson_yz_col, pearson_zx_col, k=1):
  
    volume = (4/3) * 2 * k * math.pi**2 * (df[std_x_col]+0.001) * (df[std_y_col]+0.001)* (df[std_z_col] + 0.001)* (
        2 * k * math.pi * 
        (1 - df[pearson_xy_col]**2).apply(math.sqrt) * 
        (1 - df[pearson_yz_col]**2).apply(math.sqrt) * 
        (1 - df[pearson_zx_col]**2).apply(math.sqrt)
    ).apply(math.sqrt)
    
    volume = volume.fillna(0)
    df["bcea_3d_noise"] = volume
    
    return df

######################################## 11 BCEA DIFF #######################################

def calculate_bcea2d_window(df, dim1, dim2, window, k=1):
    results_before = []
    results_after = []
    
    for row_index in range(len(df)):
 
        start_before = max(0, row_index - window)
        end_after = min(len(df), row_index + window + 1)
        

        samples_before = df[start_before:row_index]
        std_x_before = np.nanstd(samples_before[dim1]) 
        std_y_before = np.nanstd(samples_before[dim2]) 
        corr_before = np.corrcoef(samples_before[dim1], samples_before[dim2])[0, 1] 
        corr_before = np.nan_to_num(corr_before, nan=0.0)
        bcea_before = 2 * k * np.pi * std_x_before * std_y_before * np.sqrt(1 - corr_before**2)

        
        samples_after = df[row_index + 1:end_after]
        std_x_after = np.nanstd(samples_after[dim1]) 
        std_y_after = np.nanstd(samples_after[dim2]) 
        corr_after = np.corrcoef(samples_after[dim1], samples_after[dim2])[0, 1] 
        corr_after = np.nan_to_num(corr_after, nan=0.0)
        bcea_after = 2 * k * np.pi * std_x_after * std_y_after * np.sqrt(1 - corr_after**2)


        # Store the results
        results_before.append(bcea_before)
        results_after.append(bcea_after)
        

    # Add results as new columns to the DataFrame
    df[f'bcea_{dim1}{dim2}_Before_Win{window}'] = np.nan_to_num(results_before, nan=0.0)
    df[f'bcea_{dim1}{dim2}_After_Win{window}'] = np.nan_to_num(results_after, nan=0.0)
    
    return df

#only one plane - yz:
def calc_feature_wind_dist_m_2dim(df, new_col_feature_dist_m, y_col_feature_bef_wind, y_col_feature_aft_wind):  # its a generic function for every column name
    
    if new_col_feature_dist_m not in df.columns:
        df[new_col_feature_dist_m] = ""
        
    df[new_col_feature_dist_m] = (
         (df[y_col_feature_aft_wind]-df[y_col_feature_bef_wind])**2 )**0.5
           
    return df 


def calculate_pearson2d(df, dim1, dim2, window): # std_x_col, std_y_col, std_z_col

    # Pearson correlation
    def rolling_corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    corr = df[dim1].rolling(window, center=True).apply(
        lambda x: rolling_corr(x, df[dim2][x.index]), raw=False
    )
    
    corr = corr.fillna(0)
    
    df[f"pearson_{dim1}_{dim2}"] = corr
    
    
    return df


def calculate_bcea3d_window_noise(df, dim1, dim2, dim3, window, k=1): #std_x_col, std_y_col, std_z_col, pearson_xy_col, pearson_yz_col, pearson_zx_col,
    results_before = []
    results_after = []
    
    for row_index in range(len(df)):
 
        start_before = max(0, row_index - window)
        end_after = min(len(df), row_index + window + 1)
        

        samples_before = df[start_before:row_index]
        
        std_x_before = np.nanstd(samples_before[dim1]) 
        std_y_before = np.nanstd(samples_before[dim2])  
        std_z_before = np.nanstd(samples_before[dim3])
        
        pearson_xy_col_bef = calculate_pearson2d(samples_before[dim1], f"{dim1}", f"{dim2}", window)
        pearson_yz_col_bef = calculate_pearson2d(samples_before[dim1], f"{dim2}", f"{dim3}", window)
        pearson_zx_col_bef = calculate_pearson2d(samples_before[dim1], f"{dim3}", f"{dim1}", window) 


        bcea_before = calculate_bcea_volume_noise(df, std_x_before, std_y_before, std_z_before, pearson_xy_col_bef, pearson_yz_col_bef, pearson_zx_col_bef, k=1)

  
        samples_after = df[row_index + 1:end_after]
        std_x_after = np.nanstd(samples_after[dim1]) 
        std_y_after = np.nanstd(samples_after[dim2])
        std_z_after = np.nanstd(samples_after[dim3])
        
        pearson_xy_col_aft = calculate_pearson2d(samples_after[dim1], f"{dim1}", f"{dim2}", window) 
        pearson_yz_col_aft = calculate_pearson2d(samples_after[dim1], f"{dim2}", f"{dim3}", window)
        pearson_zx_col_aft = calculate_pearson2d(samples_after[dim1], f"{dim3}", f"{dim1}", window)

        bcea_after = calculate_bcea_volume_noise(df, std_x_col_after, std_y_col_after, std_z_col_after, pearson_xy_col_aft, pearson_yz_col_aft, pearson_zx_col_aft, k=1)


        # Store the results
        results_before.append(bcea_before)
        results_after.append(bcea_after)
        

    # Add results as new columns to the DataFrame
    df[f'bcea3dnoise_{dim1}{dim2}_Before_Win{window}'] = np.nan_to_num(results_before, nan=0.0)
    df[f'bcea3dnoise_{dim1}{dim2}_After_Win{window}'] = np.nan_to_num(results_after, nan=0.0)
    
    return df

######################################## 11 FREQUENCY #######################################


# source: https://stackoverflow.com/questions/58427391/how-to-add-24-rows-under-each-row-in-python-pandas-dataframe 

# def interpolate(df, N, col_names):
    
#     """ 
#      adds N - number of empty rows created between two following rows in a df and interpolate the values
     
#     """  
  
#     df.copy().index = df.index * (N + 1) 
#     dftest = df.reindex(np.arange(df.index.max() + N + 1))
    
#     dftest = dftest.loc[col_names].interpolate()
    
#     return dftest

def interpolate_and_GTs_ff(df, cols_name, N ):  
    
    df_copy = df.copy()
    df_copy.index = df_copy.index * (N + 1) 
    df_new_cols = df_copy.reindex(np.arange(df_copy.index.max() + N + 1))
    
    df_interp = df_new_cols[cols_name].interpolate()
    df_new_cols.update(df_interp)
    
    gt_columns = ["GT1", "GT2","GT3", "GT4","GT5", "GT6", "GT7", "observer"]
    df_GTs_filled = df_new_cols[gt_columns].ffill()
    df_new_cols.update(df_GTs_filled)
       
    return df_new_cols


def interpolate_and_GTs_ff_reset_index(df, cols_name, N ):  
    
    df_copy = df.copy()
    df_copy.reset_index(drop=True, inplace=True)
    df_copy.index = df_copy.index * (N + 1) 
    df_new_cols = df_copy.reindex(np.arange(df_copy.index.max() + N + 1))
    
    df_interp = df_new_cols[cols_name].interpolate()
    df_new_cols.update(df_interp)
    
    gt_columns = ["GT1", "GT2","GT3", "GT4","GT5", "GT6", "GT7", "observer"]
    df_GTs_filled = df_new_cols[gt_columns].ffill()
    df_new_cols.update(df_GTs_filled)
       
    return df_new_cols



# def select_observer_freq():
#     observers_per_freq ={}
#     Hz = ['freq_N0_44Hz', 'freq_N1_87Hz', 'freq_N2_130Hz', 'freq_N3_174Hz', 'freq_N4_217Hz']
    
#     for index, freq in enumerate(Hz):
    
#         lists=[]
        
#         for i in range(index+1,54,5):
#             list = lists.append(i)
#         print(lists)
    
#         observers_per_freq[freq] = lists
    
#     return observers_per_freq


def select_observer_freq():
    
    observers_per_freq = {}  
    Hz = ['freq_N0_44Hz', 'freq_N1_87Hz', 'freq_N2_130Hz', 'freq_N3_174Hz', 'freq_N4_217Hz']
    observers = list(range(1, 54))

    # Distribute observers in a round-robin way
    for i, freq in enumerate(Hz):
        # observers_per_freq[freq] = [observers[j] for j in range(i, len(observers), len(Hz))]
        observers_per_freq[freq] = observers[i::len(Hz)]  # Take every 5th observer starting from i

    return observers_per_freq

# Run the function
obs_freq = select_observer_freq()




# def process_frequency_data(et_data, obs_freq, funcs_feat):
#     """
#     Processes eye-tracking data for different frequency groups by:
#     1. Selecting rows for each frequency group based on observer IDs.
#     2. Optionally interpolating and resetting index (if frequency > 44Hz).
#     3. Adding a count column.
#     4. Saving the processed DataFrame to a CSV file.
    
#     Parameters:
#         et_data (pd.DataFrame): The main dataset containing all observations.
#         obs_freq (dict): Dictionary mapping frequency labels to observer lists.
#         funcs_feat (module): Feature functions module (for saving and interpolation).
    
#     Returns:
#         dict: Dictionary of processed DataFrames for each frequency.
#     """

#     Hz_groups = {
#         'freq_N0_44Hz': {"reset_index": False, "interpolation_step": None},
#         'freq_N1_87Hz': {"reset_index": True, "interpolation_step": 1},
#         'freq_N2_130Hz': {"reset_index": True, "interpolation_step": 2},
#         'freq_N3_174Hz': {"reset_index": True, "interpolation_step": 3},
#         'freq_N4_217Hz': {"reset_index": True, "interpolation_step": 4},
#     }

#     processed_data = {}

#     for freq, settings in Hz_groups.items():
#         # Select rows where observer is in the given frequency group
#         rows_obs = et_data[et_data["observer"].isin(obs_freq[freq])]

#         # Apply interpolation if needed
#         if settings["reset_index"]:
#             processed_df = funcs_feat.interpolate_and_GTs_ff_reset_index(
#                 rows_obs, ["time", "L_x", "L_y", "L_z", "C_x", "C_y", "C_z"], settings["interpolation_step"]
#             )
#         else:
#             processed_df = rows_obs.copy()

#         # Add a count column
#         processed_df["count_freq"] = range(1, len(processed_df) + 1)

#         # Save the processed DataFrame
#         filename = f"data/Aa01_et_data_{freq}_processed.csv"
#         funcs_feat.save_df(processed_df, filename)

#         # Store the processed DataFrame for further use
#         processed_data[freq] = processed_df

#     return processed_data  # Returns a dictionary of all processed DataFrames
