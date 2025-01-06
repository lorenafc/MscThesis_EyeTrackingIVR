# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes


The velocity and acceleration scrip was made based on GazeParser library, by Hiroyuki Sogo. 

website: https://gazeparser.sourceforge.net/#

Sogo, H. (2013) GazeParser: an open-source and multiplatform library for low-cost eye tracking and analysis. 
Behavior Reserch Methods 45, pp 684-695, doi:10.3758/s13428-012-0286-x

"""


import os
import pandas as pd
import json
import Aa00_funcs_preprocessing_C_test_config_git_ignore as funcs


with open('config.json') as json_file:
    config = json.load(json_file)


script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
eye_tracking_data = pd.read_csv(path_file)


eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)



# if __name__ == "__main__":
    
#    eye_tracking_data_cm2deg_new
    

