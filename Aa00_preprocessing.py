# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""


import os
import pandas as pd
import json
import Aa00_funcs_preprocessing as funcs


with open('config.json') as json_file:
    config = json.load(json_file)


script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
eye_tracking_data = pd.read_csv(path_file)


eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)



# if __name__ == "__main__":
    
#    eye_tracking_data_cm2deg_new
    

