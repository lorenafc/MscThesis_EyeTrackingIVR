# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 2024

@author: Lorena Carpes

"""

import os
import pandas as pd
import Aa00_funcs_preprocessing as funcs

path_file = "\path\to\your\file.csv"
eye_tracking_data = pd.read_csv(path_file)

eye_tracking_data_cm2deg_new = funcs.process_eye_tracking_data(eye_tracking_data)


