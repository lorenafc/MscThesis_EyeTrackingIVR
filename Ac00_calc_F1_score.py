# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:46:58 2025

@author: carpe
"""


import pandas as pd
import numpy as np
import os
import json

# import Aa01_funcs_extracting_features_C_gitignore as funcs


with open('config.json') as json_file:
    config = json.load(json_file)

with open('params.json') as params_file:
    params = json.load(params_file)

script_dir = config["script_dir"]  
os.chdir(script_dir)  

path_file = os.path.join(script_dir, config["data_path"])
extracted_features = pd.read_csv(config["extracted_features_file"])


# source: https://mariofilho.com/precisao-recall-e-f1-score-em-machine-learning/

from sklearn.metrics import precision_score

# Suponha que você tenha os seguintes rótulos verdadeiros e previstos pelo modelo:
y_true = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
y_pred = [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]

# Para calcular a precisão, basta chamar a função "precision_score" passando os rótulos verdadeiros e previstos como argumentos:
precision = precision_score(y_true, y_pred)

# A precisão será um valor entre 0 e 1:
print(precision) 


from sklearn.metrics import recall_score


# Para calcular o recall, basta chamar a função "recall_score" passando os rótulos verdadeiros e previstos como argumentos:
recall = recall_score(y_true, y_pred)

# O recall será um valor entre 0 e 1:
print(recall) 


from sklearn.metrics import f1_score


# Calcular o F1 score para as previsões do modelo
f1 = f1_score(y_true, y_pred)

# Exibir
print(f1)