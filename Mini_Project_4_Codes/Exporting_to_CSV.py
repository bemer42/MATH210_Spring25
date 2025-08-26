#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 23:24:21 2024

@author: brooksemerick
"""

import pandas as pd

# This file takes Clean Full Data set and will output a set of csv files split
# by a desired category:

# Load the dataset
Clean_df = pd.read_csv('Clean_Full_data.csv')

#%% 
# Save a file for a specified field/column: 
field = 'year'    
    
# Delete any nan rows: 
Clean_df.dropna(subset=[field], inplace=True)  

cats = Clean_df[field].unique()
cats_data = {category: Clean_df[Clean_df[field] == category] for category in cats}

for category, df in cats_data.items():
    df.to_csv(f'{category}_Clean_data.csv', index=False)

