#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:07:36 2024

@author: brooksemerick
"""

import pandas as pd
import re

# Load the dataset
df = pd.read_csv('RottenTomatoes.csv')

# Remove the justification in the Certificate column: 
df['Certificate'] = df['Certificate'].str.replace(r'\(.*\)','',regex=True)



# Clean up the Certificate Column: 
rats = df['Certificate'].unique()
print(rats)
    
str_2_str = {
    ' PG ': 'PG',
    ' PG': 'PG',
    ' NR ': 'NR',
    ' NR': 'NR',
    ' R ': 'R',
    ' R': 'R',
    ' PG-13 ': 'PG-13',
    ' PG-13': 'PG-13',
    ' G': 'G',
    ' NC17': 'NC-17'
    }

df['Certificate'] = df['Certificate'].replace(str_2_str)

rats = df['Certificate'].unique()
print(rats)

# Remove "minutes" from the Runtime column: 
df['Runtime'] = df['Runtime'].str.replace(r'minutes','',regex=True)
df['Runtime'] = df['Runtime'].astype(float)

# Remove excess directors from the Director column: 
df['Director'] = df['Director'].str.split(',').str[0]

# Convert all Studio entries to string type: 
df['Studio'] = df['Studio'].astype(str)

# Delete the percent sign on the scores, then convert them to float: 
df['RT Critics Score'] = df['RT Critics Score'].str.replace(r'%','',regex=True)
df['RT Audience Score'] = df['RT Audience Score'].str.replace(r'%','',regex=True)
df['RT Critics Score'] = df['RT Critics Score'].astype(float)
df['RT Audience Score'] = df['RT Audience Score'].astype(float)

# Delete the commas in Audience reviews, then convert them to float: 
df['RT Audience Reviews'] = df['RT Audience Reviews'].str.replace(r',','',regex=True)
df['RT Audience Reviews'] = df['RT Audience Reviews'].astype(float)

# Convert Critic Reviews to float: 
df['RT Critic Reviews'] = df['RT Critic Reviews'].astype(float)


# Save Cleaned File:
print(df.dtypes)

df.to_csv('Clean_RT_data.csv', index=False)