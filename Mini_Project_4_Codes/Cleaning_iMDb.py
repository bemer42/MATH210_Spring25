#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:35:33 2024

@author: brooksemerick
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('iMDb.csv')

# Remove the justification in the Certificate column: 
df = df.drop(['Poster','Cast','Description','Review Title','Review'], axis=1)

# Remove excess genres from the Genre column: 
df['Genre'] = df['Genre'].str.split(',').str[0]

# Change the rating system: 
rats = df['Certificate'].unique()
print(rats)

str_2_str = {
    'U': 'G',
    'Approved': 'G',
    'All': 'G',
    '7': 'G',
    '12': 'G',
    '12+': 'G',
    'GP': 'PG',
    'UA': 'PG',
    'U/A': 'PG',
    'M/PG': 'PG',
    'UA 7+': 'PG-13',
    '13': 'PG-13',
    '13+': 'PG-13',
    'UA 13+': 'PG-13',
    '15': 'PG-13',
    '15+': 'PG-13',
    'UA 16+': 'R',
    'A': 'R',
    '16': 'R',
    '16+': 'R',
    'U/A 16+': 'R',
    '18': 'R',
    '18+': 'R',
    'S': 'NC-17',
    'X': 'NC-17',
    'Not Rated': 'NR',
    'Unrated': 'NR',
    '(Banned)': 'NR'
    }
    
df['Certificate'] = df['Certificate'].replace(str_2_str)

rats = df['Certificate'].unique()
print(rats)

# Remove commas in number of reviews and convert to float: 
df['iMDb Audience Reviews'] = df['iMDb Audience Reviews'].str.replace(r',','',regex=True)
df['iMDb Critic Reviews'] = df['iMDb Critic Reviews'].str.replace(r',','',regex=True)
df['iMDb Audience Reviews'] = df['iMDb Audience Reviews'].astype(float)
df['iMDb Critic Reviews'] = df['iMDb Critic Reviews'].astype(float)

# Change Critic reviews to a float: 
df['iMDb Critics Score'] = 10*df['iMDb Critics Score'].astype(float)

# Change Critic reviews to a float: 
df['iMDb Metascore'] = df['iMDb Metascore'].astype(float)

# Save Cleaned File:
print(df.dtypes)

df.to_csv('Clean_iMDb_data.csv', index=False)