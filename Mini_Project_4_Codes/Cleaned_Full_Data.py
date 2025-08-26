#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:39:59 2024

@author: brooksemerick
"""

import pandas as pd
import re

# This file takes in the full Criminal Data Set and cleans it with respect to 
# the victim and offender age column.  It also deletes unwanted columns.

# Load the dataset
Full_df = pd.read_csv('Criminal_Data_Set_1976-to-2020.csv')

#%% Cleaning of Victim Age Column:

# Drop blank columns pertaining to crimes involving more than 1 person: 
vdrop = [col for col in Full_df.columns if re.match(r'victim_([2-9]|\d{2,})', col)]
odrop = [col for col in Full_df.columns if re.match(r'offender_([2-9]|\d{2,})', col)]
rdrop = [col for col in Full_df.columns if re.match(r'victim_1_relation_to_offender_([2-9]|\d{2,})', col)]

# Full_df = Full_df.drop(columns = ['victim_2_age'])
Full_df = Full_df.drop(columns = vdrop + odrop + rdrop)

# Change relevant strings to numerical data for victim data: 
string_to_number = {
    'birth to 6 days, including abandoned infant': 0, 
    '7 days to 364 days': 0, 
    '99 years or older': 99
}

Full_df['victim_1_age'] = Full_df['victim_1_age'].replace(string_to_number)

# Change relevant strings to numerical data for offender age: 
string_to_number = {
    '99 years or older': 99
}

Full_df['offender_1_age'] = Full_df['offender_1_age'].replace(string_to_number)

# Remove all unknown and other string entries in victim and offender age columns:
Full_df = Full_df[Full_df['victim_1_age'] != 'unknown']
Full_df.dropna(subset=['victim_1_age'], inplace=True)  
Full_df = Full_df[Full_df['offender_1_age'] != 'unknown']
Full_df = Full_df[Full_df['offender_1_age'] != 'nb']
Full_df = Full_df[Full_df['offender_1_age'] != 'bb']
Full_df = Full_df[Full_df['offender_1_age'] != '2m']
Full_df.dropna(subset=['offender_1_age'], inplace=True) 

# Convert all entries in victim and offender age columns to floats:
Full_df['victim_1_age'] = pd.to_numeric(Full_df['victim_1_age'])
Full_df['offender_1_age'] = pd.to_numeric(Full_df['offender_1_age'])

# Save Cleaned File:
Full_df.to_csv('Clean_Full_data.csv', index=False)
