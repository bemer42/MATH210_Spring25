#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:12:27 2024

@author: brooksemerick
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
from tabulate import tabulate

# Load the datasets
rt = pd.read_csv('Clean_RT_data.csv')
md = pd.read_csv('Clean_iMDb_data.csv')

rt_sc = rt[['RT Critics Score','RT Audience Score']].values
md_sc = md[['iMDb Critics Score','iMDb Metascore']].values


#%%

data = rt[['RT Critics Score','RT Audience Score']]

data.plot(kind='box')



#%% Basic Descriptive Statistics:

# Five Number Summary: 
    
    
field = 'Certificate'    
    
counts = md[field].value_counts()

plt.figure(1)
counts.plot.pie(autopct='%1.1f%%', startangle=140)
plt.ylabel('')  # Optional: hide the y-label
plt.title('Genre Distribution')
plt.show()

plt.figure(2)
counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Genre Distribution', fontsize=20)
plt.xlabel('Genre', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.grid(True, which='both')
plt.suptitle('')
plt.show()


#%% Bivariate Histogram on Rotten Tomatoes: 
 
# plt.figure(3)
# plt.hist2d(x=rt_sc[:,0], y=rt_sc[:,1], bins=30, cmap = 'plasma')
# plt.title('Rotten Tomatoes Scores', fontsize=20)
# plt.xlabel('Critic Scores', fontsize=16)
# plt.ylabel('Audience Scores', fontsize=16)
# plt.grid(True, which='both')
# plt.suptitle('')
# plt.colorbar()
# plt.show()
    

# plt.figure(4)
# sb.kdeplot(x=rt_sc[:,0], y=rt_sc[:,1], bins=30, cmap = 'plasma', shade=True, thresh=0.05)
# plt.title('Rotten Tomatoes Scores', fontsize=20)
# plt.xlabel('Critic Scores', fontsize=16)
# plt.ylabel('Audience Scores', fontsize=16)
# plt.grid(True, which='both')
# plt.suptitle('')
# plt.colorbar()
# plt.show()
    

# plt.figure(5)
# plt.plot(rt_sc[:,0], rt_sc[:,1], 'ko', linewidth=6)
# plt.title('Rotten Tomatoes Scores', fontsize=20)
# plt.xlabel('Critic Scores', fontsize=16)
# plt.ylabel('Audience Scores', fontsize=16)
# plt.grid(True, which='both')
# plt.suptitle('')
# plt.show()
    
#%% Bivariate Plot of iMDb Scores: 
    
plt.figure(6)
sb.kdeplot(x=md_sc[:,1], y=md_sc[:,0], bins=30, cmap = 'plasma', shade=True, thresh=0.05)
plt.title('iMDb Scores', fontsize=20)
plt.xlabel('Critic Scores', fontsize=16)
plt.ylabel('Audience Scores', fontsize=16)
plt.grid(True, which='both')
plt.suptitle('')
plt.colorbar()
plt.show()
