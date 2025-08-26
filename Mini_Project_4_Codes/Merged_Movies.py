#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:42:14 2024

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


# Merge the two sets by common movies:
merge_df = pd.merge(rt, md, on=['Title'])   




# Save a file:
merge_df.to_csv('Merged_Movie_Data.csv', index=True)


