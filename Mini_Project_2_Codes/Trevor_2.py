#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:49:12 2024

@author: brooksemerick
"""

import pandas as pd
import numpy as np
import numpy.linalg as LA # Lets us reference Linear Algebra stuff easier
import matplotlib.pyplot as plt
import scipy as sy

# Import the data (dataframe):
df = pd.read_excel('Covid Data.xlsx', 'Cleaned Data Set') # Excel file path and sheet/tab name

# Grabbing all State rows:
Cat = 'Province_State'
Grouped = df.groupby(Cat)
df_CA = Grouped.get_group('California')

# Isolating California data to the Delta variant timeline
Delta_Date = df_CA.iloc[550:750, 3].to_numpy() - 43934
CA_Delta_Cases = df_CA.iloc[550:750, 4].to_numpy()
x = np.array(Delta_Date)

# Normalized Data:
# x = x/np.max(x)
# y = CA_Delta_Cases/np.max(CA_Delta_Cases)
y = CA_Delta_Cases

# Define Arctan Function:
def f_atan(x, c0, c1, c2, c3):
    return c0 + c1*np.pi*np.arctan(c2*(x-c3))

# Perform curve fitting:
c_atan = sy.optimize.curve_fit(f_atan, x, y) # Input the function and data (x , y)
c_atan = c_atan[0] # Call only the first element of c_fit


# Plotting variable:
x_plot = np.linspace(min(x), max(x), 1001)

# Plot Data:
plt.figure(1)
plt.plot(x, y, 'bo', linewidth=6, label = "California") # 
plt.plot(x_plot, f_atan(x_plot, c_atan[0], c_atan[1], c_atan[2], c_atan[3]), \
         'r-',linewidth = 5, label = 'Arctan Fit')

plt.show()
