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
df_TX = Grouped.get_group('Texas')
df_FL = Grouped.get_group('Florida')
df_NY = Grouped.get_group('New York')
df_PA = Grouped.get_group('Pennsylvania')

# Convert State Data to array
Date = df_CA.iloc[:, 3].to_numpy() - 43934
CA_Cases = df_CA.iloc[:, 4].to_numpy() # 5th column is the "Cases"
TX_Cases = df_TX.iloc[:, 4].to_numpy()
FL_Cases = df_FL.iloc[:, 4].to_numpy()
NY_Cases = df_NY.iloc[:, 4].to_numpy()
PA_Cases = df_PA.iloc[:, 4].to_numpy()

# Isolating California data to the Delta variant timeline
Delta_Date = df_CA.iloc[550:750, 3].to_numpy() - 43934
CA_Delta_Cases = df_CA.iloc[550:750, 4].to_numpy()
x = np.array(Delta_Date)

def f(x, L, k, x0):
    return 1 / (1 + L*np.exp(-k * (x - x0)))

# Perform curve fitting:
c_fit = sy.optimize.curve_fit(f, x/np.max(x), CA_Delta_Cases/np.max(CA_Delta_Cases)) # Input the function and data (x , y)
c_fit = c_fit[0] # Call only the first element of c_fit

print(f"Coefficients: {c_fit}")

x_fit = np.linspace(min(x), max(x), 101)/np.max(x)
y_fit = f(x_fit, c_fit[0], c_fit[1], c_fit[2])

# Overall Figure Formatting
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
# fig.suptitle('Cumulative COVID Cases by State', fontsize=20)
# fig.supylabel('Cumulative Cases (millions)', fontsize=15)
# fig.supxlabel('Days since 4/12/2020', fontsize=15)

# Plot the data for subplot 1
# ax1.plot(Date, PA_Cases/1e6, 'b-', linewidth=2, label = "Pennsylvania") # 
# ax1.plot(Date, TX_Cases/1e6, 'r-', linewidth=2, label = "Texas") # 
# ax1.plot(Date, FL_Cases/1e6, 'g-', linewidth=2, label = "Florida") # 
# ax1.plot(Date, NY_Cases/1e6, 'k-', linewidth=2, label = "New York") # 

# Plot 1 area formatting
# ax1.tick_params(labelsize=10) # Tick mark labels
# ax1.legend() # Adds a legend
# ax1.grid(True, which='both') # Adds grid lines for both x & y axes
# ax1.ticklabel_format(useOffset=False)
# ax1.minorticks_on()

# Plot the data for subplot 2
plt.plot(x/np.max(x), CA_Delta_Cases/np.max(CA_Delta_Cases), 'b-', linewidth=4, label = "California") # 
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label = "Curve Fit")

# Plot 2 area formatting
# ax2.tick_params(labelsize=10) # Tick mark labels
# ax2.legend() # Adds a legend
# ax2.grid(True, which='both') # Adds grid lines for both x & y axes
# ax2.ticklabel_format(style='plain')
# ax2.minorticks_on()

plt.show()
