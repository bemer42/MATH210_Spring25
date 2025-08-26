# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:17:52 2024

@author: Trevor Wylezik
"""

import pandas as pd
import numpy as np
import numpy.linalg as LA # Lets us reference Linear Algebra stuff easier
import matplotlib.pyplot as plt
import scipy as sy

# ------------------------ DATA PULLING ------------------------ #

# Import the data (dataframe):
df = pd.read_excel('E:\Covid Data.xlsx', 'Cleaned Data Set') # Excel file path and sheet/tab name

# Grabbing all State rows:
Cat = 'Province_State'
Grouped = df.groupby(Cat) # Grouping by the names of the states
df_CA = Grouped.get_group('California')
df_TX = Grouped.get_group('Texas')
df_FL = Grouped.get_group('Florida')
df_NY = Grouped.get_group('New York')
df_PA = Grouped.get_group('Pennsylvania')


# ------------------------ GENERIC STATE ARRAYS FOR SUBPLOT 1 ------------------------ #

# Convert State Data to array
Date = df_CA.iloc[:, 3].to_numpy() - 43934 # Dates to be "days from..."
CA_Cases = df_CA.iloc[:, 4].to_numpy() / 1e6 # 5th column is the "Cases"
TX_Cases = df_TX.iloc[:, 4].to_numpy() / 1e6 # 1e6 is 1 million
FL_Cases = df_FL.iloc[:, 4].to_numpy() / 1e6 # Allows the data to be in "millions"
NY_Cases = df_NY.iloc[:, 4].to_numpy() / 1e6 # without actually have millions of cases
PA_Cases = df_PA.iloc[:, 4].to_numpy() / 1e6


# ------------------------ CALIFORNIA DELTA VARIANT LOGISTIC FITTING ------------------------ #

# Isolating California data to the Delta variant timeline
Delta_Date = df_CA.iloc[600:700, 3].to_numpy() - 43934 # Mid October 2021 to Late April 2022
CA_Delta_Cases = df_CA.iloc[600:700, 4].to_numpy() / 1e6 # Pulling only the above date's cases (and scaling them back)
x = np.array(Delta_Date) # Allows for easier referencing later

# Normalize the data to play nice with the logistic function (I had issued before)
CA_Delta_Cases_norm = (CA_Delta_Cases - np.min(CA_Delta_Cases)) / (np.max(CA_Delta_Cases) - np.min(CA_Delta_Cases))

# Define the data vectors x and y:
x = np.array(Delta_Date-650)
y = CA_Delta_Cases_norm

# Remove problematic data points (0 and 1)
y[y<=0] = 0.01
y[y>=1] = 0.99

# Build "Logistic Vandermonde":
A = np.vstack((np.ones(len(x)), -x)).T

# Build Normal equation with the y data on log scale: 
A_norm = A.T@A
y_norm = A.T@np.log(1/y-1)

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = LA.solve(A_norm, y_norm)

# Define a and b: 
a = np.exp(c[0])
b = c[1]

# Creating arrays for plotting
x_fit = np.linspace(min(x), max(x), 101)
y_fit = (1/(1+a*np.exp(-b*x_fit))) * (np.max(CA_Delta_Cases) - np.min(CA_Delta_Cases)) + np.min(CA_Delta_Cases)


# Calculate the residual sum of squares (SS_res) on log data:
ss_res = np.sum((np.log(1/y-1) - (c[0] - c[1]*x)) ** 2)

# Calculate the total sum of squares (SS_tot) on log data:
ss_tot = np.sum((np.log(1/y-1) - np.mean(np.log(1/y-1))) ** 2)

# Compute the R^2 value for each type of fit:
R2_1 = 1 - (ss_res / ss_tot)

# Print out the parameters a and b and R^2
print("Delta Variant Logistic Coefficients:")
print(f"a = {a}")
print(f"b = {b}\n")
print(f"Delta Variant R^2 = {R2_1}\n\n")


# ------------------------ CALIFORNIA MULTI-LOGISTIC FITTING ------------------------ #

# Isolating California data to the Delta variant timeline
z = np.array(Date) # Allows for easier referencing later

# Normalize the data to play nice with the logistic function (I had issued before)
CA_Cases_norm = (CA_Cases - np.min(CA_Cases)) / (np.max(CA_Cases) - np.min(CA_Cases))

# Set up multi-logistic function
def g(z, L1, L2, L3, k1, k2, k3, z1, z2, z3):
    return (L1/(1+np.exp(-k1*(z-z1)))) + (L2/(1+np.exp(-k2*(z-z1-z2)))) + (L3/(1+np.exp(-k3*(z-z1-z2-z3))))


# Perform curve fitting:
p1 = [1/3, 1/2, 1/6, 0.02, 0.05, 0.04, 275, 500, 200] # Initial Guesses (L=1, k=small, x0=midpoint)
c_fit2 = sy.optimize.curve_fit(g, z, CA_Cases_norm, p0=p1) # Input the function and data (x , y)
c_fit2 = c_fit2[0] # Call only the first element of c_fit

# Print out the parameters (L1, L2, L3, k1, k2, k3, x1, x2, x3) and R^2
print("Multi Logistic Coefficients:")
print(f"L1 = {c_fit2[0]} , L2 = {c_fit2[1]} , L3 = {c_fit2[2]}")
print(f"k1 = {c_fit2[3]} , k2 = {c_fit2[4]} , k3 = {c_fit2[5]}")
print(f"x1 = {c_fit2[6]} , x2 = {c_fit2[7]} , x3 = {c_fit2[8]}\n")
#print(f"Multi Logistic R^2 = {R2_2}")

# Creating arrays for plotting
z_fit = np.linspace(min(z), max(z), 1061)
y_fit_norm2 = g(z_fit, c_fit2[0], c_fit2[1], c_fit2[2], c_fit2[3], c_fit2[4], c_fit2[5], c_fit2[6], c_fit2[7], c_fit2[8])
y_fit2 = y_fit_norm2 * (np.max(CA_Cases) - np.min(CA_Cases)) + np.min(CA_Cases)

# Calculate R^2:
ss_res = np.sum((CA_Cases-y_fit2)**2)
ss_tot = np.sum((CA_Cases-np.mean(CA_Cases))**2)
R2_2 = 1 - ss_res/ss_tot
print(f"Coefficient of Determination: R^2 = {R2_2}\n\n")


# ------------------------ PLOTTING AND FORMATTING ------------------------ #

# Overall Figure Formatting
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False) # Creates a figure with 3 subplots (ax1,ax2,ax3)
fig.suptitle('Cumulative COVID Cases by State', fontsize=20)
fig.supylabel('Cumulative Cases (millions)', fontsize=15)
fig.supxlabel('Days since 4/12/2020', fontsize=15)
fig.tight_layout() # Leaves some room between the elements

# Plot the data for subplot 1 (4 state's data)
ax1.plot(Date, PA_Cases, 'b-', linewidth=2, label = "Pennsylvania") # PA Data Plot
ax1.plot(Date, TX_Cases, 'r-', linewidth=2, label = "Texas") # TX Data Plot
ax1.plot(Date, FL_Cases, 'g-', linewidth=2, label = "Florida") # FL Data Plot
ax1.plot(Date, NY_Cases, 'k-', linewidth=2, label = "New York") # NY Data Plot

# Plot 1 area formatting
ax1.set_title('4 State COVID Cases') # Sets the title
ax1.tick_params(labelsize=10) # Tick mark labels
ax1.legend() # Adds a legend
ax1.grid(True, which='both') # Adds grid lines for both x & y axes
ax1.ticklabel_format(useOffset=False)
ax1.minorticks_on()

# Plot the data for subplot 2
ax2.plot(Date, CA_Cases, 'bo', markersize=1.5, label = "California") # Original Data
ax2.plot(z_fit, y_fit2, 'r-', linewidth=2.5, label = "Curve Fit") # Curve Fit

# Plot 2 area formatting
ax2.set_title('California Multi-Logistic Fit')
ax2.tick_params(labelsize=10) # Tick mark labels
ax2.legend() # Adds a legend
ax2.grid(True, which='both') # Adds grid lines for both x & y axes
ax2.ticklabel_format(style='plain')
ax2.minorticks_on()

# Plot the data for subplot 3
ax3.plot(Delta_Date, CA_Delta_Cases, 'bo', markersize=1.5, label = "California") # Original Data
ax3.plot(x_fit+650, y_fit, 'r-', linewidth=2.5, label = "Curve Fit") # Curve fit

# Plot 3 area formatting
ax3.set_title('California Delta Variant Focus')
ax3.tick_params(labelsize=10) # Tick mark labels
ax3.legend() # Adds a legend
ax3.grid(True, which='both') # Adds grid lines for both x & y axes
ax3.ticklabel_format(style='plain')
ax3.minorticks_on()

plt.show()
