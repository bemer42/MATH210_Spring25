#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:29:13 2024

@author: brooksemerick
"""
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Define the data vectors x and y:
x = np.linspace(-5, 5, int(50))
y = (50/(.8*x**2 + 1)) + .3*np.random.normal(0, 1, len(x))

# Build "Agnesi Vandermonde":
A = np.vstack((np.ones(len(x)), -x**2 * y)).T

# Build Normal equation with the y data: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = LA.solve(A_norm, y_norm)
# c = LA.inv(A_norm)@y_norm
# c = LA.lstsq(A, y)[0]

# Create ararys of data for Agnesi curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and Agnesi curve fit:
plt.figure(1)
plt.plot(x_plot, c[0]/(c[1]*x_plot**2 + 1), 'r-', linewidth=4, label = "Agnesi Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Agnesi Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('Agnesi_Curve_Fitting.eps', format='eps')

# Show plot:
plt.show()

# Output coefficients of Agnesi fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res):
ss_res = np.sum((y - (c[0]/(c[1]*x**2 + 1))) ** 2)

# Calculate the total sum of squares (SS_tot):
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")