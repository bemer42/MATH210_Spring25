# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:24:53 2024

@author: bemerick
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Define the data vectors x and y:
x = np.linspace(1, 5, int(50))
y = (6*np.exp(2*x)) + 5*np.random.normal(0, 1, len(x))

# Build "Exponential Vandermonde":
A = np.vstack((np.ones(len(x)), x)).T

# Build Normal equation with the y data on log scale: 
A_norm = A.T@A
y_norm = A.T@np.log(y)

# Solve the linear system (A^TA)c = A^Tlog(y) for c: 
c = LA.solve(A_norm, y_norm)
# c = LA.inv(A_norm)@y_norm
# c = LA.lstsq(A, y)[0]

# Define a and b: 
a = np.exp(c[0])
b = c[1]

# Create ararys of data for exponential curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and exponential curve fit:
plt.figure(1)
plt.plot(x_plot, a*np.exp(b*x_plot), 'r-', linewidth=4, label = "Exp Fit")
plt.plot(x, y, 'ko', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Exponential Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('Exponential_Curve_Fitting.eps', format='eps')

# Show plot:
plt.show()

# Output coefficients of exponential fit: 
print(f"Coefficients = {c}")

# Calculate the residual sum of squares (SS_res) on log data:
ss_res = np.sum((np.log(y) - (c[0] + c[1]*x)) ** 2)

# Calculate the total sum of squares (SS_tot) on log data:
ss_tot = np.sum((np.log(y) - np.mean(np.log(y))) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")