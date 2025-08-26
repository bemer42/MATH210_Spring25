# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:06:18 2024

@author: bemerick
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Define the matrix A and the right hand side b:
x = np.linspace(1, 10, int(50))
y = (3 + np.sin(x) + 7*np.cos(x)) + 1*np.random.normal(0, 1, len(x))

# Build "Trig Vandermonde":
A = np.vstack((np.ones(len(x)), np.sin(x), np.cos(x))).T

# Build Normal equation: 
A_norm = A.T@A
y_norm = A.T@y

# Solve the linear system Ac = y for c: 
c = LA.solve(A_norm, y_norm)
# c = LA.inv(A_norm)@y_norm
# c = LA.lstsq(A, y)[0]

# Create ararys of data and curve fit: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))

# Plot the data and trig curve fit:
plt.figure(1)
plt.plot(x_plot, c[0] + c[1]*np.sin(x_plot) + c[2]*np.cos(x_plot), 'r-', \
         linewidth=4, label = "Trig Fit")
plt.plot(x, y, 'bo', linewidth=5, label = "Data")

#Customize the plot:
plt.title('Trigonometric Fit', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Show plot:
plt.show()

# Output coefficients of trigonometric curve fit: 
print(f"Coefficients = {c}")

# Save the file in high quality format: 
plt.savefig('Trigonometric_Curve_Fitting.eps', format='eps')


# Calculate the residual sum of squares (SS_res)
ss_res = np.sum((y - (c[0] + c[1]*np.sin(x) + c[2]*np.cos(x))) ** 2)

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# Compute the R^2 value for each type of fit:
R2 = 1 - (ss_res / ss_tot)

# Output the R^2 Value for each fit: 
print(f"R^2 = {R2}")
