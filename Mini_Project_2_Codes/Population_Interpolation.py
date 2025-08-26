# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:58:34 2024

@author: bemerick
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

# Define the matrix A and the right hand side b:
x = np.array([0, 10, 20, 30])
y_1 = np.array([227.225, 249.623, 282.172, 308.282])
y_2 = np.array([984.736, 1148.364, 1263.638, 1330.141])
y_3 = np.array([78.298, 79.380, 82.184, 81.644])

A = np.vander(x, increasing=True)
DET = LA.det(A)
print(f"Determinant = {DET}")

# Solve the linear system Ac = y for c: 
c_1 = LA.solve(A, y_1)
c_2 = LA.solve(A, y_2)
c_3 = LA.solve(A, y_3)

def f(x,y):
    c = LA.solve(A,y)
    return np.polyval(np.flip(c), x)

# Create ararys of data and interpolant: 
x_plot = np.linspace(np.min(x), np.max(x), int(1e4))
p1_plot = np.polyval(np.flip(c_1), x_plot)
p2_plot = np.polyval(np.flip(c_2), x_plot)
p3_plot = np.polyval(np.flip(c_3), x_plot)

# Plot the data and interpolant:
plt.figure(1)
plt.plot(x_plot+1980, f(x_plot, y_1), 'r-', linewidth=4, label = "USA")
plt.plot(x_plot+1980, p2_plot, 'b-', linewidth=4, label = "China")
plt.plot(x_plot+1980, p3_plot, 'g-', linewidth=4, label = "Germany")
plt.plot(x+1980, y_1, 'ko', linewidth=5)
plt.plot(x+1980, y_2, 'ko', linewidth=5)
plt.plot(x+1980, y_3, 'ko', linewidth=5)

#Customize the plot:
plt.title('Population Interpolation', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Population (millions)', fontsize=15)
plt.gca().tick_params(labelsize=10)
# plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Show plot:
plt.show()

# Evaluate years: 
print(f"China Population in 1992 = {np.polyval(np.flip(c_2), 12)}")
print(f"USA Population in 1984 = {f(4,y_1)}")