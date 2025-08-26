#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 01:51:20 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective: This program will plot the surface of the objective
# function that is to be optimized for maximum volume of a box
# with dimensions x, y, z.  The code takes in a value of surface
# area and border, and outputs the optimized dimensions as well 
# as the maximum volume.  

# Create a meshgrid in x and y:
N = 1e3
x = np.linspace(1, 50, int(N))
y = np.linspace(1, 50, int(N))
X, Y = np.meshgrid(x, y)

# Define a surface area value: 
A = 1000; 

# Define surface function:
def V(x, y):
    return A * x * y / (x + y) / 2 - (x * y)**2 / (x + y)

# Create an array of volume values to be plotted: 
V_plot = V(X, Y)

# Set negative values in V_plot to 0
V_plot[V_plot < 0] = 0

# Plot surface function:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, V_plot, cmap='viridis')

#Customize plot appearance
plt.title('3D Surface Plot of the Volume Function', fontsize=20)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
ax.set_zlabel('V(X,Y)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.grid(True, which='both')
plt.minorticks_on()
ax.set_box_aspect([1,1,1])  

# Save the file in high quality format: 
plt.savefig('Box_MaxVolume.eps', format='eps')

# Show the plot
plt.show()

# Find the maximum value in V_plot
Max_Volume = np.max(V_plot)

# Find the indices where V_plot is equal to the maximum value
ind = np.where(V_plot == Max_Volume)

# Extract the corresponding X and Y values
X_max = X[ind][0]
Y_max = Y[ind][0]

# Calculate the associated value using the given formulaa
Z_max = (A - 2 * X_max * Y_max) / (2 * X_max + 2 * Y_max)

# Print the results
print(f"X value at maximum V: {X_max}")
print(f"Y value at maximum V: {Y_max}")
print(f"Z value at maximum V: {Z_max}")
print(f"Maximum Volume: {Max_Volume}")

