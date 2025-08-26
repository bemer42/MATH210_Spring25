#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 01:11:23 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective: This program will plot the three basic cases of the 2-D
# Rectangular Minimization Problem.  

# Define an array of perimeter values:
N = 1e3
a = np.linspace(0, 100, int(N))

# Define the max area function as a function of two variables:
def MinP(a,b):
    return np.sqrt(2**(4-b) * a)

# Plot all three max area functions on the same plot:
plt.figure(1)
plt.plot(a, MinP(a,0), 'k', linewidth=3, label='No Border')
plt.plot(a, MinP(a,1), 'b', linewidth=3, label='One Border')
plt.plot(a, MinP(a,2), 'r', linewidth=3, label='Two Borders')

# Customize plot appearance:
plt.title('Minimum Rectangular Perimeter', fontsize=20)
plt.xlabel('Area (A)', fontsize=15)
plt.ylabel('Minimum Perimeter (P)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()


# Save the file in high quality format: 
plt.savefig('Rectangular_MinPerimeter.eps', format='eps')

# Show the plot
plt.show()


