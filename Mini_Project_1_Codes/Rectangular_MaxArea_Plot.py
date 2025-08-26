#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 00:31:30 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective: This program will plot the three basic cases of the 2-D
# Rectangular Maximization Problem.  

# Define an array of perimeter values:
N = 1e3
p = np.linspace(0, 100, int(N))

# Define the max area function as a function of two variables:
def MaxA(p,b):
    return 2**(b-4) * p**2

# Plot all three max area functions on the same plot:
plt.figure(1)
plt.plot(p, MaxA(p,0), 'k', linewidth=3, label='No Border')
plt.plot(p, MaxA(p,1), 'b', linewidth=3, label='One Border')
plt.plot(p, MaxA(p,2), 'r', linewidth=3, label='Two Borders')

# Customize plot appearance:
plt.title('Maximum Rectangular Area', fontsize=20)
plt.xlabel('Perimeter (P)', fontsize=15)
plt.ylabel('Maximum Area (A)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()


# Save the file in high quality format: 
plt.savefig('Rectangular_MaxArea.eps', format='eps')

# Show the plot
plt.show()


