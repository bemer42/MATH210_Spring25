#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 01:19:00 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective: This program will plot the three basic cases of the 3-D
# Square-based Box Maximization Problem.  

# Define an array of perimeter values:
N = 1e3
a = np.linspace(0, 100, int(N))

# Define the max area function as a function of two variables:
def MaxV(a,b):
    return (2*a**(3/2)) / 3 / (4-b) / np.sqrt(6)

# Plot all three max area functions on the same plot:
plt.figure(1)
plt.plot(a, MaxV(a,0), 'k', linewidth=3, label='No Border')
plt.plot(a, MaxV(a,1), 'b', linewidth=3, label='One Border')
plt.plot(a, MaxV(a,2), 'r', linewidth=3, label='Two Borders')

# Customize plot appearance:
plt.title('Maximum Volume', fontsize=20)
plt.xlabel('Surface Area (A)', fontsize=15)
plt.ylabel('Maximum Volume (V)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()

# Save the file in high quality format: 
plt.savefig('Square-based_Box_MaxVolume.eps', format='eps')

# Show the plot
plt.show()
