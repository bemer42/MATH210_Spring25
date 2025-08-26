#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 01:33:55 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt

# Objective: This program will plot the three basic cases of the 3-D
# Square-based Box Minimization Problem.  

# Define an array of perimeter values:
N = 1e3
v = np.linspace(0, 100, int(N))

# Define the max area function as a function of two variables:
def MinA(v,b):
    return 6*((1-b/4)*v)**(2/3)

# Plot all three max area functions on the same plot:
plt.figure(1)
plt.plot(v, MinA(v,0), 'k', linewidth=3, label='No Border')
plt.plot(v, MinA(v,1), 'b', linewidth=3, label='One Border')
plt.plot(v, MinA(v,2), 'r', linewidth=3, label='Two Borders')

# Customize plot appearance:
plt.title('Minimum Surface Area', fontsize=20)
plt.xlabel('Volume (V)', fontsize=15)
plt.ylabel('Minimum Surface Area (A)', fontsize=15)
plt.gca().tick_params(labelsize=10)
plt.legend()
plt.grid(True, which='both')
plt.minorticks_on()


# Save the file in high quality format: 
plt.savefig('Square-based_Box_MinSurfaceArea.eps', format='eps')

# Show the plot
plt.show()