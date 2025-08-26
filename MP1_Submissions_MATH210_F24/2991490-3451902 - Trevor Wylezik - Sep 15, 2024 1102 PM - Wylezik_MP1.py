# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:13:49 2024

@author: trevo
"""

import numpy as np
import matplotlib.pyplot as plt

# Define an array of perimeter values:
N = 1001 # if you put 5, it will make up 5 lines to build a curve
V = np.linspace(0,1000,N) # Plotting a certain # of points and connecting lines

a = np.deg2rad(30) # The degree (o) of the roof slope
m1 = 0 # Wooden Roof (No material added)
m2 = 4.68 # Asphalt Roof
m3 = 9.30 # Clay Roof
m4 = 15.00 # Metal Roof
m5 = 27.63 # Slate Roof

def Cost_min(V,a,m): # Creating a function Cost_min
    beta = V / (1 + (1/2)*np.tan(a)) # Creating the beta simplification component
    delta = 45 + 15*np.tan(a) + ((15 + m)/(np.cos(a))) # Creating the delta simplification component
    sigma = 30 + 15*np.tan(a) # Creating the sigma simplification component
    result = (3/2) * ((beta*delta)**(2/3)) * ((2*sigma)**(1/3)) # Final calculation with simplification components
    return result # Return the value

# Plot the max area function against A:
plt.figure(1)
plt.plot(V,Cost_min(V,a,m1), linewidth = 2, label=f'Wood (${m1}/sq ft.)') # Wood
plt.plot(V,Cost_min(V,a,m2), linewidth = 2, label=f'Asphalt (${m2}/sq ft.)') # Asphalt
plt.plot(V,Cost_min(V,a,m3), linewidth = 2, label=f'Clay (${m3}/sq ft.)') # Clay
plt.plot(V,Cost_min(V,a,m4), linewidth = 2, label=f'Metal (${m4}/sq ft.)') # Metal 
plt.plot(V,Cost_min(V,a,m5), linewidth = 2, label=f'Slate (${m5}/sq ft.)') # Slate

# Plot area formatting
plt.title(f'Shed with Sloped Roof (α = {round(np.rad2deg(a))})',fontsize=20) # Only formats the title
plt.xlabel('Volume (Cubic ft.)', fontsize=15) # Only formats the x-axis
plt.ylabel('Minimum Cost ($)', fontsize=15) # Only formats the y-axis
plt.gca().tick_params(labelsize=10) # Tick mark labels
plt.legend() # Adds a legend
plt.grid(True, which='both') # Adds grid lines for both x & y axes
plt.minorticks_off()    

# Shows the plot:
plt.show()