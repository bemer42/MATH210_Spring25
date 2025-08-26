# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:47:56 2024

@author: kayla
"""

import math
import matplotlib.pyplot as plt


def calculate_greenhouse(S, A):
    # Calculate r using the area formula A = 2Srx + (S/2) * pi * r^2
    r = math.sqrt((5 * A) / (106 * math.pi))

    # Calculate x using the formula derived from the area
    num_= A - (S / 2) * math.pi * r**2
    den_ = 2 * S * r
    x = num_ / den_ #ft

    # Calculate the perimeter P
    P = (S + 1) * x + (2 + math.pi) * S * r #ft

    return r,x,P


# Values for S
S_values = [4, 3, 2, 1]

# Range of A values
# Range from 100 to 2000 with a step of N
N=100
A_values = range(100, 2000, N) #ft^2

# Plot the results
plt.figure(figsize=(10, 7))

for S in S_values:
    P_values = [calculate_greenhouse(S, A)[2] for A in A_values]
    #P extracted with [2]
    plt.plot(A_values, P_values, marker='o', label=f'S = {S}')

plt.title('Perimeter_min (P) vs Area (A)')
plt.xlabel('Area (A)')
plt.ylabel('Perimeter_min (P)')
plt.legend()
plt.grid(True, which='both')

# Save the file in high quality format: 
plt.savefig('Greenhouse_minPerimeter.eps', format='eps')

#show the plot
plt.show()
