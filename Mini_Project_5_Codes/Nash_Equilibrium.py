# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:14:46 2024

@author: bemerick
"""

import numpy as np
from scipy.optimize import linprog

# Define the payoff matrix to player A:
PM = np.array([
    [0, 2, 6, 0, 10],
    [-2, 0, 0, 3, 20],
    [3, 0, 0, -4, 30],
    [0, -3, 4, 0, 40]
])

# Define matrices and vectors for Player A's game: 
A_ub = np.concatenate((-PM.T, np.ones((PM.shape[1],1))), axis=1)
b_ub = np.zeros(PM.shape[1])
A_eq = np.ones((1,PM.shape[0]+1))
A_eq[0,-1] = 0
b_eq = 1
c = np.zeros(PM.shape[0]+1)
c[-1] = -1

# Define bounds on variables:
bounds = [(0,1)] * (PM.shape[0]) + [(None, None)] 

# Call linprog to solve LP: 
Result_A = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)

if Result_A.success:    
    prob_A = Result_A.x[:-1]
    v_A = Result_A.x[-1]
    print(f"Player A's Optimal Strategy: {prob_A}")
    print(f"Value of the game: {v_A}")
else:
    print("Failed to find a Nash Equilibrium")


# Define matrices and vectors for Player B's game: 
A_ub = np.concatenate((PM, -np.ones((PM.shape[0],1))), axis=1)
b_ub = np.zeros(PM.shape[0])
A_eq = np.ones((1,PM.shape[1]+1))
A_eq[0,-1] = 0
b_eq = 1
c = np.zeros(PM.shape[1]+1)
c[-1] = 1

# Define bounds on variables:
bounds = [(0,1)] * (PM.shape[1]) + [(None, None)] 

# Call linprog to solve LP: 
Result_B = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)

if Result_B.success:    
    prob_B = Result_B.x[:-1]
    v_B = Result_B.x[-1]
    print(f"Player B's Optimal Strategy: {prob_B}")
    print(f"Value of the game: {v_B}")
else:
    print("Failed to find a Nash Equilibrium")