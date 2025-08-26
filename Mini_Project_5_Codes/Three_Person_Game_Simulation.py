#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:20:20 2024

@author: brooksemerick
"""

import numpy as np
from prettytable import PrettyTable

# Define the 3D payoff matrix to player L:
L_PM = np.array([[[1, -4],
                  [2, -5]],
                 
                 [[3, -6],
                  [2, -2]]])

# Define the 3D payoff matrix to player C:
C_PM = np.array([[[1, 3],
                  [-4, -5]],
                 
                 [[-2, -6],
                  [2, 3]]])

# Define the 3D payoff matrix to player R:
R_PM = np.array([[[-2, 1],
                 [2, 10]],
                 
                 [[-1, 12],
                  [-4, -1]]])

#%%

def sim_game(L_PM, C_PM, R_PM, iterations):
    
    n_L, n_C, n_R = L_PM.shape
    Res_L, Res_C, Res_R = [], [], []
    
    # Initialize frequency and probability arrays:
    freq_L = np.ones(n_L)
    freq_C = np.ones(n_C)
    freq_R = np.ones(n_R)
    
    prob_L = freq_L / freq_L.sum()
    prob_C = freq_C / freq_C.sum()
    prob_R = freq_R / freq_R.sum()
    
    # Initialize cumulative regret arrays: 
    reg_L = np.zeros(n_L)
    reg_C = np.zeros(n_C)
    reg_R = np.zeros(n_R)
     
    for _ in range(iterations):
        # Randomly select strategies for each player:
        st_L = np.random.choice(n_L, p = prob_L)
        st_C = np.random.choice(n_C, p = prob_C)
        st_R = np.random.choice(n_R, p = prob_R)
        
        # Calculate the payoff:
        L_payoff = L_PM[st_L, st_C, st_R]
        C_payoff = C_PM[st_L, st_C, st_R]
        R_payoff = R_PM[st_L, st_C, st_R]
        
        # Append payoff to results array: 
        Res_L.append(L_payoff)
        Res_C.append(C_payoff)
        Res_R.append(R_payoff)
        
        # Update regret for Player L:
        for i in range(n_L):
            reg_L[i] += L_PM[i, st_C, st_R] - L_payoff
            
        # Update regret for Player C:
        for j in range(n_C):
            reg_C[j] += C_PM[st_L, j, st_R] - C_payoff
        
        # Update regret for Player C:
        for k in range(n_R):
            reg_R[k] += R_PM[st_L, st_C, k] - R_payoff
            
        # Update the frequencies: 
        if np.max(reg_L) > 0:
            freq_L = np.maximum(reg_L,0)
        else:
            freq_L = reg_L - np.min(reg_L) + 1
            
        if np.max(reg_C) > 0:
            freq_C = np.maximum(reg_C,0)
        else:
            freq_C = reg_C - np.min(reg_C) + 1

        if np.max(reg_R) > 0:
            freq_R = np.maximum(reg_R,0)
        else:
            freq_R = reg_R - np.min(reg_R) + 1    
        
        # Update probabilities:
        prob_L = freq_L / freq_L.sum()
        prob_C = freq_C / freq_C.sum()
        prob_R = freq_R / freq_R.sum()
        
    return Res_L, Res_C, Res_R, prob_L, prob_C, prob_R


# Run Simulation:
N = int(1e4)
Res_L, Res_C, Res_R, prob_L, prob_C, prob_R  = sim_game(L_PM, C_PM, R_PM, N)

# Estimate value of the game: 
v_L = np.round(np.mean(Res_L),4)
v_C = np.round(np.mean(Res_C),4)
v_R = np.round(np.mean(Res_R),4)

# Summary of results: 
L_sum = np.round(np.hstack((prob_L, v_L)),4)
C_sum = np.round(np.hstack((prob_C, v_C)),4)
R_sum = np.round(np.hstack((prob_R, v_R)),4)

"Strategy 1 Prob", "Strategy 2 Prob", "Payout"
row_labels = ["Strategy 1 Prob", "Strategy 2 Prob", "Payout"]
data = [row_labels, L_sum, C_sum, R_sum]
column_labels = [' ', 'Player L', 'Player C', 'Player R']
table = PrettyTable()
table.field_names = column_labels
for row in zip(*data):
    table.add_row(row)
print(table)




