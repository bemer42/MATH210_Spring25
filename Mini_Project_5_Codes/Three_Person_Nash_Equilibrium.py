#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:30:24 2024

@author: brooksemerick
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:14:46 2024

@author: bemerick
"""

import numpy as np
from scipy.optimize import linprog
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

# Define the L vs C-R coalition matrix: 
L_v_CR = np.hstack((L_PM[0,:,:],L_PM[1,:,:]))

# Define the C vs L-R coalition matrix: 
C_v_LR = np.hstack((C_PM[0,:,:].T,C_PM[1,:,:].T))

# Define the R vs L-C coalition matrix: 
R_v_LC = np.hstack((R_PM[:,:,0],R_PM[:,:,1]))

#%% Define a global function for solving both Player/Coalition games:
def LP_Game_Solver(PM):
    
    # Define matrices, vectors, and bounds for Single Player's game:
    A_ub = np.concatenate((-PM.T, np.ones((PM.shape[1],1))), axis=1)
    b_ub = np.zeros(PM.shape[1])
    A_eq = np.ones((1,PM.shape[0]+1))
    A_eq[0,-1] = 0
    b_eq = 1
    c = np.zeros(PM.shape[0]+1)
    c[-1] = -1
    bounds = [(0,1)] * (PM.shape[0]) + [(None, None)] 

    # Call linprog to solve LP for Single Player: 
    Single = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)
    
    # Define matrices, vectors, and bounds for Coalition's game: 
    A_ub = np.concatenate((PM, -np.ones((PM.shape[0],1))), axis=1)
    b_ub = np.zeros(PM.shape[0])
    A_eq = np.ones((1,PM.shape[1]+1))
    A_eq[0,-1] = 0
    b_eq = 1
    c = np.zeros(PM.shape[1]+1)
    c[-1] = 1
    bounds = [(0,1)] * (PM.shape[1]) + [(None, None)] 
    
    # Call linprog to solve LP: 
    Coalition = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)

    return Single, Coalition
        
#%% L versus C-R:
Res_L, Res_CR = LP_Game_Solver(L_v_CR)

if Res_L.success:    
    prob_L = np.round(Res_L.x[:-1],4)
    prob_C = np.round(np.array([Res_CR.x[0] + Res_CR.x[2], Res_CR.x[1] + Res_CR.x[3]]),4)
    prob_R = np.round(np.array([Res_CR.x[0] + Res_CR.x[1], Res_CR.x[2] + Res_CR.x[3]]),4)
    prob_mat = np.hstack((prob_R[0]*np.vstack((prob_C,prob_C)),prob_R[1]*np.vstack((prob_C,prob_C))))
    prob_mat[0,:] = prob_L[0]*prob_mat[0,:]
    prob_mat[1,:] = prob_L[1]*prob_mat[1,:]
    L_v_CR_Payouts = np.round(np.array([np.sum(prob_mat*np.hstack((L_PM[0,:,:], L_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((C_PM[0,:,:], C_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((R_PM[0,:,:], R_PM[1,:,:])))]),4)
        
    # Output Table of probabilities: 
    row_labels = ['Strategy 1 Prob', 'Strategy 2 Prob']
    data = [row_labels, prob_L, prob_C, prob_R]
    column_labels = ["L vs. C-R", "Player L", "Player C", "Player R"]
    table = PrettyTable()
    table.field_names = column_labels
    for row in zip(*data):
        table.add_row(row)
    print(table)
else:
    print("Failed to find a Nash Equilibrium")

#%% C versus L-R:
Res_C, Res_LR = LP_Game_Solver(C_v_LR)

if Res_C.success:    
    prob_C = np.round(Res_C.x[:-1],4)
    prob_L = np.round(np.array([Res_LR.x[0] + Res_LR.x[2], Res_LR.x[1] + Res_LR.x[3]]),4)
    prob_R = np.round(np.array([Res_LR.x[0] + Res_LR.x[1], Res_LR.x[2] + Res_LR.x[3]]),4)
    prob_mat = np.hstack((prob_R[0]*np.vstack((prob_C,prob_C)),prob_R[1]*np.vstack((prob_C,prob_C))))
    prob_mat[0,:] = prob_L[0]*prob_mat[0,:]
    prob_mat[1,:] = prob_L[1]*prob_mat[1,:]
    C_v_LR_Payouts = np.round(np.array([np.sum(prob_mat*np.hstack((L_PM[0,:,:], L_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((C_PM[0,:,:], C_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((R_PM[0,:,:], R_PM[1,:,:])))]),4)
    # Output Table of probabilities: 
    row_labels = ['Strategy 1 Prob', 'Strategy 2 Prob']
    data = [row_labels, prob_L, prob_C, prob_R]
    column_labels = ["C vs. L-R", "Player L", "Player C", "Player R"]
    table = PrettyTable()
    table.field_names = column_labels
    for row in zip(*data):
        table.add_row(row)
    print(table)
else:
    print("Failed to find a Nash Equilibrium")
    
#%% R versus L-C:
Res_R, Res_LC = LP_Game_Solver(R_v_LC)

if Res_R.success:    
    prob_R = np.round(Res_R.x[:-1],4)
    prob_L = np.round(np.array([Res_LC.x[0] + Res_LC.x[2], Res_LC.x[1] + Res_LC.x[3]]),4)
    prob_C = np.round(np.array([Res_LC.x[0] + Res_LC.x[1], Res_LC.x[2] + Res_LC.x[3]]),4)
    prob_mat = np.hstack((prob_R[0]*np.vstack((prob_C,prob_C)),prob_R[1]*np.vstack((prob_C,prob_C))))
    prob_mat[0,:] = prob_L[0]*prob_mat[0,:]
    prob_mat[1,:] = prob_L[1]*prob_mat[1,:]
    R_v_LC_Payouts = np.round(np.array([np.sum(prob_mat*np.hstack((L_PM[0,:,:], L_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((C_PM[0,:,:], C_PM[1,:,:]))),
                                        np.sum(prob_mat*np.hstack((R_PM[0,:,:], R_PM[1,:,:])))]),4)
    # Output Table of probabilities: 
    row_labels = ['Strategy 1 Prob', 'Strategy 2 Prob']
    data = [row_labels, prob_L, prob_C, prob_R]
    column_labels = ["R vs. L-C", "Player L", "Player C", "Player R"]
    table = PrettyTable()
    table.field_names = column_labels
    for row in zip(*data):
        table.add_row(row)
    print(table)
else:
    print("Failed to find a Nash Equilibrium")
    
#%%
# Summary of Coalitions: 
row_labels = ['L vs. C-R', 'C vs. L-R', 'R vs. L-C']
data = [row_labels, L_v_CR_Payouts, C_v_LR_Payouts, R_v_LC_Payouts]
column_labels = [" ", "Payout to L", "Payout to C", "Payout to R"]
table = PrettyTable()
table.field_names = column_labels
for row in zip(*data):
    table.add_row(row)
print(table)


