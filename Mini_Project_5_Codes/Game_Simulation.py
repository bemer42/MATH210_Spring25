# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:13:29 2024

@author: bemerick
"""

import numpy as np

# Define the payoff matrix to player A:
PM = np.array([
    [2, 2, 3, -1],
    [4, 3, 2, 6]
])

def sim_game(PM, iterations):
    
    n_A, n_B = PM.shape
    results = []
    
    # Initialize frequency and probability arrays:
    freq_A = np.ones(n_A)
    freq_B = np.ones(n_B)
    
    prob_A = freq_A / freq_A.sum()
    prob_B = freq_B / freq_B.sum()
    
    # Initialize cumulative regret arrays: 
    reg_A = np.zeros(n_A)
    reg_B = np.zeros(n_B)
     
    for _ in range(iterations):
        # Randomly select strategies for each player:
        st_A = np.random.choice(n_A, p = prob_A)
        st_B = np.random.choice(n_B, p = prob_B)
        
        # Calculate the payoff:
        payoff = PM[st_A, st_B]
        
        # Append payoff to results array: 
        results.append(payoff)
        
        # Update regret for Player A:
        for i in range(n_A):
            reg_A[i] += PM[i,st_B] - payoff
            
        # Update regret for Player B:
        for j in range(n_B):
            reg_B[j] -= PM[st_A,j] - payoff
        
        # Update the frequencies: 
        if np.max(reg_A) > 0:
            freq_A = np.maximum(reg_A,0)
        else:
            freq_A = reg_A - np.min(reg_A) + 1
            
        if np.max(reg_B) > 0:
            freq_B = np.maximum(reg_B,0)
        else:
            freq_B = reg_B - np.min(reg_B) + 1

        
        # Update the probabilities (phase 2): 
        # if freq_A.sum() > 0:
        #     prob_A = freq_A / freq_A.sum()
        # else:
        #     prob_A = np.ones(n_A) / n_A
            
        # if freq_B.sum() > 0:    
        #     prob_B = freq_B / freq_B.sum()
        # else:
        #     prob_B = np.ones(n_B) / n_B
        
        # # Reward for winning strategy (phase 1):
        # if payoff > 0:
        #     freq_A[st_A] += 1
        # elif payoff < 0:
        #     freq_B[st_B] += 1
        # else:
        #     pass
        
        # Update probabilities:
        prob_A = freq_A / freq_A.sum()
        prob_B = freq_B / freq_B.sum()
            
    return results, prob_A, prob_B


# Run Simulation:
N = int(1e4)
results, prob_A, prob_B = sim_game(PM,N)

# Estimate value of the game: 
v = np.mean(results)

print(f"Average payoff for Player A over {N} games: {v}")
print(f"Optimal Strategy for Player A: {prob_A}")
print(f"Optimal Strategy for Player B: {prob_B}")








