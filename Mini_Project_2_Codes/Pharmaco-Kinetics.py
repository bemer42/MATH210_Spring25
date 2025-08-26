#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 02:58:24 2024

@author: brooksemerick
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Time Discretization
t_0 = 0
t_end = 70
N_time = int(1e3)
t_span = np.linspace(t_0, t_end, N_time)

# Define parameters
kg = 0.5
kb = 0.1
absorb = 0.5
dose = 6

# Define initial conditions
Y_0 = [1, 0]

def RHS_Pharmaco_Function(t, Y):
    # Define variables
    Ag = Y[0]
    Ab = Y[1]

    # Define Intake function
    def I(t):
        return 1 * (t % dose <= absorb) + 0 * (t % dose > absorb)

    # Define differential equations
    dAg_dt = I(t) - kg * Ag
    dAb_dt = kg * Ag - kb * Ab

    # Assemble equations into a list (representing a column vector)
    return [dAg_dt, dAb_dt]

# Implement Stiff ODE solver
sol = solve_ivp(RHS_Pharmaco_Function, [t_0, t_end], Y_0, t_eval=t_span, method='BDF')

# Extract Solutions
Ag = sol.y[0, :]
Ab = sol.y[1, :]
t = sol.t

# Plot Solution
plt.figure(figsize=(10, 6))
plt.plot(t, Ag, 'k-', linewidth=5, label='A_g(t)')
plt.plot(t, Ab, 'b-', linewidth=5, label='A_b(t)')
plt.title('Pharmaco-Kinetics Trajectory', fontsize=20)
plt.xlabel('Time (t)', fontsize=16)
plt.ylabel('Concentration', fontsize=16)
plt.legend(fontsize=14)
plt.xlim([t_0, t_end])
plt.grid(True)
plt.show()