# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:17:42 2024

@author: kayla
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parameters
G = 0.1
m2 = 1000  # kg
m1 = 8000

# Differential equations
def dYdt(t, Y):
    x1, vx1, y1, vy1 = Y
    dx1dt = vx1
    dvx1dt = -G * m2 / ((x1)**2 + (y1)**2)**(3/2) * (x1)
    dy1dt = vy1
    dvy1dt = -G * m2 / ((x1)**2 + (y1)**2)**(3/2) * (y1)
    return [dx1dt, dvx1dt, dy1dt, dvy1dt]

# Time settings
t_0 = 0
t_end = 100
N = int(1e6)
t_span = np.linspace(t_0, t_end, N)

# Initial Conditions
x_0 = 8
vx_0 = 2
y_0 = 10
vy_0 = -1
Y_0 = [x_0, vx_0, y_0, vy_0]

# Solve the system
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='BDF')
x1 = sol.y[0, :]
y1 = sol.y[2, :]

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)  # Adjust limits based on the solution range
ax.set_ylim(-20, 20)

# Create line objects for the moving object and its path
line, = ax.plot([], [], 'ro')  # Red dot for the object
path, = ax.plot([], [], 'b-', lw=1)  # Blue line for the path

# Lists to store the path coordinates
x_path, y_path = [], []

# Phase plane plot one-body problem
plt.figure(1)
plt.plot(x1, sol.y[1, :], 'b-', label="mass 1")  # Phase plot for x vs vx
plt.plot(y1, sol.y[3, :], 'r-', label="mass 2")  # Phase plot for y vs vy
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title("Phase Plane Plot (One-Body Problem)")
plt.legend()
plt.grid()
plt.show()  # Show the phase plane plot in Figure 1

# Animated plot setup
fig2 = plt.figure(2)  
ax = fig2.add_subplot(1,1,1)
ax.set_xlim(-20, 20)  
ax.set_ylim(-20, 20)
ax.set_title('Animated One-body Problem with Path')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.grid(True)

# Create line objects for the moving object and its path
line, = ax.plot([], [], 'ro')  # Red dot for the object
path, = ax.plot([], [], 'b-', lw=1)  # Blue line for the path

# Lists to store the path coordinates
x_path, y_path = [], []

# Initialization function
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path

# Update function for each frame
def update(frame):
    x_path.append(x1[frame])
    y_path.append(y1[frame])
    line.set_data([x1[frame]], [y1[frame]])  # Update the dot's position
    path.set_data(x_path, y_path)            # Update the path
    return line, path

# Create the animation attached to fig2
deframe = 1000
ani = FuncAnimation(fig2, update, frames=range(0, len(t_span), deframe), init_func=init, blit=True, interval=2)

plt.show()  # Show the animated plot in Figure 2



