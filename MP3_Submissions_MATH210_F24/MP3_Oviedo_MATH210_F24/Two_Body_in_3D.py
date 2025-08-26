# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:32:14 2024

@author: kayla
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parameters
G = .1  # Gravitational constant, set to 1 for simplicity
m1 = 1000  # Mass of body 1
m2 = 600  # Mass of body 2

# Differential equations for two-body 3D motion
def dYdt(t, Y):
    # Extract positions and velocities
    x1, vx1, y1, vy1, z1, vz1 = Y[:6]   # Body 1
    x2, vx2, y2, vy2, z2, vz2 = Y[6:]   # Body 2

    # Calculate the distance components
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    # Gravitational acceleration
    dvx1dt = G * m2 * dx / r**3
    dvy1dt = G * m2 * dy / r**3
    dvz1dt = G * m2 * dz / r**3

    dvx2dt = -G * m1 * dx / r**3
    dvy2dt = -G * m1 * dy / r**3
    dvz2dt = -G * m1 * dz / r**3

    return [vx1, dvx1dt, vy1, dvy1dt, vz1, dvz1dt, 
            vx2, dvx2dt, vy2, dvy2dt, vz2, dvz2dt]

# Time settings
t_0 = 0
t_end = 100
N = 1000  # Reduced frame count for faster animation
t_span = np.linspace(t_0, t_end, N)

# Initial conditions: [x1, vx1, y1, vy1, z1, vz1, x2, vx2, y2, vy2, z2, vz2]
Y_0 = [5, 1, 5, -0.1, 2, 1,    # Body 1 initial position and velocity
       -5, -1, -5, 0.3, 2, -.2]   # Body 2 initial position and velocity

# Solve the system
sol = solve_ivp(dYdt, [t_0, t_end], Y_0, t_eval=t_span, method='RK45')
x1, y1, z1 = sol.y[0, :], sol.y[2, :], sol.y[4, :]
x2, y2, z2 = sol.y[6, :], sol.y[8, :], sol.y[10, :]

# Calculate the center of mass
com_x = (m1 * x1 + m2 * x2) / (m1 + m2)
com_y = (m1 * y1 + m2 * y2) / (m1 + m2)
com_z = (m1 * z1 + m2 * z2) / (m1 + m2)

# Set up the 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 50)
ax.set_ylim(-10, 50)
ax.set_zlim(-10, 50) 

# Create line objects for the moving objects, their paths, and the center of mass
body1, = ax.plot([], [], [], 'ro')  # Red dot for Body 1
body2, = ax.plot([], [], [], 'bo')  # Blue dot for Body 2
path1, = ax.plot([], [], [], 'r-', lw=2)  # Path of Body 1
path2, = ax.plot([], [], [], 'b-', lw=2)  # Path of Body 2
com_dot, = ax.plot([], [], [], 'go')  # Green dot for Center of Mass
com_path, = ax.plot([], [], [], 'g--', lw=3)  # Path of Center of Mass

# Initialize path lists
x1_path, y1_path, z1_path = [], [], []
x2_path, y2_path, z2_path = [], [], []
com_path_x, com_path_y, com_path_z = [], [], []

# Initialization function
def init():
    body1.set_data([], [])
    body1.set_3d_properties([])
    body2.set_data([], [])
    body2.set_3d_properties([])
    path1.set_data([], [])
    path1.set_3d_properties([])
    path2.set_data([], [])
    path2.set_3d_properties([])
    com_dot.set_data([], [])
    com_dot.set_3d_properties([])
    com_path.set_data([], [])
    com_path.set_3d_properties([])
    return body1, body2, path1, path2, com_dot, com_path

# Update function for each frame
def update(frame):
    x1_path.append(x1[frame])
    y1_path.append(y1[frame])
    z1_path.append(z1[frame])
    x2_path.append(x2[frame])
    y2_path.append(y2[frame])
    z2_path.append(z2[frame])
    
    com_path_x.append(com_x[frame])
    com_path_y.append(com_y[frame])
    com_path_z.append(com_z[frame])

    body1.set_data([x1[frame]], [y1[frame]])
    body1.set_3d_properties([z1[frame]])
    body2.set_data([x2[frame]], [y2[frame]])
    body2.set_3d_properties([z2[frame]])
    
    path1.set_data(x1_path, y1_path)
    path1.set_3d_properties(z1_path)
    path2.set_data(x2_path, y2_path)
    path2.set_3d_properties(z2_path)
    
    com_dot.set_data([com_x[frame]], [com_y[frame]])
    com_dot.set_3d_properties([com_z[frame]])
    com_path.set_data(com_path_x, com_path_y)
    com_path.set_3d_properties(com_path_z)
    
    return body1, body2, path1, path2, com_dot, com_path

# Create the animation
dframe=5
ani = FuncAnimation(fig, update, frames= range(0, len(t_span),dframe), init_func=init, blit=True, interval=3)


plt.title('Two-body Problem in 3D with Center of Mass')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Phase plane plot for the two-body problem

# plt.plot(x1, sol.y[1, :], 'r-', label="Body 1: x vs vx", lw=3)  # Body 1 phase plot
# plt.plot(y1, sol.y[3, :], 'k--', label="Body 1: y vs vy", lw=3)
# plt.plot(x2, sol.y[7, :], 'b-', label="Body 2: x vs vx", lw=3)  # Body 2 phase plot
# plt.plot(y2, sol.y[9, :], 'g--', label="Body 2: y vs vy", lw=3)
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title("Phase Plane Plot (Two-Body Problem)")
plt.legend()
plt.grid()
plt.show()

# 3D Phase space plot combining all coordinates for Body 1 and Body 2
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

# Body 1 phase space projections
ax.plot(x1, y1, sol.y[1, :], 'r-', label="Mass 1", lw=3)
ax.plot(y1, z1, sol.y[3, :], 'g-', label="Center of Mass A", lw=3)
ax.plot(z1, x1, sol.y[5, :], 'b-', label="Mass 2", lw=3)

# Body 2 phase space projections
# ax.plot(x2, y2, sol.y[7, :], 'r--', label="Body 2: (x, y, vx)")
# ax.plot(y2, z2, sol.y[9, :], 'g--', label="Body 2: (y, z, vy)")
# ax.plot(z2, x2, sol.y[11, :], 'b--', label="Body 2: (z, x, vz)")

# Labels and title
ax.set_xlabel("Position (y)")
ax.set_ylabel("Position (x)")
ax.set_zlabel("Velocity (z)")
plt.title("Combined 3D Phase Space for Body 1 and Body 2")
plt.legend()
plt.show()

