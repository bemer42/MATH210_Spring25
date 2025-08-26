#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#parameters
G = 0.1
m1 = 1000
m2 = 800

def dXYZdt(t, Y):
    #define body 1 variables
    x1 = Y[0]
    vx1 = Y[1]
    y1 = Y[2]
    vy1 = Y[3]
    z1 = Y[4]
    vz1 = Y[5]

    #define body 2 variables
    x2 = Y[6]
    vx2 = Y[7]
    y2 = Y[8]
    vy2 = Y[9]
    z2 = Y[10]
    vz2 = Y[11]

    #define RHS functions for body 1
    dx1dt = vx1
    dvx1dt = -G*m2 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (x1-x2)
    dy1dt = vy1
    dvy1dt = -G*m2 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (y1-y2)
    dz1dt = vz1
    dvz1dt = -G*m2 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (z1-z2)

    #define RHS functions for body 2
    dx2dt = vx2
    dvx2dt = -G*m1 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (x2-x1)
    dy2dt = vy2
    dvy2dt = -G*m1 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (y2-y1)
    dz2dt = vz2
    dvz2dt = -G*m1 / ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(3/2) * (z2-z1)

    return [dx1dt, dvx1dt, dy1dt, dvy1dt, dz1dt, dvz1dt, dx2dt, dvx2dt, dy2dt, dvy2dt, dz2dt, dvz2dt]

#time discretization
N = 10000
t_0 = 0
t_end = 10
t_span = np.linspace(t_0, t_end, N)

#initial condition
x1_0 = 1
vx1_0 = -1.5
y1_0 = 1
vy1_0 = 1.5
z1_0 = 1
vz1_0 = -1
x2_0 = -1
vx2_0 = 1.5
y2_0 = -1
vy2_0 = -1.5
z2_0 = -1
vz2_0 = 2

Y_0 = [x1_0, vx1_0, y1_0, vy1_0, z1_0, vz1_0, x2_0, vx2_0, y2_0, vy2_0, z2_0, vz2_0]

#solve the initial value problems using solve_ivp
sol = solve_ivp(dXYZdt, [t_0, t_end], Y_0, t_eval = t_span, method = 'BDF')

#extract the time and solution from sol
t = sol.t
x1 = sol.y[0, :]
vx1 = sol.y[1, :]
y1 = sol.y[2, :]
vy1 = sol.y[3, :]
z1 = sol.y[4, :]
vz1 = sol.y[5, :]
x2 = sol.y[6, :]
vx2 = sol.y[7, :]
y2 = sol.y[8, :]
vy2 = sol.y[9, :]
z2 = sol.y[10, :]
vz2 = sol.y[11, :]

# #plot x, y, and z vs t for each body
# plt.figure(1)
# plt.plot(t, x1, 'b-', linewidth = 4, label = "X1 Displacement")
# plt.plot(t, y1, 'r-', linewidth = 4, label = "Y1 Displacement")
# plt.plot(t, z1, 'g-', linewidth = 4, label = "Z1 Displacement")
# plt.plot(t, x2, 'm-', linewidth = 4, label = "X2 Displacement")
# plt.plot(t, y2, 'y-', linewidth = 4, label = "Y2 Displacement")
# plt.plot(t, z2, 'k-', linewidth = 4, label = "Z2 Displacement")

# #customize plot appearance:
# plt.title("Two-Body Problem Displacements", fontsize = 20)
# plt.xlabel("t", fontsize = 15)
# plt.ylabel("x(t), y(t), z(t)", fontsize = 15)
# plt.gca().tick_params(labelsize = 10)
# plt.legend()
# plt.grid(True, which = "both")
# plt.minorticks_on()

# #show plot
# plt.show()

# #plot vx, vy, and vz vs t for each body
# plt.figure(2)
# plt.plot(t, vx1, 'b-', linewidth = 4, label = "X1 Velocity")
# plt.plot(t, vy1, 'r-', linewidth = 4, label = "Y1 Velocity")
# plt.plot(t, vz1, 'g-', linewidth = 4, label = "Z1 Velocity")
# plt.plot(t, vx2, 'm-', linewidth = 4, label = "X2 Velocity")
# plt.plot(t, vy2, 'y-', linewidth = 4, label = "Y2 Velocity")
# plt.plot(t, vz2, 'k-', linewidth = 4, label = "Z2 Velocity")

# #customize plot appearance:
# plt.title("Two-Body Problem Velocities", fontsize = 20)
# plt.xlabel("t", fontsize = 15)
# plt.ylabel("x(t), y(t), z(t)", fontsize = 15)
# plt.gca().tick_params(labelsize = 10)
# plt.legend()
# plt.grid(True, which = "both")
# plt.minorticks_on()

# #show plot
# plt.show()

#calculate the center of mass
com_x = (m1 * x1 + m2 * x2) / (m1 + m2)
com_y = (m1 * y1 + m2 * y2) / (m1 + m2)
com_z = (m1 * z1 + m2 * z2) / (m1 + m2)

#create a figure for the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

#plot the trajectories for both bodies
ax.plot(x1, y1, z1, 'b-', linewidth=4, label="Body 1 Trajectory")
ax.plot(x2, y2, z2, 'r-', linewidth=4, label="Body 2 Trajectory")

#plot initial positions for both bodies
body1_dot, = ax.plot([], [], [], 'mo', markersize=20, label = "Body 1")
body2_dot, = ax.plot([], [], [], 'yo', markersize=20, label = "Body 2")
com_dot, = ax.plot([], [], [], 'go', markersize=20, label = "Center of Mass")

#set plot limits
x_limits = [min(np.min(x1), np.min(x2)) - 1, max(np.max(x1), np.max(x2)) + 1]
y_limits = [min(np.min(y1), np.min(y2)) - 1, max(np.max(y1), np.max(y2)) + 1]
z_limits = [min(np.min(z1), np.min(z2)) - 1, max(np.max(z1), np.max(z2)) + 1]

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
ax.set_zlim(z_limits)

#customize plot appearance
ax.set_title("Two-Body Problem Phase Plane", fontsize=20)
ax.set_xlabel("x(t)", fontsize=15)
ax.set_ylabel("y(t)", fontsize=15)
ax.set_zlabel("z(t)", fontsize=15)
ax.legend()
ax.tick_params(labelsize=10)

#define the update function for animation
def update(frame):
    body1_dot.set_data(x1[frame], y1[frame])
    body1_dot.set_3d_properties(z1[frame])
    body2_dot.set_data(x2[frame], y2[frame])
    body2_dot.set_3d_properties(z2[frame])
    com_dot.set_data(com_x[frame], com_y[frame])
    com_dot.set_3d_properties(com_z[frame])
    return body1_dot, body2_dot, com_dot

#create the animation
dframe = 10
ani = FuncAnimation(fig, update, frames=range(0,len(t),dframe), interval=1, blit=True)

#show plot
plt.show()