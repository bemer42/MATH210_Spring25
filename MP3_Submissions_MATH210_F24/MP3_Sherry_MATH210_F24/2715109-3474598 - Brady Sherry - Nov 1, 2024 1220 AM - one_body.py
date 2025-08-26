#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

#parameters
G = 0.1
m2 = 1000

def dXYdt(t, Y):
    #define variables
    x = Y[0]
    vx = Y[1]
    y = Y[2]
    vy = Y[3]

    #define RHS functions
    dxdt = vx
    dvxdt = -G*m2 / (x**2 + y**2)**(3/2) * x
    dydt = vy
    dvydt = -G*m2 / (x**2 + y**2)**(3/2) * y

    return [dxdt, dvxdt, dydt, dvydt]

#time discretization
N = 10000
t_0 = 0
t_end = 100
t_span = np.linspace(t_0, t_end, N)

#initial condition
x_0 = 8
vx_0 = 2
y_0 = 10
vy_0 = -1
Y_0 = [x_0, vx_0, y_0, vy_0]

#solve the initial value problems using solve_ivp
sol = solve_ivp(dXYdt, [t_0, t_end], Y_0, t_eval = t_span, method = 'BDF')

#extract the time and solution from sol
t = sol.t
x = sol.y[0, :]
vx = sol.y[1, :]
y = sol.y[2, :]
vy = sol.y[3, :]

# #plot x and y vs t
# plt.figure(1)
# plt.plot(t, x, 'b-', linewidth = 4, label = "X Displacement")
# plt.plot(t, y, 'r-', linewidth = 4, label = "Y Displacement")

# #customize plot appearance:
# plt.title("One-Body Problem Displacements", fontsize = 20)
# plt.xlabel("t", fontsize = 15)
# plt.ylabel("x(t), y(t)", fontsize = 15)
# plt.gca().tick_params(labelsize = 10)
# plt.legend()
# plt.grid(True, which = "both")
# plt.minorticks_on()

# #show plot
# plt.show()

# #plot vx and vy vs t
# plt.figure(1)
# plt.plot(t, vx, 'b-', linewidth = 4, label = "X Velocity")
# plt.plot(t, vy, 'r-', linewidth = 4, label = "Y Velocity")

# #customize plot appearance:
# plt.title("One-Body Problem Velocities", fontsize = 20)
# plt.xlabel("t", fontsize = 15)
# plt.ylabel("x(t), y(t)", fontsize = 15)
# plt.gca().tick_params(labelsize = 10)
# plt.legend()
# plt.grid(True, which = "both")
# plt.minorticks_on()

# #show plot
# plt.show()

#plot x vs y
fig, ax = plt.subplots()
ax.plot(x, y, 'k-', linewidth=6, label = 'Trajectory')
ax.plot(0, 0, 'ro', markersize=20, label='Earth')
satellite_point, = ax.plot([], [], 'bo', markersize=20, label = 'Satellite')

#customize plot appearance
ax.set_title("One-Body Problem Phase Plane", fontsize=20)
ax.set_xlabel("x(t)", fontsize=15)
ax.set_ylabel("y(t)", fontsize=15)
ax.tick_params(labelsize=10)
ax.legend()
ax.grid(True, which="both")
ax.minorticks_on()
ax.set_xlim(np.min(x) - 1, np.max(x) + 1)
ax.set_ylim(np.min(y) - 1, np.max(y) + 1)

#define the update function for animation
def update(frame):
    satellite_point.set_data(x[frame], y[frame])
    return satellite_point,

#create the animation
dframe = 20
ani = FuncAnimation(fig, update, frames=range(0, len(t), dframe), interval=10, blit=True)

#show plot
plt.show()