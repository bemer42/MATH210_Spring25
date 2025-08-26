#############################################################################
# Program Title: One_body_2D
# Creation Date: 10/24/24
# Description: This program will model an objects attraction to another 
# object.
#
##### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
##### Functions
def dSdt(t, S):
    x = S[0]
    y = S[1]
    vx = S[2]
    vy = S[3]
    
    dxdt = vx
    dydt = vy
    dvxdt = x * -G*m2 / (x**2 + y**2)**(3/2)
    dvydt = y * -G*m2 / (x**2 + y**2)**(3/2)
    return [dxdt, dydt, dvxdt, dvydt]


##### Parameters
G = 0.1
m1 = 1
m2 = 1000

#Time span
N = int(1e3)
t0 = 0
tend = 100
tspan = np.linspace(t0, tend, N)

x0 = 10
y0 = 8
vx0 = 2
vy0 = -1
S0 = [x0, y0, vx0, vy0]






sol = sci.solve_ivp(dSdt, [t0, tend], S0, t_eval=tspan)
t = sol.t
x = sol.y[0, :]
y = sol.y[1, :]
vx = sol.y[2, :]
vy = sol.y[3, :]


for i in range(len(x)):
    plt.figure(1)
    plt.clf()
    plt.plot(0, 0, 'go', markersize=10, label='Earth')
    plt.plot(x[i], y[i], 'b.', markersize=8, label='Sattelite')
    plt.plot(x[:i], y[:i], 'k--', label='Flight Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Phase Plot')
    plt.legend()
    plt.pause(0.01)


plt.figure(2)
plt.plot(t, x, 'r-', label='Displacement in x-direction')
plt.plot(t, y, 'b-', label='Displacement in y-direction')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement v. Time')
plt.legend()

plt.figure(3)
plt.plot(t, vx, 'r-', label='Velocity in x-direction')
plt.plot(t, vy, 'b-', label='Velocity in y-direction')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity v. Time')
plt.legend()
plt.show()

#Last Updated: 


