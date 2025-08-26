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
import matplotlib.animation as animation
##### Functions
def dSdt(t, S):
    x = S[0]
    y = S[1]
    z = S[2]
    vx = S[3]
    vy = S[4]
    vz = S[5]
    
    x2 = S[6]
    y2 = S[7]
    z2 = S[8]
    vx2 = S[9]
    vy2 = S[10]
    vz2 = S[11]
    
    dx1dt = vx
    dy1dt = vy
    dz1dt = vz
    dvx1dt = (x-x2) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    dvy1dt = (y-y2) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    dvz1dt = (z-z2) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    
    dx2dt = vx2
    dy2dt = vy2
    dz2dt = vz2
    dvx2dt = (x2-x) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    dvy2dt = (y2-y) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    dvz2dt = (z2-z) * -G*m2 / ((x-x2)**2 + (y-y2)**2 + (z-z2)**2)**(3/2)
    return [dx1dt, dy1dt, dz1dt, dvx1dt, dvy1dt, dvz1dt, dx2dt, dy2dt, dz2dt, dvx2dt, dvy2dt, dvz2dt]

def mass_center(x, y, z, x2, y2, z2):
    distance = m2*np.sqrt((x2-x)**2 + (y2-y)**2 + (z2-z)**2) / (m1+m2)
    scale = distance/np.sqrt((x2-x)**2 + (y2-y)**2 + (z2-z)**2)
    x_pos = x + (x2-x)*scale
    y_pos = y + (y2-y)*scale
    z_pos = z + (z2-z)*scale
    return x_pos, y_pos, z_pos

def mass_scale(m1, m2):
    scale = m2/m1
    m1_marker = 10
    m2_marker = m1_marker*scale
    return [m1_marker, m2_marker]

def update(i):
    ax.cla()
    
    #Mass bodies
    ax.plot(x1[i], y1[i], z1[i], 'b.', markersize=mass_scale(m1, m2)[0], label='Object A')
    ax.plot(x2[i], y2[i], z2[i], 'r.', markersize=mass_scale(m1, m2)[1], label='Object B')
    ax.plot(x_center[i], y_center[i], z_center[i], 'yo', label='Center of Mass')

    #Mass tails
    if i < 30:
        ax.plot(x1[:i], y1[:i], z1[:i], 'k:', linewidth=2)
        ax.plot(x2[:i], y2[:i], z2[:i], 'k:', linewidth=2, label='Object Path')
        ax.plot(x_center[:i], y_center[:i], z_center[:i], 'k:', label='Center of Mass')
    else:
        ax.plot(x1[i-30:i], y1[i-30:i], z1[i-30:i], 'k:', linewidth=2)
        ax.plot(x2[i-30:i], y2[i-30:i], z2[i-30:i], 'k:', linewidth=2)
        ax.plot(x_center[i-30:i], y_center[i-30:i], z_center[i-30:i], 'k:')

    #Plot parameters
    plt.title('3D Two Body')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc='upper right')
    #Auto rotate
    #ax.view_init(i, 30)
    
    return ax,
##### Parameters
G = 0.1
m1 = 1000
m2 = 4000

#Time span
N = int(2e3)
t0 = 0
tend = 1000
tspan = np.linspace(t0, tend, N)

#Set initial conditions
x0 = -10
y0 = -5
z0 = 0
vx0 = 1
vy0 = 2
vz0 = 0

x20 = 20
y20 = 10
z20 = 0
vx20 = -1
vy20 = -1
vz20 = 0
S0 = [x0, y0, z0, vx0, vy0, vz0, x20, y20, z20, vx20, vy20, vz20]

#Solve IVP
sol = sci.solve_ivp(dSdt, [t0, tend], S0, t_eval=tspan)
t = sol.t
x1 = sol.y[0, :]
y1 = sol.y[1, :]
z1 = sol.y[2, :]
vx = sol.y[3, :]
vy = sol.y[4, :]
vz = sol.y[5, :]

x2 = sol.y[6, :]
y2 = sol.y[7, :]
z2 = sol.y[8, :]
vx2 = sol.y[9, :]
vy2 = sol.y[10, :]
vz2 = sol.y[11, :]

#Calculate location of center of mass
x_center, y_center, z_center = mass_center(x1, y1, z1, x2, y2, z2)


#Initialize 3d plot and animate data
fig = plt.figure(1)
ax = plt.figure(1).add_subplot(projection='3d')
ani = animation.FuncAnimation(fig, update, frames=range(0, tend, 5), blit=False)

writer = animation.PillowWriter(fps=15, 
                                metadata=dict(artist='Me'),
                                bitrate=1800)
#ani.save('/home/ccrites/Documents/Mathematical Typesetting/Project 3/Figures/2_body.gif', writer=writer)
plt.show()


#Last Updated: 

#Goals 
#Movie plot of 1 body in 2D +
#Movie plot of 2 bodies in 3D +
#Plot center of mass for 2 body problem +

