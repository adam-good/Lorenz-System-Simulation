import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def dx(t,x,y,z):
    return sigma*(y-x)

def dy(t,x,y,z):
    return x*(rho-z)-y

def dz(t,x,y,z):
    return x*y - beta*z

def euler_method(initial_state,functions,h,Nt):
    dx,dy,dz = functions
    output = np.zeros(shape=[Nt,3])
    output[0] = initial_state
    for t in range(Nt-1):
        x,y,z = output[t]
        next_x = x + h*dx(t,x,y,z)
        next_y = y + h*dy(t,x,y,z)
        next_z = z + h*dz(t,x,y,z)
        output[t+1] = (next_x,next_y,next_z)
    return output


Nt = 2000
t0 = 0
lt = 8*np.pi
t_space,dt = np.linspace(t0,lt,Nt,retstep=True)
h=dt
lorenz_system = euler_method(
    (1.0,1.0,1.0),
    (dx,dy,dz),
    h,
    Nt
)
x = lorenz_system[:,0]
y = lorenz_system[:,1]
z = lorenz_system[:,2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x,y,z)
plt.savefig('outputs/Lorenz_System.png')
plt.show()