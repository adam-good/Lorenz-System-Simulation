import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

Nt = 10000
t0 = 0
lt = 6*np.pi
t_space,h = np.linspace(t0,lt,Nt,retstep=True)

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

def dx(x,y,z):
    return sigma*(y-x)

def dy(x,y,z):
    return x*(rho-z)-y

def dz(x,y,z):
    return x*y - beta*z

def euler_method(initial_state, derivatives, h,Nt):
    dx,dy,dz = derivatives
    output = np.full(shape=[Nt,3],fill_value=np.nan)
    output[0] = initial_state

    for t in range(Nt-1):
        x,y,z = output[t]
        next_x = x + h*dx(x,y,z)
        next_y = y + h*dy(x,y,z)
        next_z = z + h*dz(x,y,z)
        output[t+1] = (next_x,next_y,next_z)

    return output

x,y,z = 1.,1.,1.
lorenz_system = euler_method(
    (x,y,z),
    (dx,dy,dz),
    h,
    Nt
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-25,25)
ax.set_ylim(0,50)

xx = lorenz_system[:,0]
zz = lorenz_system[:,2]
ax.plot(xx,zz)
line, = ax.plot([],[],'o')


num_frames = 1000
def init_plot():
    line.set_data(x,z)
    return line

def update_plot(i):
    global x,y,z
    offset = int(Nt/num_frames)
    x,y,z = lorenz_system[i*offset]
    line.set_data([x],[z])
    return line

anim = animation.FuncAnimation(fig,update_plot,frames=num_frames,interval=1,repeat=False)

GifWriter=animation.writers['imagemagick']
gifwriter = GifWriter(fps=15,bitrate=4096)
# anim.save("outputs/foooo.gif",writer=gifwriter)

plt.show()