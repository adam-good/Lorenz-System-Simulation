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

# initial_states = np.array([
#     [1.,1.,1.],
#     [2.,2.,2.],
#     [3.,3.,3.],
# ])
initial_states = np.array([[1+i/100,1.,1.] for i in range(300)])

output_states = np.array([
    euler_method(
        (x,y,z),
        (dx,dy,dz),
        h,
        Nt
    )
    for (x,y,z) in initial_states
])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-25,25)
ax.set_ylim(0,50)

line, = ax.plot([],[],'o')

num_frames = 1000
def init_plot():
    line.set_data(x,z)
    return line


x = initial_states[:,0]
y = initial_states[:,1]
z = initial_states[:,2]
def update_plot(i):
    global x,y,z
    offset = int(Nt/num_frames)
    x = output_states[:,i*offset,0]
    y = output_states[:,i*offset,1]
    z = output_states[:,i*offset,2]
    line.set_data(x,z)
    return line,

anim = animation.FuncAnimation(fig,update_plot,init_func=init_plot,frames=num_frames,interval=1,repeat=False)

VidWriter=animation.writers['ffmpeg']
vidwriter = VidWriter(fps=15,bitrate=4096)
anim.save("outputs/Lorenz_Particle_System.mp4",writer=vidwriter)

# GifWriter=animation.writers['imagemagick']
# gifwriter = GifWriter(fps=30,bitrate=4096)
# anim.save("outputs/Lorenz_System_Particle.gif",writer=gifwriter)

plt.show()