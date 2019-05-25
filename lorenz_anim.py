import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

Nt = 10000
t0 = 0
lt = 6*np.pi
t_space,dt = np.linspace(t0,lt,Nt,retstep=True)
h=dt

init_state = (1.0,1.0,1.0)
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
    output = np.full(shape=[Nt,3],fill_value=np.nan)
    output[0] = initial_state

    for t in range(Nt-1):
        x,y,z = output[t]
        next_x = x + h*dx(t,x,y,z)
        next_y = y + h*dy(t,x,y,z)
        next_z = z + h*dz(t,x,y,z)
        output[t+1] = (next_x,next_y,next_z)

    return output

def calc_dist(a,b,Nt):
    return [np.linalg.norm(a[:i]-b[:i]) for i in range(Nt)]
        

lorenz_system1 = euler_method(
    init_state,
    (dx,dy,dz),
    h,
    Nt
)
x1 = lorenz_system1[:,0]
y1 = lorenz_system1[:,1]
z1 = lorenz_system1[:,2]

sigma += 0.0001
lorenz_system2 = euler_method(
    init_state,
    (dx,dy,dz),
    h,
    Nt
)
x2 = lorenz_system2[:,0]
y2 = lorenz_system2[:,1]
z2 = lorenz_system2[:,2]

dist = calc_dist(lorenz_system1, lorenz_system2,Nt)

fig = plt.figure()
ax1 = fig.add_subplot(221,projection='3d')
ax2 = fig.add_subplot(222,projection='3d')
ax3 = fig.add_subplot(223)

line1, = ax1.plot([],[],[])
ax1.set_xlim(-20,20)
ax1.set_ylim(-20,30)
ax1.set_zlim(0,50)
ax1.set_title(fr"Lorenz System 1")
ax1.set_xlabel(r"x")
ax1.set_ylabel(r"y")
ax1.set_zlabel(r"z")

line2, = ax2.plot([],[],[])
ax2.set_xlim(-20,20)
ax2.set_ylim(-20,30)
ax2.set_zlim(0,50)
ax2.set_title(r"Lorenz System 2")
ax2.set_xlabel(r"x")
ax2.set_ylabel(r"y")
ax2.set_zlabel(r"z")

line3, = ax3.plot([],[])
ax3.set_xlim(t0,lt)
ax3.set_ylim(min(dist),max(dist))
ax3.set_title(r"Difference $L_2$ norm")
ax3.set_xlabel(r"Time")
ax3.set_ylabel(r"Difference")

plt.subplots_adjust(bottom=0.1,top=0.9)
plt.tight_layout()

num_frames = 100
def update1(i):
    offset=int(Nt/num_frames)
    x = lorenz_system1[:i*offset,0]
    y = lorenz_system1[:i*offset,1]
    z = lorenz_system1[:i*offset,2]
    line1.set_data(x,y)
    line1.set_3d_properties(z)
    return line1

def update2(i):
    offset=int(Nt/num_frames)
    x = lorenz_system2[:i*offset,0]
    y = lorenz_system2[:i*offset,1]
    z = lorenz_system2[:i*offset,2]
    line2.set_data(x,y)
    line2.set_3d_properties(z)
    return line2

def update3(i):
    offset=int(Nt/num_frames)
    y = dist[:i*offset]
    x = t_space[:i*offset]
    line3.set_data(x,y)
    return line3,

def update(i):
    l1 = update1(i)
    l2 = update2(i)
    l3 = update3(i)
    return (l1,l2,l3)

anim = animation.FuncAnimation(fig,update,frames=num_frames,interval=1,repeat=False)

VidWriter=animation.writers['ffmpeg']
GifWriter=animation.writers['imagemagick']
vidwriter = VidWriter(fps=15,bitrate=4096)
gifwriter = GifWriter(fps=15,bitrate=4096)
anim.save("outputs/Lorenz_System.mp4",writer=vidwriter)
anim.save("outputs/Lorenz_System.gif",writer=gifwriter)

plt.show()