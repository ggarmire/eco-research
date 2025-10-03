#region preamble
print('\n')
# replicating this paper: https://sprott.physics.wisc.edu/pubs/paper288.pdf to understand
# better how chaos appears in GLV systems. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import math, sys
from scipy import integrate
from matplotlib.animation import FuncAnimation

#sys.path.insert(1, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic/functions')


seed = np.random.randint(0, 1000)
#seed = 171
print('seed = ', seed)
np.random.seed(seed)



# endregion

#region functions 
def derivative(x, t, r, A):
        n = len(x0)
        rx = np.multiply(r, x)
        dxdt = np.multiply(rx, (np.ones(n) - np.dot(A, x)))
        for i in range(0, len(x0)):
            if x[i] <= 0:
                dxdt[i] == 0
        return dxdt

def lv_classic(x0, t, r, A): 
    result = integrate.odeint(derivative, x0, t, args = (r, A))
    return result

def Jac(x, t, r, A):
    Ax = np.dot(A, x)
    xA = np.dot(np.diag(x), A)
    Jac = np.diag(r) - np.diag(np.multiply(r, Ax)) - np.dot((np.diag(r), Ax))
    return Jac

#
#endregion




# region example values 
t = np.linspace(0, 1000, 100000)
rex = [1, 0.72, 1.53, 1.27]
Aex = [[1, 1.09, 1.52, 0], [0, 1, 0.44, 1.36], [2.33, 0, 1, 0.47], [1.21, 0.51, 0.35, 1]]
#x0 = np.random.uniform(0.2, 0.8, 4)

x0 = [0.3013, 0.4586, 0.1307, 0.3557]

# region integrate 
result = lv_classic(x0, t, rex, Aex)
# endregion 
print('min population value: ', np.min(result))
print('avg pops:', np.mean(result[-1000:-1, 0]), np.mean(result[-1000:-1, 1]), np.mean(result[-1000:-1, 2]), np.mean(result[-1000:-1, 3]))

#region analysis 
tsimend = 10
dt = 0.1
tsim = np.arange(0, tsimend+dt, dt)

def LE_spec(f, x0, tsim, r, A):
    n = len(x0)
    gamma = np.zeros(n)
    Q = np.identity(n)
    result = integrate.odeint(derivative, x0, tsim, args = (r, A))
    for i in range(len(tsim)):
        xt = result[i,:]
        jact = np.dot(np.diag(xt), A)
        Qt = np.dot(jact, Q)
        Q, R = np.linalg.qr(Qt)
        for k in range(n):
            gamma[k] += R[k][k]
    LE = np.zeros(n)
    for j in range(n):
        LE[j] = gamma[j]/tsim[-1]
    return LE

LE = LE_spec(derivative, x0, tsim, rex, Aex)
print(LE)


# region LEs with package









# endregion

# region plotting 
plt.figure()
for i in range(4):
    plt.plot(t, result[:,i], '-')
plt.grid()


x,y,z,c = result.T

# Build line segments
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# Line collection with colors from 4th column
lc = Line3DCollection(segments, cmap='viridis', norm=plt.Normalize(c.min(), c.max()))
lc.set_array(c)
lc.set_linewidth(1)
ax.add_collection3d(lc)
# Set limits so plot scales correctly
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())
# Add colorbar
cb = fig.colorbar(lc, ax=ax)
cb.set_label("4th column")


# animated
'''# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initialize empty Line3DCollection
lc = Line3DCollection([], cmap='viridis', norm=plt.Normalize(c.min(), c.max()))
lc.set_linewidth(2)
ax.add_collection(lc)

# Set limits
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(z.min(), z.max())

# Add colorbar
cb = fig.colorbar(lc, ax=ax)
cb.set_label("4th column")

# Update function
def update(frame):
    segs = segments[:frame]        # up to current frame
    lc.set_segments(segs)
    lc.set_array(c[:frame])        # color values up to frame
    return lc,

# Animate
ani = FuncAnimation(fig, update, frames=len(segments), interval=0.001, blit=False)

'''



plt.show()

# endregion 