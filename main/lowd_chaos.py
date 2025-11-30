#region preamble
print('\n')
# replicating this paper: https://sprott.physics.wisc.edu/pubs/paper288.pdf to understand
# better how chaos appears in GLV systems. 

import numpy as np
import matplotlib.pyplot as plt
import math, sys
from scipy.integrate import solve_ivp

sys.path.insert(1, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic/funcs')
from funcs.gglyapunov import LE_spec, LE_lead


seed = np.random.randint(0, 1000)
#seed = 171
print('seed = ', seed)
np.random.seed(seed)

# endregion

#region functions 
def derivative(t, x, r, A):
        rx = np.multiply(r, x)
        dxdt = r * x * (1 - A.dot(x))
        dxdt = np.where(x <= 0, 0.0, dxdt)
        return dxdt

def Jac(t, x, r, A):
    n = len(x)
    J = -(r*x)[:, None]*A
    diag = r*(1-A.dot(x))
    J[np.arange(n), np.arange(n)] += diag
    return J
#
#endregion
# region example values 

rex = np.array([1, 0.72, 1.53, 1.27])
Aex = np.array([[1, 1.09, 1.52, 0], 
       [0, 1, 0.44, 1.36], 
       [2.33, 0, 1, 0.47], 
       [1.21, 0.51, 0.35, 1]])

#x0 = np.random.uniform(0.2, 0.8, 4)

x0 = np.array([0.3013, 0.4586, 0.1307, 0.3557])

# result with numerical integration

# region LEs with package
tend = 1500
twarm =1000

# LEs averaged over many runs 


LEs, t, result = LE_spec(derivative, Jac, x0, tend, twarm, ds=2, p=(rex, Aex), result=True)
print('LEs from 1 run:\n', LEs)
print('LE sum:', np.sum(LEs))
runs = 1
LE1sum = 0
LEssum = np.zeros(2)
for run in range(runs): 
    print('run', run)
    x0i = np.random.uniform(0.2, 0.8, 4)
    #LEi = LE_lead(derivative, Jac, x0i, tend,  twarm, ds=0.5, p=(rex, Aex))
    LEs  = LE_spec(derivative, Jac, x0i, tend, twarm, ds=2, p=(rex, Aex), nLE = 2)
    print('LEs:', LEs)
    #LE1sum += LEi
    LEssum += LEs

#LE1 = LE1sum/ runs
LE = LEssum/runs

#print('L1E1:', LE1)
print('LE:', LE)



sol = solve_ivp(derivative, [0, tend], x0, method='RK45', rtol=1e-9, atol=1e-9, args = (rex, Aex))
tsol = sol.t
ressol = sol.y

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, result[0,:])
ax.plot(t, result[1,:])
ax.plot(t, result[2,:])
ax.plot(t, result[3,:])
ax.grid()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tsol, ressol[0,:])
ax.plot(tsol, ressol[1,:])
ax.plot(tsol, ressol[2,:])
ax.plot(tsol, ressol[3,:])
ax.grid()


plt.show()

# endregion

'''# region plotting 
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

'''
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





# endregion 