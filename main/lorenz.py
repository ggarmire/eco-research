# region preamble
# practicing finding LEs on Lorenz attractor. 

import sys
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#sys.path.insert(1, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic/main/')

from funcs.gglyapunov import LE_spec

def lorenz(x, t, *p):
    # differential equations for the lorenz attractor. 
    sigma, rho, beta = p
    dx = [0,0,0]
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho-x[2])-x[1]
    dx[2] = x[0]*x[1]-beta*x[2]
    return dx 

def lorenz_jac(x, *p):
    sigma, rho, beta = p
    J = [[0, sigma, -sigma], 
         [rho-x[2], -1, -x[0]], 
         [x[1], x[0], -beta]]
    return J

# region parameters 
sigma = 10; rho = 28; beta = 8./3.
p = sigma, rho, beta

tsim = np.linspace(0, 300, 30000)
x0 = [1, 1, 1]
x02 = [1.01, 1.01, 1.01]

# region integration 
result = integrate.odeint(lorenz, x0, tsim, args=p)
x = result[:, 0]
y = result[:, 1]
z = result[:, 2]

# compute LEs
print(type(p), p)
LEs = LE_spec(lorenz, lorenz_jac, x0, 10, 100, 0.01, 0.1, p)
print(LEs)



# region plotting 
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot3D(x, y, z, lw = 0.5, alpha = 0.6)
ax.grid()

plt.show()


