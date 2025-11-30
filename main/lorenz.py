# region preamble
# practicing finding LEs on Lorenz attractor. 

import sys
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#sys.path.insert(1, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic/main/')

from funcs.gglyapunov import LE_spec

'''def lorenz(x, t, *p):
    # differential equations for the lorenz attractor. 
    sigma, rho, beta = p
    dx = [0,0,0]
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho-x[2])-x[1]
    dx[2] = x[0]*x[1]-beta*x[2]
    return dx '''

def lorenz(t, x, sigma, rho, beta):
    # differential equations for the lorenz attractor. 
    dx = np.empty(3)
    dx[0] = sigma*(x[1]-x[0])
    dx[1] = x[0]*(rho-x[2])-x[1]
    dx[2] = x[0]*x[1]-beta*x[2]
    return dx

def lorenz_jac(t, x, sigma, rho, beta):
    J = [[-sigma, sigma, 0], 
         [rho-x[2], -1, -x[0]], 
         [x[1], x[0], -beta]]
    return J

# region parameters 
sigma = 10; rho = 28; beta = 8./3.
#x0 = [1, 1, 1]
x0 = [0.9, 0.9, 0.9]

# compute LEs

LEs, t, result = LE_spec(lorenz, lorenz_jac, x0, 200, 20, 0.1, p=(sigma, rho, beta), result=True)

print ('LES:', LEs)

# region integration 

# region plotting 
# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot3D(result[0,:], result[1,:], result[2,:])
ax.grid()

plt.show()



