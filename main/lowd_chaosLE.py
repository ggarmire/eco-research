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
from funcs.pyLyapunov import computeLE


#sys.path.insert(1, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic/functions')


seed = np.random.randint(0, 1000)
#seed = 171
print('seed = ', seed)
np.random.seed(seed)



# endregion

#region functions 
def derivative(t, x, p):
    r = p[0]; A = p[1]
    n = len(r)
    rx = np.multiply(r, x)
    dxdt = np.multiply(rx, (np.ones(n) - np.dot(A, x)))
    for i in range(0, len(x0)):
        if x[i] <= 0:
            dxdt[i] == 0
    return dxdt

def lv_Jac(x, t, p):
    r = p[0], A = p[1]
    Ax = np.dot(A, x)
    xA = np.dot(np.diag(x), A)
    Jac = np.diag(r) - np.diag(np.multiply(r, Ax)) - np.dot((np.diag(r), Ax))
    return Jac



#endregion


ttrans = np.arange(0.0, 100.0, 0.001)
#ttrans = None
t = np.arange(0.0, 100.0, 0.001)


# region example values 
t = np.linspace(0, 1000, 10000)
rex = [1, 0.72, 1.53, 1.27]
Aex = [[1, 1.09, 1.52, 0], [0, 1, 0.44, 1.36], [2.33, 0, 1, 0.47], [1.21, 0.51, 0.35, 1]]
pex = [rex, Aex]
x0 = np.random.uniform(0.2, 0.8, 4)
# region integrate 
# endregion 

#region analysis 

LE = computeLE(derivative, lv_Jac, x0, t, pex)
print(LE[-1])


# endregion

