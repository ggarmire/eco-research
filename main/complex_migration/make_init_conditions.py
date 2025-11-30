'''
This script makes the initial conditions: i.e. evolves the system until a state is reached with complex variations 
but no extinctions for enough time to calculate LEs, for several different initial conditions. 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

import sys 
sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic')

from funcs.lv_patches_functions import A_matrix_patches, D_matrix_patches, dxdt_multipatch, Jacobian_multipatch, x0_vector

def init_condition(s, M, Aseed, nruns, runtime, LEtime, A, B, D, Nc):
    '''
    s = number of species 
    M = number of patches 
    Aseed = random number generator for A 
    nruns = number of initial conditions to get back 
    '''
    x0seed = 0
    ngoodruns = 0
    x0seeds = np.empty(nruns, dtype=int)
    x0s = np.empty((s*M, nruns))

    while ngoodruns < nruns:
        x0 = x0_vector(s, M, x0seed)       # generate random x0 vector using run number 
        tspan = [0, runtime]
        print('initial condition seed: ', x0seed)

        # solve system for runtime:
        sol =  solve_ivp(dxdt_multipatch, tspan, x0, method='RK45', rtol=1e-9, atol=1e-9, args=(B,A,D,Nc))
        tsol = sol.t
        result = sol.y
        xf1 = result[:, -1]
        spec_left = np.sum(xf1>Nc)      # counts species left after runtime (times 8)

        LEtspan = [0, LEtime]
        # solve again for the next time: 
        sol =  solve_ivp(dxdt_multipatch, LEtspan, xf1, method='RK45', rtol=1e-9, atol=1e-9, args=(B,A,D,Nc))
        tsol = sol.t
        xf2 = result[:, -1]
        spec_left2 = np.sum(xf1>Nc)
        # verify no species died during LE run time:
        if spec_left2 == spec_left:
            # if no species died, add this to list for LE calculation
            x0seeds[ngoodruns] = x0seed
            x0s[:, ngoodruns] = xf1
            ngoodruns += 1
        x0seed += 1
    # return array of initial conditions for calculating LE spectrum 
    return x0s




    