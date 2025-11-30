# goal here is to replicate https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007827. 
# see functions as well! 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

import sys 
sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic')

from funcs.lv_patches_functions import A_matrix_patches, D_matrix_patches, dxdt_multipatch, Jacobian_multipatch, x0_vector
from funcs.gglyapunov import LE_lead, LE_spec
from make_init_conditions import init_condition

# region choose varibles 

# for system:
s = 20     # number of species 
M = 8       # number of patches
Aseed = 86      # seed to set A, for same interaction matrix each run 
nruns = 5      # number of initial conditions to try 

# for interaction matrices:
C = 1./8.
Amean = 0.3 # mean of A values 
Asigma = 0.45 # std of A values 
d = 1e-3  # dispersal rate
Nc = 1e-15  # 0 cutoff for species 
rho = 0.95  # correlation between A of different patches 

# for LE calculation:
runtime = 1500      # time to run initial conditions before doing LE calculation
LEcalctime= 200        # time to calculate LEs 
warmuptime = 50        # time to warm up ONS 

LEtime = LEcalctime + warmuptime
ds = 5     # ONS interval in LE Calculation 
nLEs = 50        # number of LEs to calculate 

# set matrices for system:
A = A_matrix_patches(s, M, C, Amean, Asigma, rho, seed=Aseed) # interaction matrix
B = np.full((s,M), 1)   # growth rate vector
D = D_matrix_patches(s, M, d)   # migration matrix

# region get initial conditions before LE calculation 
x0s = init_condition(s, M, Aseed, nruns, runtime, LEtime, A, B, D, Nc)

# region LE calculation: 
LEs = np.empty((nLEs, nruns))
for run in range(nruns):
    print('run # ', run)
    x0 = x0s[:, run]
    LEi = LE_spec(dxdt_multipatch, Jacobian_multipatch, x0, warmuptime, LEtime, ds, p=(B,A, D, Nc), nLE=nLEs)
    LEs[:, run] = LEi

LEmean = np.mean(LEs, axis = 1)
print(LEmean)

savetitle = str('LE_s'+str(s)+'M'+str(M)+'A'+str(Aseed)+'ds'+str(ds)+'t'+str(LEcalctime)+'nLE'+str(nLEs))

#np.save(savetitle, LEs)