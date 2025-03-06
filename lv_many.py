
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import integrate
import random 
import math
from scipy.optimize import curve_fit

from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH

# set number of times you solve:
runs = 1000


# set constant things: A, t, x0, n
n = 10



t = np.linspace(0, 50, 1000)
sigma2 = 0.5
C = 0.1 

A = A_matrix(n, C, sigma2, seed=1, LH=1)

for r in A:
    print(r)

x0 = 0.5*np.ones(n)    

# make empty matrices
xfs = np.zeros((n, runs))
mucs = np.zeros(runs)
muas = np.zeros(runs)
fs = np.zeros(runs)
gs = np.zeros(runs)

zs = np.zeros((int(n/2), runs))
n_stable = np.zeros(runs)
n_species = np.zeros(runs)

# run function 


for i in range(runs):
    seed = i
    muc = 0.5
    mua = 0.5
    f = np.random.uniform(low=0.1, high = 0.9)
    g = np.random.uniform(low=0.1, high = 0.9)
    
    #M = M_matrix(n, muc, mua, f, g)
    M = M_matrix(n, f, g, mua, muc)
    result = lv_LH(x0, t, A, M)

    xfs[:, i] = result[-1, :]
    mucs[i] = muc
    muas[i] = mua
    fs[i] = f
    gs[i] = g

    species_left = 0
    species_stable = 0

    for j in range(n):
        if result[-1, j] > 1e-3:
            species_left +=1
        if abs((result[-1, j]-result[-2, j]) / result[-1, j]) < 1e-3:
            species_stable +=1
        if j%2 == 0:
            zs[int(j/2), i]= (result[-1,j]/(result[-1,j]+result[-1,j+1]))

    n_species[i] = species_left
    n_stable[i] = species_stable


    '''print("xfs: ",  xfs[:,i])
    print("species_left:", n_species[i])
    print("species_stable:", n_stable[i])'''

        
## fitts
pars_f_xfs = np.zeros((2,n))
pars_g_xfs = np.zeros((2,n))
covs_f_xfs = np.zeros(n)
covs_g_xfs = np.zeros(n)

def linear(x, m, b): 
    y = m*x+b
    return y


for i in range(n):
    k, cov = curve_fit(linear, fs, xfs[i])
    pars_f_xfs[0,i] = k[0]; pars_f_xfs[1,i] = k[1]
    l, cov = curve_fit(linear, gs, xfs[i])
    pars_g_xfs[0,i] = l[0]; pars_g_xfs[1,i] = l[1]

for i in range(n):
    print('xf', i, '=', pars_f_xfs[0, i], 'f +', pars_f_xfs[1,i])
    print('xf', i, '=', pars_g_xfs[0, i], 'g +', pars_g_xfs[1,i])


## plotting 

# f,g vs remaining species
plt.figure()
plt.grid()
plt.title("f, g vs. species left")
plt.plot(fs, n_species, '.b', label = 'mua')
plt.plot(gs, n_species, '.g', label = 'muc')
plt.xlabel('f, g')
plt.ylabel('species remaining')

#f, g vs stable species
plt.figure()
plt.grid()
plt.title("muc, mua vs. species stable")
plt.plot(fs, n_stable, '.b', label = 'mua')
plt.plot(gs, n_stable, '.g', label = 'muc')
plt.xlabel('f, g')
plt.ylabel('species stable')


# f, g vs juvinile fractions
plt.figure()
plt.grid()
plt.title("muc, mua vs. juvinile fraction")
#print(int(n/2))
plt.plot(gs, zs[0, :], '.m', alpha=0.4, label = '1, muc')
plt.plot(fs, zs[0, :], '.r', alpha=0.4, label = '1, mua')
plt.plot(gs, zs[1, :], '.c', alpha=0.4, label = '2, muc')
plt.plot(fs, zs[1, :], '.b', alpha=0.4, label = '2, mua')
plt.plot(gs, zs[2, :], '.g', alpha=0.4, label = '3, muc')
plt.plot(fs, zs[2, :], '.y', alpha=0.4, label = '3, mua')
plt.xlabel('f, g')
plt.ylabel('final juvinile fraction')
plt.legend()




# f, g vs final pops
plt.figure()
plt.grid()
plt.title("muc, mua vs. final population")
for i in range(n):
    if i%2 ==0:
        plt.plot(fs, xfs[i, :], '.b',  mfc = 'none')
        plt.plot(gs, xfs[i, :], '.g', mfc = 'none')
    else:
        plt.plot(fs, xfs[i, :], '.b')
        plt.plot(gs, xfs[i, :], '.g')


# f, g vs species 1, 2 final pops
plt.figure()
plt.grid()
plt.title("muc, mua vs. final population of only species 1, 2")
plt.plot(gs, xfs[0, :], '.m', alpha=0.4, label = '1c, mua')
plt.plot(gs, xfs[1, :], '.r', alpha=0.4, label = '1a, mua')
plt.plot(gs, xfs[2, :], '.c', alpha=0.4, label = '2c, mua')
plt.plot(gs, xfs[3, :], '.b', alpha=0.4, label = '2a, mua')
plt.plot(fs, xfs[0, :], '.m', mfc = 'none', alpha=0.4, label = '1c, muc')
plt.plot(fs, xfs[1, :], '.r', mfc = 'none', alpha=0.4, label = '1a, muc')
plt.plot(fs, xfs[2, :], '.c', mfc = 'none', alpha=0.4, label = '2c, muc')
plt.plot(fs, xfs[3, :], '.b', mfc = 'none', alpha=0.4, label = '2a, muc')



plt.plot(fs, linear(fs, pars_f_xfs[0, 0], pars_f_xfs[1, 0]), '-m')
plt.plot(gs, linear(gs, pars_g_xfs[0, 0], pars_g_xfs[1, 0]), '-m')
plt.plot(fs, linear(fs, pars_f_xfs[0, 1], pars_f_xfs[1, 1]), '-r')
plt.plot(gs, linear(gs, pars_g_xfs[0, 1], pars_g_xfs[1, 1]), '-r')
plt.plot(fs, linear(fs, pars_f_xfs[0, 2], pars_f_xfs[1, 2]), '-c')
plt.plot(gs, linear(gs, pars_g_xfs[0, 2], pars_g_xfs[1, 2]), '-c')
plt.plot(fs, linear(fs, pars_f_xfs[0, 3], pars_f_xfs[1, 3]), '-b')
plt.plot(gs, linear(gs, pars_g_xfs[0, 3], pars_g_xfs[1, 3]), '-b')

plt.legend(loc='best', fontsize = 8, ncol=2)

plt.xlabel('f, g')
plt.ylabel('final population density')




plt.show()





    


