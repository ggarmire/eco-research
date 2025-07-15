# this is for many runs without final population constraint. 




# libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import random 
from matplotlib.colors import LogNorm
import time

#region Set Variables 

# stuff that gets changed: 
n = 20
runs = 500

# A matrix 
random.seed(1)
C = 1

# M matrix 
muc = -0.8
mua = -0.5
f = 1.5
g = 1.2


# stuff that does not get changed:
s = n/2
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 500)
M = M_matrix(n, muc, mua, f, g)
One = np.ones(n)
nk = 13
Ks = np.linspace(1.1, 0.1, nk)
pct_stables = []
sigmas_A = []
sigma2s_A = []

for Kval in Ks:
    start = time.time()
    n_survives = []     # number of subspecies that survive 

    K_set = Kval
    sigma2 = K_set**2/n*2
    sigmas_A.append(sigma2**0.5)
    sigma2s_A.append(sigma2)
    # region loop 
    for run in range(runs):
        seed = run
        np.random.seed(seed)
        #if run %87 == 0:
        #    print(run)

        # make A matrix 
        A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
        A_rows = np.dot(A, One)

        # run ODE solver 
        result = lv_LH(x0, t, A, M)         # with scaled M
        xf = result[-1, :]

        n_survive = n
        for species in range(n):
            if xf[species] < 1e-3:
                n_survive -= 1
        n_survives.append(n_survive)
    
    pct_blk_stable = n_survives.count(n) / runs * 100
    pct_stables.append(pct_blk_stable)
    end = time.time()
    print('K:', K_set, ', pct stable:', pct_blk_stable, 'took ', end-start, ' seconds')




# region plot setup 
fsize = (6,6)

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+', $n=$'+str(n))


# region plotting 

# plot K vs percent of cases stable 
plt.figure(figsize=fsize)
plt.plot(Ks, pct_stables, '.')
plt.figtext(0.13, 0.12, mpar_text)
plt.ylabel('percent of runs with stability')
plt.xlabel('K value')
plt.title('percentage stable of '+str(runs)+'random cases changing K')
plt.grid()
#plt.ylim(0.06, 1.24)

# plot sigma vs percent of cases stable 
plt.figure(figsize=fsize)
plt.plot(sigmas_A, pct_stables, '.')
plt.figtext(0.13, 0.12, mpar_text)
plt.ylabel('percent of runs with stability')
plt.xlabel('sigma used to construct A')
plt.title('percentage stable of '+str(runs)+'random cases changing sigma')
plt.grid()





plt.show()



