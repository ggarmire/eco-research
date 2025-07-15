
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
from lv_functions import x0_vec
import random 
import math


#region initial conditions c

# values to set 
n = 20     # number of species 
s = int(n / 2)
K_set = 0.7
C = 1

muc = -0.5
mua = -0.5
f = 1.5
g = 1
vs = np.linspace(0.1, 2, 31)     # scaling factor in m



# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5
M_pre = M_matrix(n, muc, mua, f, g)
One = np.ones(n)

x0 = x0_vec(n, 1)
t = np.linspace(0, 200, 500)

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")

runs = 100

stable_avg = []
n_survive_avg = []


for v in vs: 
    print('on v =', v)
    n_survives = []
    stable = []
    for run in range(runs):
        seed = run
        np.random.seed(seed)
        #if run %87 == 0:
        #    print(run)

        # make A matrix 
        A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
        A_rows = np.dot(A, One)
        A_rows_sign = np.sign(A_rows)
        #Ars_max.append(np.max(A_rows))

        # scale M matrix for something what the hell

        M_rows = np.dot(M_pre, One)
        #scales = -v * np.divide(A_rows, M_rows)
        scales = -v * A_rows
        M = np.multiply(M_pre, np.outer(scales, np.ones(n)))  
        
        # calculate jacobian given final values 

        # run ODE solver 
        result = lv_LH(x0, t, A, M)         # with scaled M
        xf = result[-1, :]

        n_survive = n

        for species in range(n):
            if xf[species] < 1e-3:
                n_survive -= 1
        n_survives.append(n_survive)

        if n_survive == n: stable.append(1)
        elif n_survive != n: stable.append(0)

    stable_avg.append(np.mean(stable))
    n_survive_avg.append(np.mean(n_survive))

    
    

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', K='+str(K_set))




# plotting 
fsize = (6,6)
plt.plot(vs, stable_avg, '.')
plt.xlabel('scaling factor')
plt.ylabel('fraction of cases stable')
plt.title('fraction of stable cases for given v, for different As')
plt.grid()
plt.figtext(0.13, 0.12, plot_text)


plt.show()










