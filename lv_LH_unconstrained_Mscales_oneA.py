
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
K_set = 0.6
C = 1

muc = -0.5
mua = -0.5
f = 1.5
g = 1

seed = 1


# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5
M_pre = M_matrix(n, muc, mua, f, g)

x0 = x0_vec(n, 1)
t = np.linspace(0, 200, 500)

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")


rs = np.linspace(0, 1, 21)
m_runs = 100
    
A = A_matrix(n, C, sigma2, seed, LH=1)
A_rs = np.dot(A, np.ones(n))
print('A seed:', seed, ', max row sum:', np.max(A_rs))


# loop through values of r
fracs = []
for r_int in range(len(rs)): 
    if r_int % 5 == 0: print(r_int)
    r = rs[r_int]
    # for each r, run a bunch of random scale vectors, calculate percent stable
    stables = []
    for run in range(m_runs):
        np.random.seed(run)
        scales = np.random.uniform(1-r, 1+r, s)
        M = M_matrix(n, muc, mua, f, g)
        for j in range(s):
            M[2*j,:] = scales[j]*M[2*j,:]
            M[2*j+1,:] = scales[j]*M[2*j+1,:]
        result = lv_LH(x0, t, A, M)
        min_pop = min(result[-1, :])
        if min_pop < 1e-3: stable = 0
        else: stable = 1
        stables.append(stable)
    frac = np.average(stables)
    fracs.append(frac)



# plotting 
fsize = (6,6)

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', A seed ='+str(seed)+ ', K='+str('%.3f'%K))
plot_text2 = str('max row sum in A:'+str('%.3f'%np.max(A_rs)))
box_par = dict(boxstyle='square', facecolor='white', alpha = 0.5)

plt.figure(figsize=fsize)
plt.plot(rs, fracs, '.')
plt.grid()
plt.xlabel('r value')
plt.ylabel('percent of random scale vectors that are stable ')
plt.title('fraction of stable cases for given r, for set A matrix')
plt.figtext(0.13, 0.12, plot_text)
plt.figtext(0.55, 0.84, plot_text2, bbox=box_par)

plt.show()










