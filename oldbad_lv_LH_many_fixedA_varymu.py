
#%% libraries 
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
import random 


# set problem up: number of species, complexity
n = 20
sigma2 = 1.8**2/n
C = 1     
K = (C*sigma2*n)**0.5
print("complexity: ", K)

# set initial values, A matrix, time 
np.random.seed(1)

x0 = np.random.normal(loc=1, scale=0.1, size=n)
for i in range(n):
    while x0[i] <= 0: 
        x0[i] = np.random.normal(loc=1, scale=0.1)


A = A_matrix(n, C, sigma2, seed=203, LH=1) 

t_end = 400     # length of time 
Nt = 1000
t = np.linspace(0, t_end, Nt)

# number of times to run ODE, and output vectors
runs = 100

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)

gs = np.zeros(runs)
fs = np.zeros(runs)
muas = np.zeros(runs)
mucs = np.zeros(runs)



for run in range(runs):
    seed = run
    np.random.seed(seed)

    g = 0.25
    f = 0.25
    muc = np.random.uniform(low=0, high = 1)
    mua = 0.25  #np.random.uniform(low=-1, high = 10)

    fs[run] = f
    gs[run] = g
    muas[run] = mua
    mucs[run] = muc

    #print('f:',f,'g:',g)
    M = M_matrix(n, muc, mua, f, g)

    result = lv_LH(x0, t, A, M)
    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-3:
            species_left+=1
        if abs(result[-1, i] - result [-2, i]) < 1e4:
            species_stable += 1
    
    n_species[run] = species_left
    n_stable[run] = species_stable
    
    Jac = LH_jacobian(result[-1, :], A, M)
    Jvals, Jvecs = np.linalg.eig(Jac)

    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))




# figures 

stables = np.ma.masked_less_equal(eigs_real_max, 0)
unstables = np.ma.masked_greater(eigs_real_max, 0)

plt.figure(figsize=(10,8))
'''ax = plt.axes(projection='3d')
ax.plot3D(mucs, muas, stables, 'ob', ms=2, alpha=.5)
ax.plot3D(mucs, muas, unstables, 'og', ms=2, alpha=.5)
ax.grid()
ax.set_title('Maximum Real Eigenvalue of J, for varied mu')
ax.set_xlabel('mu_c')
ax.set_ylabel('mu_a')
ax.set_zlabel('max real eigenvalue')'''

plt.plot(mucs, stables, 'ob', ms=2, alpha=.5)
plt.plot(mucs, unstables, 'og', ms=2, alpha=.5)
plt.grid()
plt.title('Maximum Real Eigenvalue of J, for varied mu')
plt.xlabel('mu_c')
plt.ylabel('max real eigenvalue')
plt.ylim(-5, 5)




plt.show()









    


