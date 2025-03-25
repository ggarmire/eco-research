
#%% libraries 
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 


# set problem up: number of species, complexity
n = 20

C = 1    # connectedness|
sigma2 = .9**2/n;       ## variance in off diagonals of interaction matrix
K = (C*sigma2*n)**0.5
print("complexity: ", K)

# set initial values, A matrix, time 
np.random.seed(1)
x0 = 0.5 * np.ones(n) 
for i in range(n):
    x0[i] += np.random.normal(loc=0, scale=0.05)

A = A_matrix(n, C, sigma2, seed=137, LH=1) 
A_rowsums = np.dot(A, np.ones(n))

t_end = 30     # length of time 
Nt = 1000
t = np.linspace(0, t_end, Nt)

# number of times to run ODE, and output vectors
runs = 1000

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

    g = np.random.uniform(low=-.5, high = 10)
    f = np.random.uniform(low=-.5, high = 10)
    muc = 0.5
    mua = 0.5

    fs[run] = f
    gs[run] = g
    muas[run] = mua
    mucs[run] = muc

    #print('f:',f,'g:',g)
    M = M_matrix(n, muc, mua, f, g)
    M_rowsums = np.dot(M, np.ones(n))

    for i in range(n):
        for j in range(n):
            if i%2 ==0:
                M[i][j] = M[i][j]*A_rowsums[i]/M_rowsums[i]
            if i%2 ==1:
                M[i][j] = M[i][j]*A_rowsums[i]/M_rowsums[i]
        
    result = lv_LH(x0, t, M, A)
    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-3:
            species_left+=1
        if abs(result[-1, i] - result [-2, i]) < 1e4:
            species_stable += 1
    
    n_species[run] = species_left
    n_stable[run] = species_stable

    delt = np.dot(A, result[-1, :])
    
    Jac = M + A
    for i in range(n):
        Jac[i][i] += A_rowsums[i]
    Jvals, Jvecs = np.linalg.eig(Jac)

    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))




# figures 

stables = np.ma.masked_less_equal(eigs_real_max, 0)
unstables = np.ma.masked_greater(eigs_real_max, 0)

plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
ax.plot3D(fs, gs, stables, 'ob', ms=2, alpha=.5)
ax.plot3D(fs, gs, unstables, 'og', ms=2, alpha=.5)
ax.grid()
ax.set_title('Maximum Real Eigenvalue of J, for f and g')
ax.set_xlabel('f')
ax.set_ylabel('g')
ax.set_zlabel('max real eigenvalue')



plt.show()









    


