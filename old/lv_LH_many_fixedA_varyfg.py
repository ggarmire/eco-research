
#%% libraries 
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import x0_vec
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import random 


# set problem up: number of species, complexity
n = 20
C = 1    # connectedness|
sigma2 = 1.2**2/n*2;       ## variance in off diagonals of interaction matrix
K = (C*sigma2*n/2)**0.5
print("complexity ( classic): ", K)

# set initial values, A matrix, time 
np.random.seed(1)

x0 = x0_vec(n)
seed = 314
A = A_matrix(n, C, sigma2, seed, LH=1) 
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
A_rowsums = np.dot(A, np.ones(n))

t_end = 500000     # length of time 
Nt =20000
t = np.linspace(0, t_end, Nt)

# number of times to run ODE, and output vectors
runs = 200

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)
meigs_real_max = np.zeros(runs)

gs = np.zeros(runs)
#fs = np.zeros(runs)
muas = np.zeros(runs)
mucs = np.zeros(runs)

fs = np.linspace(0, 2, runs)
xstar = 1


for run in range(runs):
    seed = run
    np.random.seed(seed)

    #g = np.random.uniform(low=-.5, high = 1)
    muc = -0.5
    mua = -0.5
    g = 1.
    f = fs[run]
    if f == -muc:
        f == -muc + 0.001


    fs[run] = f
    gs[run] = g
    muas[run] = mua
    mucs[run] = muc

    #print('f:',f,'g:',g)
    M = M_matrix(n, muc, mua, f, g)
    M_rowsums = np.dot(M, np.ones(n))
    
    if xstar == 1:
        xs = np.ones(n)
        for i in range(0, n, 2):
            xs[i] = 1
        #print(xs)
        A_rows = np.dot(A, xs)
        M_rows = np.dot(M, xs)
        scales = -np.divide(np.multiply(A_rows, xs), M_rows)
        M = np.multiply(M, np.outer(scales, np.ones(n)))
    
    mvals, mvecs = np.linalg.eig(M)
    meigs_real_max[run] = np.max(np.real(mvals))

   
    result = lv_LH(x0, t, A, M)
    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-6:
            species_left+=0.5
        if abs(result[-1, i] - result [-2, i]) < 1e-4:
            species_stable += 0.5
    
    n_species[run] = species_left
    n_stable[run] = species_stable

    if xstar ==1: 
        Jac = LH_jacobian(n, A, M, xs)
        #Jac = LH_jacobian_norowsum(result[-1, :], A, M)
    elif xstar ==0:
        Jac = LH_jacobian_norowsum(result[-1, :], A, M)
    Jvals, Jvecs = np.linalg.eig(Jac)

    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))

    '''if np.max(np.real(Jvals)) > 50 and f < 0.4:
        print('f: ', f)'''




# figures 

#stables = np.ma.masked_outside(eigs_real_max, 0, 50)
stables = np.ma.masked_less_equal(eigs_real_max, 0)
unstables = np.ma.masked_greater(eigs_real_max, 0)
#zeros = np.ma.masked_outside(eigs_real_max, -1e-2, 1e-2)

plt.figure(figsize=(6,5))
'''ax = plt.axes(projection='3d')
ax.plot3D(fs, gs, stables, 'ob', ms=2, alpha=.5)
ax.plot3D(fs, gs, unstables, 'og', ms=2, alpha=.5)
ax.grid()
ax.set_title('Maximum Real Eigenvalue of J, for f and g')
ax.set_xlabel('f')
ax.set_ylabel('g')
ax.set_zlabel('max real eigenvalue')'''

plt.plot(fs, stables, '-b', ms=2, alpha=.5)
plt.plot(fs, unstables, '-g', ms=2, alpha=.5)
plt.grid()
plt.title('Maximum Real Eigenvalue of J, for varied f')
plt.xlabel('f')
plt.ylabel('max real eigenvalue')
plt.ylim(-2, 50)


plt.figure(figsize=(6,5))
plt.plot(fs, n_species, '.b', ms=2, alpha=.5)
plt.grid()
plt.title('# of species left, for varied f, x*=1')
plt.xlabel('f')
plt.ylabel('# of species left ')
#plt.ylim(-2, 0)

plt.figure(figsize=(6,5))
plt.plot(fs, meigs_real_max, '.r', ms=2, alpha=.5)
plt.grid()
plt.title('max real eig of M, for varied f, x*/=1')
plt.xlabel('f')
plt.ylabel('max real eig of M')
#plt.ylim(-2, 50)


plt.show()









    


