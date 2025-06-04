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

#region Set Variables 

# stuff that gets changed: 
n = 40
runs = 500

# A matrix 
random.seed(1)
K_set = 0.1
C = 1

# M matrix 
muc = -0.5
mua = -0.5
f = 1.5
g = 1


# stuff that does not get changed:
s = n/2
x0 = x0_vec(n, 1)
sigma2 = K_set**2/n*2
t = np.linspace(0, 200, 500)
M = M_matrix(n, muc, mua, f, g)
print('n:', n, ', sigma:', '%.3f'%(sigma2**0.5))
One = np.ones(n)

#region make arrays 
# A matrix: 
eigs_A = []     # eigenvalues of A 
maxeig_A = []
A_rowsums = []      # rowsums of A 
Ars_max = []        # max rowsum of A 

# M' matrix: M + delta Ars
eigs_Mp = []
maxeig_Mp = []

# Jacobian:
eigs_J = []     # eigenvalues of the jacobian 
eigs_J_died = []        # eigenvalues when not all species survive 
maxeig_J = []       # max eigenvalue for each run
maxeig_J_complex = []       # max eigenvalue, storing the complex value
maxeig_J_complex_died = []

# other:
n_survives = []     # number of subspecies that survive 

# endregion


# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %87 == 0:
        print(run)

    # make A matrix 
    A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
    A_rows = np.dot(A, One)
    A_rowsums.extend(A_rows)
    Ars_max.append(np.max(A_rows))
    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)

    Mp = M + np.diag(A_rows)        # mprime = m + delta
    Mpvals, Mpvecs = np.linalg.eig(Mp)
    eigs_Mp.extend(Mpvals)

    # run ODE solver 
    result = lv_LH(x0, t, A, M)         # with scaled M
    xf = result[-1, :]

    Jac = LH_jacobian_norowsum(xf, A, M)
    Jvals, Jvecs = np.linalg.eig(Jac) 
    eigs_J.extend(Jvals)

    n_survive = n
    for species in range(n):
        if xf[species] < 1e-3:
            n_survive -= 1
    n_survives.append(n_survive)
    if n_survive < n:
        eigs_J_died.extend(Jvals)
        maxeig_J_complex_died.append(np.max(Jvals))
        #print('seed:', seed, 'max real eig:',np.max(Jvals), 'species left:', n_survive)

    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10)      # nonzero eigenvalues of the two, since the zeros dissapear later 
    Mpvalsm = np.ma.masked_inside(Mpvals, -1e-10, 1e-10)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))
    maxeig_J_complex.append(np.max(Jvals))


# region analysis 
avg_n_survives = np.average(n_survives)
pct_blk_stable = n_survives.count(20) / runs * 100
print('K:', K_set, 'n:', n, 'average survived:', '%.3f'%avg_n_survives, 'stability', '%.3f'%pct_blk_stable, '% of runs')


# region plot setup 
fsize = (6,6)

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g))
apar_text = str('n='+ str(n)+', K='+str(K_set))


# region plotting 

# plot max rowsum vs number of remaining species 
plt.figure(figsize=fsize)
plt.plot(n_survives, Ars_max, '.')
plt.figtext(0.13, 0.12, mpar_text)
plt.xlabel('number of surviving subspecies')
plt.ylabel('Max rowsum in A')
plt.title('number of subspecies surviving, n='+str(n)+', K='+str(K_set))
plt.grid()

plt.figure(figsize=fsize)
plt.plot(Ars_max, maxeig_J, '.')
plt.figtext(0.13, 0.12, mpar_text)
plt.xlabel('Max rowsum in A')
plt.ylabel('Maximum eigenvalue of J')
plt.title('Max eigenvalue vs. maximum rowsum in A, n='+str(n)+', K='+str(K_set))
plt.grid()

plt.figure(figsize=fsize)
plt.plot(n_survives, maxeig_J, '.')
plt.figtext(0.13, 0.12, mpar_text)
plt.xlabel('number of surviving subspecies')
plt.ylabel('Maximum eigenvalue of J')
plt.title('Max eigenvalue vs. subspecies surviving, n='+str(n)+', K='+str(K_set))
plt.grid()




plt.show()



