
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 

n = 20

x0 = 0.5* np.ones(n)
np.random.seed(1)
for i in range(n):
    x0[i] += np.random.normal(loc=0, scale=0.2)

C = 1    # connectedness|
sigma2 = .9**2/n;       ## variance in off diagonals of interaction matrix
t_end = 40     # length of time 
Nt = 1000
K = (C*sigma2*n)**0.5
print("complexity: ", K)

r = np.ones(n)

runs = 1000

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)


for run in range(runs):
    seed = run
    np.random.seed(seed)

    A = A_matrix(n, C, sigma2, seed, LH=1) 
    
    #print(A)

    numzeros = 0
    A_rowsums = np.zeros(n)
    for i in range(n):
        #print(A[i, :])
        for j in range(n):
            A_rowsums[j] += A[j][i]

    muc = 0.5
    mua = 0.5
    f = 0.5
    g = 0.5
    M = M_matrix(n, muc, mua, f, g)

    M_rowsums = np.dot(M, np.ones(n))

    #print(M)

    for i in range(n):
        for j in range(n):
            M[i][j] = M[i][j]*A_rowsums[i]/M_rowsums[i]
            
    evals, evecs = np.linalg.eig(A)


    t = np.linspace(0, t_end, Nt)
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

    
    Jac = A+M
    for i in range(n):
        Jac[i][i] += A_rowsums[i]

    Jvals, Jvecs = np.linalg.eig(Jac)
    #print(Jac)
    #print(Jvals)
    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))

    if -0.01<np.max(np.real(Jvals))<0.01:
        print("seed: ", seed, 'max real eigenvalue:', np.max(np.real(Jvals)))

name = str('f5g5muc5mua5')

plt.figure(figsize=(7, 7))
plt.grid()
plt.title("eigenvalues of Jacobian")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(eigs_real, eigs_imag, 'o', ms=2, alpha=.5)
plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5.pdf')
'''plt.xlim([-350, 50])
plt.ylim([-50, 50])
plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5_zoom.pdf')
plt.xlim([-50, 50])
plt.ylim([-50, 50])
plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5_zoom_zoom.pdf')'''

plt.figure(figsize=(5,5))
plt.grid()
plt.title("max real eigenvalue vs. number of stable species")
plt.xlabel('max real component of an eigenvalue')
plt.ylabel('number of stable species')
plt.plot(eigs_real_max, n_stable + n_species - n, 'o', ms=3, alpha=.5)
#plt.savefig('figures/lv_LH/fixed_M/maxJvalstable_f5g5muc5mua5.pdf')

plt.show()

    


    


