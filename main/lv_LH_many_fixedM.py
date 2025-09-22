
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 

n = 20

x0 = x0_vec(n, 1)

C = 1    # connectedness|
sigma2 = 0.99**2/n;       ## variance in off diagonals of interaction matrix
t_end = 100     # length of time 
Nt = 1000
K = (C*sigma2*n)**0.5
print("complexity: ", K)

runs = 1000

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)


muc = -0.5
mua = -0.5
f = 0.3
g = 1


for run in range(runs):
    seed = run
    np.random.seed(seed)

    A = A_matrix(n, C, sigma2, seed, LH=1) 
    A_rowsums = np.dot(A, np.ones(n))
    A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
    Avals, Avecs = np.linalg.eig(A_classic)

    M = M_matrix(n, muc, mua, f, g)
    M_rowsums = np.dot(M, np.ones(n))

    for i in range(n):
        for j in range(n):
            M[i][j] = -M[i][j]*A_rowsums[i]/M_rowsums[i] 

    t = np.linspace(0, t_end, Nt)
    result = lv_LH(x0, t, A, M)

    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-6:
            species_left+=1
        if abs(result[-1, i] - result [-2, i]) < 1e4:
            species_stable += 1
    
    n_species[run] = species_left
    n_stable[run] = species_stable

    
    Jac = LH_jacobian(n, A, M)
    Jvals, Jvecs = np.linalg.eig(Jac)
    #print(Jac)
    #print(Jvals)
    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))

    if -0.15<np.max(np.real(Jvals))<-0.1:
        if species_left == 20:
            print("seed: ", seed, 'max real eigenvalue:', np.max(np.real(Jvals)))
            print('max eig of A: ', np.max(np.real(Avals)))
            #print('species remaining: ', species_left)
    '''if 0 < np.max(np.real(Avals)) < 0.1: 
        if species_left == 20:
            print("seed: ", seed, 'max real eigenvalue:', np.max(np.real(Jvals)))
            print('max eig of A: ', np.max(np.real(Avals)))'''


name = str('f5g5muc5mua5')

plt.figure(figsize=(7, 7))
plt.grid()
plt.title("eigenvalues of Jacobian, f=0.9")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(eigs_real, eigs_imag, 'o', ms=2, alpha=.5)
plt.xlim([-13, 6])
plt.ylim([-2.5, 2.5])
plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5.pdf')
'''plt.xlim([-350, 50])
plt.ylim([-50, 50])
plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5_zoom.pdf')

plt.savefig('figures/lv_LH/fixed_M/Jvals_f5g5muc5mua5_zoom_zoom.pdf')'''

'''plt.figure(figsize=(5,5))
plt.grid()
plt.title("max real eigenvalue vs. number of stable species")
plt.xlabel('max real component of an eigenvalue')
plt.ylabel('number of stable species')
plt.plot(eigs_real_max, n_stable + n_species - n, 'o', ms=3, alpha=.5)
#plt.savefig('figures/lv_LH/fixed_M/maxJvalstable_f5g5muc5mua5.pdf')'''

plt.show()

    


    


