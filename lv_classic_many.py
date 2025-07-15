
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import x0_vec
from lv_functions import lv_classic
import random 

n = 20

x0 = x0_vec(n,1)
sigma2 = 1**2/n
C = 1
t_end = 100     # length of time 
Nt = 1000
K = (C*sigma2*n)**0.5
print("complexity: ", K)

runs = 1000

# make empty matrices

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)

abundances = []


for run in range(runs):
    seed = run
    np.random.seed(seed)
    A = A_matrix(n, C, sigma2, seed, LH=0) 
    A_rowsums = np.dot(A, np.ones(n))

    r = -np.dot(A, np.ones(n))      # this is what makes all the equilibrium populations the same. 
    #r = np.ones(n)
    evals, evecs = np.linalg.eig(A)

    #print("max eigenvalue: ", np.max(np.real(evals)))


    t = np.linspace(0, t_end, Nt)
    result = lv_classic(x0, t, A, r)

    xf = result[-1, :]

    species_left = 0
    species_stable = 0
    for i in range(n):
        if xf[i] > 1e-4:
            species_left+=1
        if abs(result[-1, i] - result [-2, i]) < 1e-4:
            species_stable += 1
    
    n_species[run] = species_left
    n_stable[run] = species_stable
    eigs_real[:, run] = np.real(evals)
    eigs_imag[:, run] = np.imag(evals)
    eigs_real_max[run] = np.max(np.real(evals))

    if 0 < np.max(np.real(evals)) and species_left == 10:
        k = np.argmax(np.real(evals)) 
        if np.imag(evals)[k] != 0:
            print('seed:', seed, 'max real eig:', np.max(np.real(evals)), ', species left: ', species_left)
    abundances.extend(xf)





print('max imaginary eigenvalue: ', np.max(eigs_imag))

plt.figure()
plt.grid()
plt.title("eigenvalues of A")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(eigs_real, eigs_imag, 'o', ms=1)

plt.figure()
plt.grid()
plt.title("max real eigenvalue vs. number of stable species")
plt.xlabel('max real component of an eigenvalue')
plt.ylabel('number of stable species')
plt.plot(eigs_real_max, n_stable, 'o', ms=3)

plt.figure()
plt.hist(abundances, bins=100)
plt.grid()






plt.show()

    


    


