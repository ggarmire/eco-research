
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
import random 

n = 20

x0 = 0.5 * np.ones(n)
#r = np.random.uniform(low=0, high=1, size=n)
r = np.ones(n)
C = 0.1    # connectedness|
sigma2 = 0.5;       ## variance in off diagonals of interaction matrix
t_end = 30     # length of time 
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


for run in range(runs):
    seed = run
    np.random.seed(seed)

    A = A_matrix(n, C, sigma2, seed, LH=0) 

    A_rowsums = np.zeros(n)
    for i in range(n):
        #print(A[i, :])
        for j in range(n):
            A_rowsums[j] += A[j][i]

    for i in range(n):
        r[i] = A_rowsums[i]        # this is what makes all the equilibrium populations the same. 
            
    evals, evecs = np.linalg.eig(A)

    #print("max eigenvalue: ", np.max(np.real(evals)))

    def derivative(x, t, r, A):
        for i in range(0, n):
            if x[i] <=0:
                x[i] = 0
        dxdt = np.multiply(r, x) - np.multiply(x, np.dot(A, x))
        for i in range(0, n):
            if x[i]<=0:
                    dxdt[i] == 0
        return dxdt

    t = np.linspace(0, t_end, Nt)
    result = integrate.odeint(derivative, x0, t, args = (r, A))

    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-3:
            species_left+=1
        if abs(result[-1, i] - result [-2, i]) < 1e4:
            species_stable += 1
    
    n_species[run] = species_left
    n_stable[run] = species_stable
    eigs_real[:, run] = np.real(evals)
    eigs_imag[:, run] = np.imag(evals)
    eigs_real_max[run] = np.max(np.real(evals))


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
plt.plot(eigs_real_max, n_stable + n_species - 20, 'o', ms=3)






plt.show()

    


    


