
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 

n = 20

x0 = 0.5 * np.ones(n)
C = 0.1    # connectedness|
sigma2 = 0.5;       ## variance in off diagonals of interaction matrix
t_end = 30     # length of time 
Nt = 1000
K = (C*sigma2*n)**0.5
print("complexity: ", K)

r = np.ones(n)

runs = 2000

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)

gs = np.zeros(runs)
fs = np.zeros(runs)
muas = np.zeros(runs)
mucs = np.zeros(runs)

A = A_matrix(n, C, sigma2, 1, LH=1) 
#print(A)

numzeros = 0
A_rowsums = np.zeros(n)
for i in range(n):
    #print(A[i, :])
    for j in range(n):
        A_rowsums[j] += A[j][i]
print('rowsum: ',A_rowsums)


for run in range(runs):
    seed = run
    np.random.seed(seed)

    

    g = np.random.uniform(low=-.25, high = 3)
    f = 0.5
    muc = 0.5
    mua = 0.5

    fs[run] = f
    gs[run] = g
    muas[run] = mua
    mucs[run] = muc

    #print('f:',f,'g:',g)
    M = M_matrix(n, muc, mua, f, g)
    #print(M)
    #print(A_rowsums[5])
    #print(f+mua)

    for i in range(n):
        for j in range(n):
            if i%2 ==0:
                M[i][j] = M[i][j]*A_rowsums[i]/(muc+f) 
            if i%2 ==1:
                M[i][j] = M[i][j]*A_rowsums[i]/(mua+g) 
    #print(M)
            
    evals, evecs = np.linalg.eig(A)

    def derivative(x, t, r, A):
        for i in range(0, n):
            if x[i] <=0:
                x[i] = 0
        dxdt = np.multiply(r, x) + np.multiply(x, np.dot(A, x))
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

    delt = np.dot(A, result[-1, :])
    
    Jac = np.zeros((n,n))
    for i in range(n):

        for k in range(n):
            Jac[i][k] = M[i][k] + result[-1, i]*A[i][k]
            if(i==k):
                Jac[i][k] += delt[i]
    Jvals, Jvecs = np.linalg.eig(Jac)
    #print(Jac)
    #print(Jvals)
    eigs_real[:, run] = np.real(Jvals)
    eigs_imag[:, run] = np.imag(Jvals)
    eigs_real_max[run] = np.max(np.real(Jvals))




# figures 
plt.figure(figsize=(7, 7))
plt.grid()
plt.title("Max Eigenvalue of J")
plt.xlabel('g')
plt.ylabel('max real eigenvalue')
#plt.plot(muas, eigs_real_max, 'ob', ms=2, alpha=.5, label = 'mua')
plt.plot(gs, eigs_real_max, 'og', ms=2, alpha=.5)
plt.savefig('figures/lv_LH/fixed_A/maxJval_g_mua5muc5f5.pdf')



plt.show()









    


