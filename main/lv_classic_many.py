
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
sigma2 = 0.5**2/n
C = 1
t_end = 100     # length of time 
Nt = 1000
K = (C*sigma2*n)**0.5

xstar = 0

print("complexity: ", K)


runs = 1000

# make empty matrices

n_stable = np.zeros(runs)
n_species = np.zeros(runs)
eigs_real = np.zeros((n, runs))
eigs_imag = np.zeros((n, runs))
eigs_real_max = np.zeros(runs)

abundances = []

muc = -1
mua = -0.7
f = 1.5
g = 1.2
z = (muc-mua+((muc-mua)**2+4*g*f)**0.5) / (2*g)


R  = z*g+mua / (1+z)
if xstar == 0:
    r = np.ones(n)
elif xstar == 1:
    r = R*np.ones(n)
for run in range(runs):
    seed = run
    np.random.seed(seed)
    A = A_matrix(n, C, sigma2, seed, LH=0) 
    A_rowsums = np.dot(A, np.ones(n))

    if xstar == 1: r = -np.dot(A, np.ones(n))      # this is what makes all the equilibrium populations the same. 
    
    xf = -np.dot(np.linalg.inv(A), r)
    Jac = np.multiply(np.outer(xf, np.ones(n)), A)
    evals, evecs = np.linalg.eig(Jac)

    #print("max eigenvalue: ", np.max(np.real(evals)))

    eigs_real[:, run] = np.real(evals)
    eigs_imag[:, run] = np.imag(evals)
    eigs_real_max[run] = np.max(np.real(evals))



apar_text = str('n='+ str(n)+', K='+str(K)+', '+str(runs)+' runs'+', r='+str('%0.3f'%R))


print('max imaginary eigenvalue: ', np.max(eigs_imag))

plt.figure()
plt.grid()
plt.title("eigenvalues of Jacobian")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(eigs_real, eigs_imag, 'o', ms=1)
plt.figtext(0.13, 0.15, apar_text)

plt.figure()
plt.grid()
plt.title("max real eigenvalue vs. number of stable species")
plt.xlabel('max real component of an eigenvalue')
plt.ylabel('number of stable species')
plt.plot(eigs_real_max, n_stable, 'o', ms=3)




plt.show()

    


    


