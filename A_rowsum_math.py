#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
import random 
import math

#region variables to change
K_set = 0.6
n = 30
C = 1
sigma2 = K_set**2/n*2
#sigma2 = 0.04
#K_set = (sigma2*n/2)**0.5
print('n: ', n, ', K: ', K_set, ', sigma^2: ', sigma2)

A_rowsums = []
A_eigs = []
A_eigs_classic = []
A_rs_max = []

randoms = []

One = np.ones(n)

runs = 30

s = int(n/2)

#endregion 

# region define function 

def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian


def gauss2(x, A, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    return A * np.exp(-2*((x)**2) / (2 * sigma2)) - 2

# endregion

pos_rs = 0
# region loop 
for run in range(runs):
    seed = run
    if run %778 == 0:
        print(run)

    A = A_matrix(n, C, sigma2, seed, LH=1) 
    #print(A)
    A_classic = A_matrix(s, C, sigma2, seed, LH=0)

    A_rs = np.dot(A, One)

    Avals, Avecs = np.linalg.eig(A)
    Avals_classic, Avecs_classic = np.linalg.eig(A_classic)

    A_rowsums.extend(A_rs)
    A_eigs.extend(Avals)

    A_eigs_classic.extend(Avals_classic)

    A_rs_max.append(np.max(A_rs))

    if np.max(A_rs) > 0:
        pos_rs += 1
    print('seed:', seed, ', A_rs:', np.max(A_rs))

    '''for i in range(s):
        rand = np.random.normal(0, (s-1)*sigma2) - 2
        randoms.append(rand)
        randoms.append(rand)'''

    #if abs(np.max(np.real(Avals))) <= 1e-10:
    #    print(seed)



print('sigma2:', sigma2, ', K:', K_set, ', pct positive rowsums:', pos_rs/runs * 100, '%')
#region post 

A_eigs_realaxis = []
A_eigs_realaxis_classic = []

for i in range(len(A_eigs)):
    if A_eigs[i].imag < 1e-7:
        A_eigs_realaxis.append(A_eigs[i].real)
for i in range(len(A_eigs_classic)):
    if A_eigs_classic[i].imag < 1e-7:
        A_eigs_realaxis_classic.append(A_eigs_classic[i].real)


# fit histogram of row sums 
c_rs, b_rs = np.histogram(np.real(A_rowsums), bins=200)

c_rs_max, be_rs_max = np.histogram(A_rs_max, bins = 50)
bc_rs_max = (be_rs_max[:-1] + be_rs_max[1:]) / 2



c_rand, b_rand = np.histogram(randoms, bins=200)
b_rs_centers = np.real((b_rs[:-1] + b_rs[1:]) / 2)

p0_rs = [float(np.max(c_rs)), -2, 4*(s-1)*sigma2]

pars_rs, cov_rs = curve_fit(gaussian, b_rs_centers, c_rs, p0_rs)

#A_rs_int = np.sum(c_rs)
#c_rs = c_rs * (1/A_rs_int)

'''A_rand_int = np.sum(c_rand)
c_rand = c_rand * (1/A_rand_int)

b_rand_centers = np.real((b_rand[:-1] + b_rand[1:]) / 2)'''


#p0_rs2 = [2, 2*(s-1)*sigma2]
#p0_rs = [(sigma2*2*math.pi)**(-0.5), -2, (s-1)*sigma2]#

#pars_rs2, cov_rs2 = curve_fit(gauss2, b_rs_centers, c_rs, p0_rs2)


plot_text = str('K = '+str(K_set)+', s = '+str(s)+'; '+str(runs)+' matrices')
fsize = (7, 7)


plt.figure(figsize=fsize)
plt.plot(b_rs_centers, c_rs, label = 'rowsums of A_LH')
#plt.plot(b_rand_centers, c_rand)
plt.plot(b_rs_centers, gaussian(b_rs_centers, *pars_rs), linestyle='dashed', label = 'fit of rowsum dist')
plt.plot(b_rs_centers, gaussian(b_rs_centers, pars_rs[0], p0_rs[1], p0_rs[2]), linestyle='dotted', label = 'N~(-2, 4(s-1)sig^2')
plt.xlabel('Rowsum of A')
plt.ylabel('counts')
plt.legend()
plt.grid()

plt.figure(figsize=fsize)
plt.plot(bc_rs_max, c_rs_max)
plt.xlabel('max rowsum in A')
plt.ylabel('counts')
plt.legend()
plt.grid()


#plt.plot(b_crs_centers, gauss2(b_crs_centers, *pars_rs2), '-b')
#plt.plot(b_crs_centers, gaussian(b_crs_centers, pars_rs[0], -2, (s-1)*sigma2), '-r')
print('predicted: ',p0_rs, ', actual: ',pars_rs)


'''
plt.figure(figsize=fsize)
plt.title('Eigenvalues of A matrix')
plt.plot(np.real(A_eigs), np.imag(A_eigs), '.b', alpha=0.3, label='A_LH')
plt.plot(np.real(A_eigs_classic), np.imag(A_eigs_classic), '.r', alpha=0.3, label='A_classic')
plt.xlabel('real axis')
plt.ylabel('imaginary axis')
plt.figtext(0.13, 0.86, plot_text)
plt.axis('square')
plt.legend()
plt.grid()

plt.figure()
plt.hist(A_eigs_realaxis, bins = 30, label = 'A_LH')
plt.hist(A_eigs_realaxis_classic, bins = 30, alpha=0.5, label = 'A_classic')
plt.title('real component of eigenvalues on the real axis')
plt.xlabel('real component')
plt.ylabel('counts')
plt.legend()
plt.grid()

plt.figure()
plt.hist(np.abs(A_eigs), bins = 100, alpha=1, label = 'A_LH')
plt.hist(np.abs(A_eigs_classic), bins = 200, alpha=0.5, label = 'A_classic')
plt.title('norm of eigenvalues of A')
plt.xlabel('norm of eigenvalue')
plt.ylabel('counts')
plt.figtext(0.13, 0.86, plot_text)
plt.yscale('log')
plt.legend()
plt.grid()

plt.figure()
plt.hist(np.abs(A_eigs), bins = 100, alpha=1, label = 'A_LH')
plt.hist(np.abs(A_eigs_classic), bins = 200, alpha=0.5, label = 'A_classic')
plt.title('norm of eigenvalues of A')
plt.xlabel('norm of eigenvalue')
plt.ylabel('counts')
plt.figtext(0.13, 0.86, plot_text)
plt.legend()
plt.grid()
'''


plt.show()



