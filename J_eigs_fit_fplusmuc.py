
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
import pandas as pd 

#region variables 

# constant variables 
n = 50
length = 25
# to change
K_set = 0.4
sigma2 = K_set**2/n*2

fs = np.random.uniform(0.55, 2, size=length)
fplusmuc = []

g = 1
mua = -0.5

# to fill: 
means = []
sigma2s = []


# constants for use 

One = np.ones(n)
s = int(n/2)
x0 = x0_vec(n)
C = 1

runs = 500

# region function for fit 

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



# region make loop over values of changing variable
 
for i in range(length):

    f = fs[i]
    muc = random.uniform(-f+0.01, -0.001)
    fplusmuc.append(f + muc)

    M_pre = M_matrix(n, muc, mua, f, g)

    if i%5 == 0: print('on loop ', i, 'f/g/muc/mua/K: ', f, g, muc, mua, K_set)

    eigs = []

    # region loop thru different values of A 
    for run in range(runs):
        seed = run
        np.random.seed(seed)

        # make A: 
        A = A_matrix(n, C, sigma2, seed, LH=1) 

        # fix M:
        A_rows = np.dot(A, One)
        M_rows = np.dot(M_pre, One)
        scales = -np.divide(np.multiply(A_rows, One), M_rows)
        M = np.multiply(M_pre, np.outer(scales, np.ones(n)))

        # make Jacobian: 
        Jac = LH_jacobian(n, A, M, One) 
        Jvals, Jvecs = np.linalg.eig(Jac)   

        eigs.extend(Jvals)

    # region get the real axis eigenvalues

    eigs_real_axis = []
    for j in range(len(eigs)):
        if abs(eigs[j].imag) <= 1e-7:
            eigs_real_axis.append(eigs[j].real)
    
    # region make and fit histogram 
    nbins = 70
    histrange = (np.min(eigs_real_axis), np.max(eigs_real_axis))
    counts, bin_edges = np.histogram(np.real(eigs_real_axis), bins=nbins, range = histrange)
    bin_centers = np.real((bin_edges[:-1] + bin_edges[1:]) / 2)


    #print(histrange)

    b1 = np.digitize(-2-2.5*K_set, bin_edges)
    b2 = np.digitize(-2+2.5*K_set, bin_edges)

    #print(b1, b2)

    r1 = bin_centers[:b1]; r2 = bin_centers[b2:]
    c1 = counts[:b1]; c2 = counts[b2:]
    fitx = []; fity = []
    fitx.extend(r1); fitx.extend(r2)
    fity.extend(c1); fity.extend(c2)

    p0 = [100, -7, 1067*sigma2]

    pars, cov = curve_fit(gaussian, fitx, fity, p0)
    means.append(pars[1])
    sigma2s.append(pars[2])


df = pd.DataFrame({'f': fs, 'fplusmuc': fplusmuc, 'mean': means, 'sigma2': sigma2s})

df.to_csv('J_fit_data_fplusmuc.csv', index = False)

plt.figure()
plt.plot(fplusmuc, means, label = 'fplusmuc')
plt.xlabel('f')
plt.ylabel('mean of fit ')
plt.grid()


plt.figure()
plt.plot(fplusmuc, sigma2s, label = 'K')
plt.xlabel('f')
plt.ylabel('sigma2 of fit')
plt.grid()


plt.show()



