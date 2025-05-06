
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

# to change
sigma2s = np.random.uniform(0.0001, 0.1, 50)
n = 20
s = n/2

fs = []
gs = []
mucs = []
muas =[]

# to fill: 
Jmeans = []
Jmeanerrs = []
Jsigma2s = []
Jsigerrs = []

M_means = []
M_sig2s = []

# constants for use 
fs = np.random.uniform(0.6, 2, len(sigma2s))
gs = np.random.uniform(0.6, 2, len(sigma2s))
mucs = np.random.uniform(-0.5, 0, len(sigma2s))
muas = np.random.uniform(-0.5, 0, len(sigma2s))

print(fs)

C = 1

runs = 1200

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

def linear(x, m, b):
    return m*x + b



# region make loop over values of changing variable
 
for i in range(len(sigma2s)):

    sigma2 = sigma2s[i]


    K_set = (sigma2 * s)**0.5

    One = np.ones(n)
    s = int(n/2)
    x0 = x0_vec(n)

    f = fs[i]; g = gs[i]; muc = mucs[i]; mua = muas[i]

    M_pre = M_matrix(n, muc, mua, f, g)
    l = (muc*mua - f*g) / ((muc + f)*(g+mua))
    M_means.append(2*l -2)
    M_sig2s.append(4*(s-1)*sigma2*(l-1)**2)

    #if i%5 == 0: print('on loop ', i, 'f/g/muc/mua/K: ', f, g, muc, mua, K_set)
    print('loop ', i, ', n=', n, 'f/g/muc/mua:', f, g, muc, mua)

    eigs = []

    # region loop thru different values of A 
    for run in range(runs):
        seed = run

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
    Jmeans.append(pars[1])
    Jmeanerrs.append(cov[1,1]**0.5)
    #print('mean err:', cov[1,1]**0.5)
    Jsigma2s.append(pars[2])
    Jsigerrs.append(cov[2,2]**0.5)
    #print('std err:', cov[2,2]**0.5)


parsmean, covsmean = curve_fit(linear, M_means, Jmeans)
t_mean = str('y='+str('%0.3f'%parsmean[0])+'x+'+str('%0.3f'%parsmean[1]))

parssig, covssig = curve_fit(linear, M_sig2s, Jsigma2s)
t_sig = str('y='+str('%0.3f'%parssig[0])+'x+'+str('%0.3f'%parssig[1]))

plt.figure()
plt.plot(sigma2s, Jmeans, '.', label = 'mean')
plt.xlabel('n')
plt.ylabel('mean of J')
plt.grid()


plt.figure()
plt.plot(sigma2s, Jsigma2s, '.', label = 'sigma')
plt.xlabel('n')
plt.ylabel('sigma2 of J')
plt.grid()

plt.figure()
plt.title('Eigs of M vs. J, varied sigma^2')
plt.errorbar(M_means, Jmeans, yerr=Jmeanerrs,  fmt='.', label = 'mean')
plt.plot(np.linspace(np.min(M_means), np.max(M_means), 4), linear(np.linspace(np.min(M_means), np.max(M_means), 4), *parsmean), '-', label = 'fit')
plt.plot(np.linspace(np.min(M_means), np.max(M_means), 4), np.linspace(np.min(M_means), np.max(M_means), 4), '--', label = 'x=y')
plt.legend()
plt.xlabel('eig of M')
plt.ylabel('mean of J')
plt.figtext(0.4, 0.4, t_mean)
plt.grid()

plt.figure()
plt.title('Sigma^2 M vs. J, varied sigma^2')
plt.errorbar(M_sig2s, Jsigma2s, yerr=Jsigerrs, fmt='.', label = 'sigma')
plt.plot(np.linspace(np.min(M_sig2s), np.max(M_sig2s), 4), np.linspace(np.min(M_sig2s), np.max(M_sig2s), 4), '--', label = 'x=y')

plt.plot(np.linspace(np.min(M_sig2s), np.max(M_sig2s), 4), linear(np.linspace(np.min(M_sig2s), np.max(M_sig2s), 4), *parssig), '-', label = 'fit')
plt.legend()
plt.xlabel('sigma2 of M')
plt.ylabel('sigma2 of J')
plt.figtext(0.4, 0.4, t_sig)
plt.grid()





plt.show()



