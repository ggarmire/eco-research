
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
length = 20
# to change
K_sets = 1*np.ones(length*5)
mucs = -0.5*np.ones(length*5)
muas = -0.5*np.ones(length*5)
gs = 1*1*np.ones(length*5)
fs = 1.5*np.ones(length*5)

K_sets[0:length] = np.linspace(0.001, 2, length)
mucs[length:2*length] = np.linspace(-1.49, -1, length)
muas[2*length:3*length] = np.linspace(-0.99, -0.5, length)
fs[3*length:4*length] = np.linspace(0.51, 1.5, length)
gs[4*length:5*length] = np.linspace(0.51, 1.5, length)


# to fill: 
means = []
sigma2s = []
covmeans = []
covsigma2s = []



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
 
for i in range(5*length):
    f = fs[i]
    g = gs[i]
    muc = mucs[i]
    mua = muas[i]
    K = K_sets[i]
    sigma2 = K**2/n*2


    M_pre = M_matrix(n, muc, mua, f, g)

    print('on loop ', i, 'f/g/muc/mua/K: ', f, g, muc, mua, K)

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

    b1 = np.digitize(-2-2.5*K, bin_edges)
    b2 = np.digitize(-2+2.5*K, bin_edges)

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
    covmeans.append(cov[1])
    covsigma2s.append(cov[2])


df = pd.DataFrame({'f': fs, 'g': gs, 'muc': mucs, 'mua': muas, 'K': K_sets,
                   'mean': means, 'sigma2': sigma2s, 'covmean': covmeans, 'covsigma2': covsigma2s})

df.to_csv('J_fit_data.csv', index = False)

plt.figure()
plt.plot(K_sets[0*length:1*length], means[0*length:1*length], label = 'K')
plt.plot(mucs[1*length:2*length], means[1*length:2*length], label = 'muc')
plt.plot(muas[2*length:3*length], means[2*length:3*length], label = 'mua')
plt.plot(fs[3*length:4*length], means[3*length:4*length], label = 'f')
plt.plot(gs[4*length:5*length], means[4*length:5*length], label = 'g')
plt.grid()
plt.legend()

plt.figure()
plt.plot(K_sets[0*length:1*length], sigma2s[0*length:1*length], label = 'K')
plt.plot(mucs[1*length:2*length], sigma2s[1*length:2*length], label = 'muc')
plt.plot(muas[2*length:3*length], sigma2s[2*length:3*length], label = 'mua')
plt.plot(fs[3*length:4*length], sigma2s[3*length:4*length], label = 'f')
plt.plot(gs[4*length:5*length], sigma2s[4*length:5*length], label = 'g')
plt.grid()
plt.legend()

plt.show()



