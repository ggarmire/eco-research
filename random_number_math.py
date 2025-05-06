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

# region setup 
s = 20
n = 2*s

sigma2 = 0.5

sigma = sigma2**0.5

runs = 100000

rs_classic = []
rs_LH = []

rands = []
rands2 = []




# region fitting function

def gauss(x, A, mu, sigma2):
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


# region loop 

for run in range(runs):
    #np.random.seed(run)

    rands.append(np.random.normal(0, (4*(s-1))**0.5*sigma))
    rands2.append(np.random.normal(0, sigma) + np.random.normal(0, sigma))

    rands_classic = np.random.normal(0, sigma, size=s-1)
    rs = np.sum(rands_classic) 
    rs_classic.append(rs)
    rs_LH.append(2*rs)
    rs_LH.append(2*rs)




# region plost analysis 

bins = 100

# just random numbers: 
count_rand, bin_rand = np.histogram(rands, bins = bins)
count_rand2, bin_rand2 = np.histogram(rands2, bins = bins)

bc_rand = (bin_rand[:-1]+bin_rand[1:])/2
bc_rand2 = (bin_rand2[:-1]+bin_rand2[1:])/2

p0_rand = [np.max(count_rand), 0, sigma2]
p0_rand2 = [np.max(count_rand2), 0, 2*sigma2]

pars_rand, covs_rand = curve_fit(gauss, bc_rand, count_rand, p0_rand, maxfev = 2000)
pars_rand2, covs_rand2 = curve_fit(gauss, bc_rand2, count_rand2, p0_rand2, maxfev = 2000)
print('pars random:', pars_rand)
print('pars2 random:', pars_rand2)

# the classic case:
count_classic, bin_classic = np.histogram(rs_classic, bins = bins)
bc_classic = (bin_classic[:-1] + bin_classic[1:]) / 2

p0_classic = [float(np.max(count_classic)), -1, (s-1)*sigma2]
pars_classic, covs_classic = curve_fit(gauss, bc_classic, count_classic, p0 = p0_classic)

print('pars classic guess: ', p0_classic) 
print('pars classic: ', pars_classic)

# LH case:
count_LH, bin_LH = np.histogram(rs_LH, bins = bins)
bc_LH = (bin_LH[:-1] + bin_LH[1:]) / 2

p0_LH = [float(np.max(count_LH)), -2, 4*(s-1)*sigma2]
pars_LH, covs_LH = curve_fit(gauss, bc_LH, count_LH, p0 = p0_LH)

print('pars LH guess: ', p0_LH) 
print('pars LH: ', pars_LH)


# region plotting 

# just random numbers
plt.figure()
plt.plot(bc_rand, count_rand)
plt.plot(bc_rand2, count_rand2)
plt.plot(bc_rand, gauss(bc_rand, *pars_rand))
plt.plot(bc_rand2, gauss(bc_rand2, *pars_rand2))


# row sums 
plt.figure()
plt.plot(bc_rand, count_rand, label = 'random')
plt.plot(bc_classic, count_classic, label='classic dist')
plt.plot(bc_classic, gauss(bc_classic, *pars_classic), label='classic fit')

#plt.plot(bc_classic, 2* count_classic, label='2*classic dist')

plt.plot(bc_LH, count_LH, label='LH dist')
plt.plot(bc_LH, gauss(bc_LH, *pars_LH), label='classic fit')

plt.legend()

plt.show()

