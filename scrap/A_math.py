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
from scipy import stats 
#region variables to change
K_set = 1
n = 30
C = 1
sigma2 = K_set**2/n*2

s = int(n/2)
seed = 1

K_lh = (sigma2*n*C)**0.5
K_classic = (sigma2*s*C)**0.5

print('LH K:', K_lh, ', K_classic:', K_classic)

# endregion variables 

runs = 300
A_rowsums = []
A_inv_rowsums = []

for run in range(runs):
    A = A_matrix(s, C, sigma2, run, LH=0)
    A_inv = np.linalg.inv(A)
    A_rs = np.dot(A, np.ones(s))
    A_inv_rs = np.dot(A_inv, np.ones(s))
    A_rowsums.extend(A_rs)
    A_inv_rowsums.extend(A_inv_rs)

def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma2: variance of Gaussian
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian
        
range = [np.min(A_rowsums), np.max(A_rowsums)]

rs_counts, rs_be = np.histogram(A_rowsums, 100, range=range)
rs_bc = (rs_be[:-1] + rs_be[1:]) / 2

rsi_counts, rsi_be = np.histogram(A_inv_rowsums, 100, range=range)
rsi_bc = (rsi_be[:-1] + rsi_be[1:]) / 2

pars_rs, covs_rs = curve_fit(gaussian, rs_bc, rs_counts)
pars_rsi, covs_rsi = curve_fit(gaussian, rsi_bc, rsi_counts)

rng = np. linspace(np.min(A_rowsums), np.max(A_rowsums), 100)
print('sigma2: ', sigma2, ', fit sigma2: ', pars_rs[2])
print(pars_rs)
print(pars_rsi)

plt.figure()
plt.stairs(rs_counts, rs_be, fill =True)
plt.plot(rng, gaussian(rng, *pars_rs), '--')
plt.xlim((-5, 3))
plt.grid()

plt.figure()
plt.stairs(rsi_counts, rsi_be, fill =True, alpha = 0.7)
plt.plot(rng, gaussian(rng, *pars_rsi), '--')
plt.plot(rng, gaussian(rng, *pars_rs), '--')
plt.grid()

plt.figure()
plt.plot(A_rowsums, A_inv_rowsums, '.')
plt.grid()


plt.show()