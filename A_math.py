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
K_set = 0.7071
n = 20
C = 1
sigma2 = K_set**2/n*2

s = int(n/2)
seed = 1

K_lh = (sigma2*n*C)**0.5
K_classic = (sigma2*s*C)**0.5

print('LH K:', K_lh, ', K_classic:', K_classic)

# endregion variables 


A_classic = A_matrix(s, C, sigma2, seed, 0)
A_lh = A_matrix(s, C, sigma2, seed, 1)

A_inv = np.linalg.inv(A_classic)

Avals_cl, bad = np.linalg.eig(A_classic)
Avals_lh, bad = np.linalg.eig(A_lh)
Avals_cl_inv, bad = np.linalg.eig(A_inv)

Aval_mag_cl = np.sqrt(np.square(np.real(Avals_cl)) + np.square(np.imag(Avals_cl)))

Aval_mag_inv = np.sqrt(np.square(np.real(Avals_cl_inv)) + np.square(np.imag(Avals_cl_inv)))

print('classic: ', Avals_cl)
print('inverse: ', Avals_cl_inv)



# region plotting 

xval = np.linspace(np.min(Avals_cl), np.max(Avals_cl), 100)
val = 1/xval


fsize = (6,6)


plt.figure(figsize = fsize)
plt.plot(sorted(-abs(Avals_cl)), sorted(-abs(Avals_cl_inv), reverse=True), '.')
plt.plot(sorted(Avals_cl), sorted(Avals_cl_inv, reverse=True), '.')
plt.plot(sorted(Aval_mag_cl*-1), sorted(Aval_mag_inv*-1, reverse=True), 'o', mfc='None')
plt.plot(xval, val, '--')
plt.grid()

plt.show()

