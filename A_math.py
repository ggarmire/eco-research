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
K_set = 0.8
n = 30
C = 1
sigma2 = K_set**2/n*2
#sigma2 = 0.04
#K_set = (sigma2*n/2)**0.5
print('n: ', n, ', K: ', K_set, ', sigma^2: ', sigma2)

s = int(n/2)
# endregion variables 

#region makw arrays 
A_classic_rs = []
A_classic_inv_rs = []
#endregion

runs = 1000
One = np.ones(n)
One_cl = np.ones(s)
zest = 1e-5

maxeig_A_cl = []
maxeig_A_LH = []
#region loop
for run in range(runs): 
    A_classic = A_matrix(s, C, sigma2, run, LH=0)
    A_cl_inv = np.linalg.inv(A_classic)
    Aclvals, trash = np.linalg.eig(A_classic)
    maxeig_A_cl.append(np.max(np.real(Aclvals)))
    
    A_classic_rs.extend(np.dot(A_classic, One_cl))
    A_classic_inv_rs.extend(np.dot(A_cl_inv, One_cl))

    A_LH = A_matrix(n, C, sigma2, run, LH=1)
    ALHvals, trash =np.linalg.eig(A_LH)
    maxeig_A_LH.append(np.max(np.real(np.ma.masked_inside(ALHvals, -zest, zest))))

#endregion loop

# region fit

#endregion


# region plotting 

fsize = (6, 6)
plt.figure(figsize=fsize)
plt.plot(A_classic_rs, A_classic_inv_rs, '.', alpha = 0.5)
plt.xlabel('rowsum')
plt.ylabel('inverse of rowsum')
plt.grid()

plt.figure(figsize=fsize)
plt.plot(maxeig_A_cl, maxeig_A_LH, '.')
plt.xlabel('classic')
plt.ylabel('LH')
plt.grid()
plt.title('max eig of A')

plt.show()








