#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
'''from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec'''

import random 
import math

'''ef M_matrix_rand(n, mumuc, smuc, mumua, smua, muf, sf, mug, sg, seed):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    np.random.seed(seed)
    M = np.zeros((n, n))
    for i in range (0, n-1):
        if i %2 == 0:
            M[i][i] = np.random.uniform(mumuc-smuc, mumuc+smuc)
            print(i, 'muc')
            M[i][i+1] = np.random.uniform(muf-sf, muf+sf)
            print(i, 'f')
            M[i+1][i] = np.random.uniform(mug-sg, mug+sg)
            print(i, 'g')
            M[i+1][i+1] = np.random.uniform(mumua-smua, mumua+smua)
            print(i, 'muA')
        
        
    return M

M = M_matrix_rand(6, -1, 0.5, -2, 0.1, 10, 1, 100, 1, 1)
print(M)
'''

'''x = np.array([1,4,3,2,7,3,7,9])
C = x.reshape(-1, 2).sum(axis=1)
J = x[::2]
A = x[1::2]
print(C)
print(J)
print(A)'''

