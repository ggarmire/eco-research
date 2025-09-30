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

def bates(n, N):
    x = np.zeros(N)
    for i in range(n):
        x += np.random.uniform(0, 1, N)
        print(i)
    xvec = 1./n * x
    return xvec

b = bates(3, 3000)

c = np.ones(3000)+ 0.2*(b-0.5)

plt.figure()

plt.hist(b, bins=30, histtype='step')
plt.hist(c, bins=30, histtype='step')


plt.show()
