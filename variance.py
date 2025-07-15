#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
from lv_functions import x0_vec
import random 
import math

right_0 = 0
right_1 = 0

'''for i in range(100000):
    rands = np.random.normal(0, 0.5, 320)
    std_calc_0 = np.std(rands)
    std_calc_1 = np.std(rands, ddof = 1)
    diff_0 = abs(0.5-std_calc_0)
    diff_1 = abs(0.5-std_calc_1)
    if diff_0 < diff_1:
        right_0 += 1
    elif diff_1 < diff_0:
        right_1 += 1'''

'''print('0 right: ', right_0)
print('1 right: ',  right_1)'''

'''xf = np.array([0, 1, 2, 3, 5, 0, 1, 2])
print((xf>1).sum())'''

c = np.array([1+0j, 2+1j, 3, 4, 1, 4+9j, 2+0.5j])

closest = c[(np.abs(c-2.4)).argmin()]
print(closest)
c_mask = np.array(c)    
c_comp = [1, 3, 4]
for i in c_comp:
    closest = c[(np.abs(c-i)).argmin()]
    ind = np.where(c = closest)
    