#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math

n = 4
C = 1
sigma2 = 0.1
seed = 1
jscale = 0.8 

A = A_matrix(n, C, sigma2, seed, 0)
print(A)

xf = [0.8, 0.9, 1, 1.1]

xf0 = np.outer(xf, np.ones(n))
print(xf0)
