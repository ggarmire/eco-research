
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

seed = 24 #24 18 658
#seed = random.randint(0, 1000)
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)
t = np.linspace(0, 500, 1000)
K_set = 0.5
C = 1

muc = -0.5
mua = -0.5
f = 1.5
g = 1.2

p = 501 # number of percentage values run 

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
#if K!=K_set:
#  raise Exception("K set does not match K.")

# region set matrices 

# A matrix:
A = A_matrix(n, C, sigma2, seed, LH=1)

# A matrix specs
Avals, Avecs = np.linalg.eig(A)
Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10) 
max_eig_A = np.max(Avalsm)
max_eigs_A = max_eig_A*np.ones(p)

A_rowsums = np.dot(A, np.ones(n))
A_rs_max = np.max(A_rowsums)

# for m matrix:
M_pre = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M_pre)

# region run unconstrained case 
result_un = lv_LH(x0, t, A, M_pre)
xf_un = result_un[-1,:]

print(xf_un)
xs = np.ones(n)
species_left_un = 0
for i in range(n):
    if xf_un[i] > 1e-3:
        species_left_un += 1 
    if i % 2 == 0:
        xs[i] = xf_un[i]/xf_un[i+1]     # setting the juvinile fractions now! 
print(xs)

print('Species remain:', species_left_un, ', max rowsum:', A_rs_max)

diff_from_1 = np.ones(n) - xf_un

# region run constrained case with juvinile fraction from unconstrained 

# scale M with constriant: 

#print(xs)
A_rows = np.dot(A, xs)
M_rows = np.dot(M_pre, xs)
scales = -np.divide(np.multiply(A_rows, xs), M_rows)
M = np.multiply(M_pre, np.outer(scales, np.ones(n)))
if np.max(np.diag(M)) > 0: print('M has a positive diagonal.')
if np.min(np.diag(M, 1)) < 0: print('M has a negative f value.')
if np.min(np.diag(M, -1)) < 0: print('M has a negative g value.')

# region run unconstrained case 
result = lv_LH(x0, t, A, M)
xf = result[-1,:]





# region plot
fsize = (6, 6)


colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', A seed ='+str(seed)+ ', K='+str('%.3f'%K))


plt.figure()
plt.grid()
title = str('Species Population over time, unconstrained')
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result_un[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result_un[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
    else:
        plt.plot(0, result_un[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result_un[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.13, 0.12, plot_text)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)


plt.figure()
plt.grid()
title = str('Species Population over time, constrained by unconstrained zs')
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.13, 0.12, plot_text)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)






plt.show()
