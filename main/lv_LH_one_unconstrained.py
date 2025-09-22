
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

seed = random.randint(0, 1000)
seed = 270
print("seed: ", seed)



#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)
t = np.linspace(0, 500, 1000)
K_set = 0.6
C = 1

muc = -0.5
mua = -0.5
f = 1.5
g = 1

# option for M
mrandscale = 1      # random scale of blocks in M
r = 0    # range of scales away from 1 
#seed2 = random.randint(0, 1000)
seed2 = 3
print('seed2:', seed2)
#


# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")

# region set matrices 
A = A_matrix(n, C, sigma2, seed, LH=1)
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
Avals, Avecs = np.linalg.eig(A)

A_rowsums = np.dot(A, np.ones(n))
print('A max rowsum:', np.max(A_rowsums))


# for m matrix:
M = M_matrix(n, muc, mua, f, g)

if mrandscale == 1:
    np.random.seed(seed2)
    scales = np.random.uniform(1-r, 1+r, s)
    print('scales:', scales)
    for i in range(s):
        M[2*i,:] = scales[i]*M[2*i,:]
        M[2*i+1,:] = scales[i]*M[2*i+1,:]

mvals, mvecs = np.linalg.eig(M)
#print('scaled M:', M)


# region run function: 
result = lv_LH(x0, t, A, M)

#%% Stats: 
species_left = 0
species_stable = 0
for i in range(n):
    if result[-1, i] > 1e-3:
        species_left+=1
        if abs((result[-1, i]-result[-2, i]) / result[-1, i]) < 0.001:
            species_stable +=1

print("species remaining:", species_left, "sepcies stable: ", species_stable)


# region Calculate the Jacobian

Jac = LH_jacobian_norowsum(result[-1, :], A, M)
#print("Jacobian: ", Jac)
Jvals, Jvecs = np.linalg.eig(Jac)

#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', A seed ='+str(seed)+ ', K='+str('%.3f'%K))
if species_left == n:
    plot_text2 = str('Stable: ' + str(species_left) + ' survive')
else: 
    plot_text2 = str('Unstable: ' + str(species_left) + ' survive')
box_par = dict(boxstyle='square', facecolor='white', alpha = 0.5)


# region figures 

plt.figure()
plt.grid()
title = str('Species Population over time, r = '+str(r))
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
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
plt.figtext(0.4, 0.8, plot_text2, bbox=box_par)
#plt.figtext(0.5, 0.80, plot_text3)
#plt.figtext(0.5, 0.76, plot_text4)
plt.semilogx

legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)

#plt.ylim(-0.1, 6)

#plt.ylim(min(0, np.min(result)-0.1), 1.1*np.max(result))



plt.figure()
plt.grid()
plt.plot(np.real(Jvals), np.imag(Jvals), '.')



plt.show()