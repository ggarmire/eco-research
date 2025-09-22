#region preamble 
print('\n')
import numpy as np
import matplotlib.pyplot as plt 
import sys
from lv_functions import A_matrix

# this code is intended to replicate the results of "Effect of population abundances on the stability of large random ecosystems"
# https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.022410

# endregion



# region variables
s = 500
# A matrix
mud = -1
sigma2d = 0
K_set = 0.5
mu = 0
sigma2 = K_set**2/s
C = 1

seed = np.random.randint(0, 1000)

K = (s*sigma2)**0.5
print('complexity K=', K)


# final abundances
mux = 1
sigmax = 0.1
# LH variables 
muc = -0.8
mua = -0.3
f = 1.5
g = 1 

z = ((muc - mua)+ ((mua - muc)**2 + 4*g*f)**0.5) / (2 * g)
R = muc + f/z
print('z = ', z, ', R = ', R)

Aeigs = []
Meigs = []
Meigs_setR = []
#endregion 

# region make arrays
A = np.empty((s, s))
xf_cl = np.random.uniform(0.1, 2, s)
xf = np.column_stack(( xf_cl*z, xf_cl)).ravel()




'''for i in range(s):
    for j in range(s):
        if j == i: A[i][j] = np.random.normal(mud, sigma2d**0.5)
        elif j!= i: A[i][j] = np.random.normal(mu, sigma2**0.5)'''
A_cl = A_matrix(s, C, sigma2, seed, LH=0)
A = A_matrix(2*s, C, sigma2, seed, LH=1)
M = np.dot(np.diag(xf), A)

A_inv = np.linalg.inv(A_cl)

xf_setR_adult = -np.dot(A_inv, R*np.ones(s))  # solve classical system
xf_setR = np.repeat(xf_setR_adult, 2)   # make unscaled
xf_setR[::2] *= z     # scale child
M_setR = np.dot(np.diag(xf_setR), A)

# endregion

Avals, no = np.linalg.eig(A)
Mvals, no = np.linalg.eig(M)
Mvals_setR, no = np.linalg.eig(M_setR)
Rf = -np.dot(A, xf)
Rone = -np.dot(A, np.ones(2*s))

Aeigs.extend(Avals)
Meigs.extend(Mvals)
Meigs_setR.extend(Mvals_setR)

range = [min(np.min(Rf), np.min(Rone)), max(np.max(Rf), np.max(Rone))]

rx = [min(0, np.min(xf_setR)), max(2, np.max(xf_setR))]

r_counts, r_be = np.histogram(Rf, 20, range)
rone_counts, rone_be = np.histogram(Rone, 20, range)

xf_counts, xf_be = np.histogram(xf, 20, rx)

xfR_counts, xfR_be = np.histogram(xf_setR, 20, rx)
mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str(z)+', R = '+str(R)+')')
apar_text = str('n='+ str(2*s)+', A seed ='+ str(seed)+', K='+str(K_set))


# region plot
plt.figure()
#plt.plot(np.real(Aeigs), np.imag(Aeigs), '.', alpha =0.8)
plt.plot(np.real(Meigs_setR), np.imag(Meigs_setR), '.', alpha =0.8, label = 'uniform R_eff')
plt.plot(np.real(Meigs), np.imag(Meigs), '.', alpha =0.8, label = 'randomly chosen x*')
plt.figtext(0.2, 0.15, apar_text)
plt.figtext(0.2, 0.12, mpar_text)

plt.grid()
plt.legend()

plt.figure()
plt.stairs(r_counts, r_be, fill=True)

plt.stairs(rone_counts, rone_be, fill=True, alpha=0.5)
plt.grid()

plt.figure()
plt.stairs(xfR_counts, xfR_be, fill=True, label = 'uniform R_eff')
plt.stairs(xf_counts, xf_be, fill=True, alpha=0.5, label = 'randomly chosen x*')
plt.xlabel('final abundances')
plt.grid()
plt.legend()

plt.show()
