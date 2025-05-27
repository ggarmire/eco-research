
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

pi = math.pi

#region variables to change
K_set = 0.4
muc = -0.5
mua = -0.5
f = 1.5
g = 1

z = 0.7
xstar = 1
n = 50

runs = 5000
#endregion

#region set up other variables
x0 = x0_vec(n)
C = 1
sigma2 = K_set**2/n*2
print('simga2 A:', sigma2)
t_end = 30     # length of time 
Nt = 1000
s = n/2

xs = np.ones(n)
for i in range(0, n, 2):
    xs[i] = z
#endregion

# region analytically calculate eigs of M, without A 
M_pre = M_matrix(n, muc, mua, f, g)
eigs_Mpre, bad = np.linalg.eig(M_pre)

print('eigs before any scaling: ', eigs_Mpre[0:2])


Mdotxs = np.dot(M_pre, xs)
Mdotxsdiv = np.divide(Mdotxs, xs)       # = Adotxs
Mdotxsdiv2 = Mdotxsdiv*2/(z+1)

M_scale = np.divide(M_pre, Mdotxsdiv2)
eigs_Mscale, bad = np.linalg.eig(M_scale)
print('eigs Mscale = ', eigs_Mscale[0:2])

Mprime = M_scale - (z+1)/2*np.identity(n)

eigs_Mprime, bad = np.linalg.eig(Mprime)
print('eigs_Mprime: ',eigs_Mprime[0:4])

l2p = eigs_Mprime[1]

sigma2_Mdist = l2p**2 * 4*(s-1) * sigma2

eigs_Mprimenum = []
eigs_A = []

# region M from traditional rowsum stuff 
M_rs = np.dot(M_pre, xs)
M = np.zeros((n,n))
for run in range(runs):
    seed = run
    A = A_matrix(n, C, sigma2, seed, LH=1)
    A_rs = np.dot(A, xs)
    for i in range(n):
        for j in range(n):
            M[i,j] = -M_pre[i,j]/M_rs[i] * A_rs[i] * xs[i]

    Mprime_num = M + np.diag(A_rs)

    eigs_Mpn, bad = np.linalg.eig(Mprime_num)
    eigs_Mprimenum.extend(eigs_Mpn)

    eigsa_run, bad = np.linalg.eig(A)
    eigs_A.extend(eigsa_run)
    

#region define function 

def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian


# region make histograms 

# histograms: 
nbins = 200

counts_mp, bins_mp = np.histogram(np.real(eigs_Mprimenum), bins=nbins)
print('integral: ', np.sum(counts_mp))

bc_mp = np.real((bins_mp[:-1] + bins_mp[1:]) / 2)
binwidth = bins_mp[2] - bins_mp[1]
fitminbin = 0
fitmaxbin = int((-0.1 - bins_mp[0]) / binwidth)

A_guess = s*runs / ((2*pi*sigma2_Mdist)**0.5) * binwidth

# region fit histograms 


p0 = [A_guess, 2*l2p, sigma2_Mdist]

parsmp, covsp = curve_fit(gaussian, bc_mp[fitminbin:fitmaxbin], counts_mp[fitminbin:fitmaxbin], p0)


#region make figures 
plt.figure()
plt.plot(bc_mp, counts_mp, label='Mprime')
plt.plot(bc_mp, gaussian(bc_mp, *parsmp), label = 'fit of Mprime')
plt.plot(bc_mp, gaussian(bc_mp, *p0), label = 'fit guess')
plt.grid()
plt.legend()


print('pars guess: ', p0)#
print('pars true: ', parsmp)
'''
plt.figure()
plt.grid()
plt.plot(bc_mdelt, counts_mdelt, label='M+delta')
plt.plot(bc_mdelt, gaussian(bc_mdelt, *p0delt), label = 'fit guess')
plt.legend()

x1 = np.min(Ars_max)
x2 = np.max(Ars_max)
space = np.linspace(x1, x2,10)
spacey = -m2delt * space


plt.figure()
plt.plot(Ars_max, Meig_max, '.', label = "M' eigenvalues")
plt.plot(space, spacey, '--', label = "y= -$\lambda_2'$*x")
plt.xlabel('Max row sum in A')
plt.ylabel('Max eigenvalue of M')
plt.legend()
plt.grid()
'''



plt.show()


