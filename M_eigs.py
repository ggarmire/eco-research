
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

#region variables to change
K_set = 0.4
muc = -0.3
mua = -0.5
f = 1.5
g = 1

z = 1
xstar = 1
n = 50


runs = 1000
#endregion

#region set up other variables

x0 = x0_vec(n)
C = 1
sigma2 = K_set**2/n*2
t_end = 30     # length of time 
Nt = 1000
M_pre = M_matrix(n, muc, mua, f, g)
xs = np.ones(n)
for i in range(0, n, 2):
    xs[i] = z
#eigs_real = np.zeros((n, runs))
#eigs_imag = np.zeros((n, runs))
#eigs_real_max = np.zeros(runs)

eigs_M = []
eigs_Mdelt = []

M = M_matrix(n, muc/(muc+f), mua/(mua+g), f/(muc+f), g/(mua+g))
print(M)

eigs_M_unscaled, trash = np.linalg.eig(M)
eigs_M_unscaled_minusI, trash = np.linalg.eig(M+np.identity(n))
print('eigs of M, before random number: ', eigs_M_unscaled)
print('eigs of M+I, before scaling: ', eigs_M_unscaled_minusI)
#endregion 

Ars_max = []
Meig_max = []

# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %778 == 0:
        print(run)

    # make matrices 
    A = A_matrix(n, C, sigma2, seed, LH=1) 


    A_rows = np.dot(A, xs)
    M_rows = np.dot(M_pre, xs)
    scales = -np.divide(np.multiply(A_rows, xs), M_rows)
    M = np.multiply(M_pre, np.outer(scales, np.ones(n)))

    Mdelt = M + np.diag(A_rows)

    Mvals, Mvecs = np.linalg.eig(M)
    eigs_M.extend(Mvals)


    Mdeltvals, Mdeltvecs = np.linalg.eig(Mdelt)
    eigs_Mdelt.extend(Mdeltvals)

    MdeltM = np.ma.masked_inside(Mdeltvals, -1e-10, 1e-10)

    Ars_max.append(np.max(A_rows))
    Meig_max.append(np.max(MdeltM))



    
#endregion


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

def gauss2(x, A1, mu1, sigma21, A2, mu2, sigma22):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    gaussian = A1 * np.exp(-((x - mu1)**2) / (2 * sigma21)) + A2 * np.exp(-((x - mu2)**2) / (2 * sigma22))
    return gaussian


# region make histograms 

# histograms: 
nbins = 200

counts_m, bins_m = np.histogram(np.real(eigs_M), bins=nbins)
counts_m = counts_m / np.sum(counts_m)
print('integral: ', np.sum(counts_m))

bc_m = np.real((bins_m[:-1] + bins_m[1:]) / 2)

counts_mdelt, bins_mdelt = np.histogram(np.real(eigs_Mdelt), bins=nbins)
counts_mdelt = counts_mdelt/np.sum(counts_mdelt)
bc_mdelt = np.real((bins_mdelt[:-1] + bins_mdelt[1:]) / 2)


# region fit histograms 


m1 = 1              
m2 = (mua*muc - f*g)/((mua+g)*(muc+f))      # these are the eigenvlaues of the 
print('m1: ', m1, ', m2: ', m2)


sig2new = 4*(n/2 -1) * sigma2

s21 = sig2new*m1**2
s22 = sig2new*m2**2


p0 = [0.5/((2*3.1416*s21)**0.5), 2*m1, s21, 0.5/((2*3.1416*s22)**0.5), 2*m2, s22]

parsm, covsm = curve_fit(gauss2, bc_m, counts_m, p0)


m1delt = m1 - 1
m2delt = m2 - 1 
s21delt = sig2new*m1delt**2
s22delt = sig2new*m2delt**2
p0delt = [0.0145, 2*m2delt, s22delt]

#region make figures 
plt.figure()
plt.plot(bc_m, counts_m, label='M')
plt.plot(bc_m, gauss2(bc_m, *parsm), label = 'fit of M')
plt.plot(bc_m, gauss2(bc_m, parsm[0], p0[1], p0[2], parsm[3], p0[4], p0[5]), label = 'fit guess')
#plt.plot(bc_m, gauss2(bc_m, *p0), label = 'fit guess')
plt.grid()
plt.legend()


print('pars guess: ', p0)#
print('pars true: ', parsm)

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



print('pars for delt+m: ', p0delt)

plt.show()


