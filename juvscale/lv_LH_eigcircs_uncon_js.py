# this is for many runs without final population constraint. 

print('\n')

# region libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.stats import linregress
from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
import random 
from matplotlib.colors import LogNorm
import math
#endregion libraries

#region variables that change  

# stuff that gets changed: 
n = 20
runs = 1000

zest = 1e-5      # anything smaller = 0

# A matrix 
random.seed(1)
K_set = 0.6
C = 1

j = 1

# M matrix
# for z=1, have g-f = muc-mua 
muc = -2
mua = -0.5
f = 1.5
g = 1.2

# endregion

#region variables that dont change

# stuff that does not get changed:
s = int(n/2)
sigma2 = K_set**2/n*2
t = np.linspace(0, 500, 1000)
x0 = x0_vec(n, 1)

M = M_matrix(n, muc, mua, f, g)
print('n:', n, 'K:', K_set, ', sigma:', '%.3f'%(sigma2**0.5))
mpre_vals, trash = np.linalg.eig(M)
print('M vals:', mpre_vals[0], mpre_vals[1])
One = np.ones(n)

z = (muc-mua+((muc-mua)**2+4*g*f)**0.5) / (2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua
Rvec = R_a * np.ones(s) 
print('z =','%.3f'%z, 'R child =', '%.3f'%(R_c/(1+z)), ', R adult =', '%.3f'%(R_a/(1+z)))

# endregion variables

#region make arrays 

# Jacobian:
eigs_J = []     # eigenvalues of the jacobian 
maxeig_J = []       # max eigenvalue for each run

final_abundances_an = []
stable_true = []
# endregion arrays 


# region loop 
for run in range(runs):
    #region setup and A
    seed = run
    np.random.seed(seed)
    if runs > 1000 and run %945 == 0:
        print(run)

    # make A matrix 
    A_classic = A_matrix(s, C, sigma2, seed, LH = 0)
    A = A_matrix_juvscale(n, C, sigma2, seed, j)      #random a matrix
    # endregion

    # region analytical final abundances 
    Aprime = (j*z+1)*A_classic + (j*z-z)*np.identity(s)
    Ap_inv= np.linalg.inv(Aprime)
    xf_an_adult = -np.dot(Ap_inv, Rvec)
    xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
    xf_an[::2] *= z     # scale child 
    final_abundances_an.extend(xf_an)


    # region jacobian
    Jac = LH_jacobian(A, M, xf_an)
    Jvals, trash = np.linalg.eig(Jac)
    eigs_J.extend(Jvals)
    maxeig_J.append(np.max(np.real(Jvals)))
    #print(maxeig_J[-1])
        
    stable_true.append(np.max(np.real(Jvals)) <= 0)


# region analysis 
frac_blk_stable = np.mean(stable_true)
print('K:', K_set, 'n:', n, ', j:', j, ', average survived:', '%.1f'%(frac_blk_stable*100), '% of runs')
print('min abundance: ', np.min(final_abundances_an), ', max abundabce: ', np.max(final_abundances_an))

# make histograms
nbins = 200
abun_an_mean = np.mean(final_abundances_an); abun_an_std = np.std(final_abundances_an)
#histmin = abun_an_mean - 5*abun_an_std; histmax = abun_an_mean + 5*abun_an_std
abmin = -1; abmax = 1.5
#histmin = -10; histmax = 10

abun_an_counts, abun_an_be = np.histogram(final_abundances_an, bins=nbins, range = [abmin, abmax])
abun_an_bc = (abun_an_be[:-1] + abun_an_be[1:]) / 2

# histograms of analytical eigs:
ebins = 100
#emin = np.min(np.real(eigs_J)); emax = np.max(np.real(eigs_J))
emin = -5; emax = 5
Jan_counts, Jan_be = np.histogram(np.real(eigs_J), bins=ebins, range = [emin, emax])


# fit analytical abundances with gaussian
def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma2: variance of Gaussian
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian

p0_abun = [0.9*np.max(abun_an_counts), abun_an_mean, abun_an_std**2]
pars_abun_an, covs_abun_an = curve_fit(gaussian, abun_an_bc, abun_an_counts, p0_abun)

#print('pars guess: \n', p0_abun)
print('pars fit: \n', pars_abun_an)
# endregion analysis

# region plot setup 
fsize = (6,6)

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', K='+str(K_set)+', '+str(runs)+' runs'+', '+str('%.1f'%(frac_blk_stable*100)) +'% stable')

stability_text = str(str('%.3f'%(frac_blk_stable*100))+' % stable') #, predicted '+str('%.3f'%predict_stable_frac)+'%')

# endregion plot setup 

# region plotting 

# eigenvalues distribution
plt.figure(figsize=fsize)
plt.plot(np.real(eigs_J), np.imag(eigs_J), '.', ms = 4, color = 'C2', label = 'J eigenvalues')
plt.grid()
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('Eigenvalues of J, K='+str(K_set)+', j='+str(j))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)
plt.legend()
plt.xlim(-6.5, 2.5)
plt.ylim(-0.7, 0.7)

# histograms of real eigenvalues, for analytical solutions 
plt.figure(figsize=fsize)
plt.stairs(Jan_counts, Jan_be, fill=True, alpha = 0.7, label = 'Jacobian')
#plt.stairs(Aan_counts, Aan_be, fill=True, alpha = 0.5, label = 'A (scaled)')
plt.xlabel('real eigenvalue component')
plt.ylabel('counts')
plt.title('Real components of Analytical Jacobian Eigenvalues')
plt.grid()
plt.legend()


# abundance distribution
plt.figure(figsize=fsize)
plt.stairs(abun_an_counts, abun_an_be, fill=True, alpha = 0.7, label = 'analytical solutions')
plt.plot(abun_an_bc, gaussian(abun_an_bc, *pars_abun_an), '-', label = 'fit of analytical')
plt.grid()
plt.xlabel('final abundance of species')
plt.ylabel('counts')
plt.title('final abundances from unconstrained cases, K='+str(K_set))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)
plt.figtext(0.13, 0.85, stability_text)
#plt.figtext(0.13, 0.79, "M eigenvalues: "+str(mpre_vals[0:2]))
plt.figtext(0.13, 0.79, 'Fit mean:'+str('%0.3f'%(pars_abun_an[1]))+', fit std.dev: '+str('%0.3f'%(pars_abun_an[2]**0.5)))
plt.legend(loc='upper right')
#plt.xlim(-1, 1.5)

'''# abundance vs A rowsum 
plt.figure(figsize=fsize)
plt.plot(A_rowsums, final_abundances_an, '.', alpha = 0.1)
plt.xlabel('(true) A rowsum')
plt.ylabel('species abundance')
plt.grid()
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)

# species stable vs max eig of J
plt.figure(figsize=fsize)
plt.plot(maxeig_J, n_survives, '.', label='J')
plt.plot(maxeig_A, n_survives+0.1*np.ones(runs), '.', label='A')
plt.plot(maxeig_Axs, n_survives+0.2*np.ones(runs), '.', label='A scaled')
plt.plot(maxeig_A_cl, n_survives+0.3*np.ones(runs), '.', label='A classical')
plt.plot(maxeig_Axs_cl, n_survives+0.4*np.ones(runs), '.', label='A cl scaled')
plt.grid()
plt.xlabel('max real eig')
plt.ylabel('number of surviving species (simulated)')
plt.legend()

# max eigenvalue of A - classic vs LH
plt.figure(figsize=fsize)
plt.plot(maxeig_Axs_cl, maxeig_Axs, 'o', alpha =0.6, label = 'scaled')
plt.plot(maxeig_A_cl, maxeig_A, 'o', alpha =0.6, label = 'unscaled')
plt.legend()
plt.xlabel('classical A')
plt.ylabel('LH A')
plt.title('Max eigenvalue of A - classical vs LH')
plt.grid()

# max eigenvalue of A - scaled vs unscaled 
plt.figure(figsize=fsize)
plt.plot(maxeig_A, maxeig_Axs, 'o', alpha =0.6, label = 'classical')
plt.plot(maxeig_A_cl, maxeig_Axs_cl, 'o', alpha =0.6, label = 'LH')
plt.legend()
plt.xlabel('unscaled A')
plt.ylabel('scaled A')
plt.title('Max eigenvalue of A - unscaled vs scaled ')
plt.grid()
'''







plt.show()

print('\n')

