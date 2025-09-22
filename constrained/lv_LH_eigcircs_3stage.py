# this is for many runs without final population constraint. 

print('\n')

# region libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.stats import linregress
from lv_functions import A_matrix
from lv_functions import A_matrix3
from lv_functions import M_matrix3
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
import random 
from matplotlib.colors import LogNorm
import math
#endregion libraries

#region variables that change  

# stuff that gets changed: 
n = 21
runs = 500

zest = 1e-4      # anything smaller = 0

# A matrix 
random.seed(1)
K_set = 0.57735
C = 1

# M matrix 
mu1 = -0.5; mu2 = -0.6; mu3 = -0.7
f12 = 1.4; f13 = 1.5; f23 = 1.6
g21 = 0.9; g31 = 1; g32 = 1.1

z1 = 1.3098610680199345
z2 = 1.1319313094443162


# endregion

#region variables that dont change

# stuff that does not get changed:
s = int(n/3)
sigma2 = K_set**2/s/C
t = np.linspace(0, 500, 1000)
x0 = x0_vec(n, 1)

M = M_matrix3(n, mu1, mu2, mu3, f12, f13, f23, g21, g31, g32)
print('n:', n, 'K:', K_set, ', sigma:', '%.3f'%(sigma2**0.5))
print('LH K: ', K_set*3**0.5)
mpre_vals, trash = np.linalg.eig(M)
print('M vals:', mpre_vals[0], mpre_vals[1], mpre_vals[2])
One = np.ones(n)

R = g31*z1+g32*z2+mu3
Rvec = R/(1+z1+z2) * np.ones(s)
print('R:', R)

zvec = []
for i in range(s):
    zvec.extend([z1, z2, 1])
# endregion variables

#region make arrays 
# A matrix: 
eigs_A = []     # eigenvalues of A 
maxeig_A = []
A_rowsums = []      # rowsums of A 
Arowsums_max = []        # max rowsum of A 

eigs_A_cl = []
maxeig_A_cl = []

eigs_Axs = []
maxeig_Axs = []
A_rowsums_xs = []
eigs_Axs_cl = []
maxeig_Axs_cl = []
A_rowsums_xs_cl = []

# M' matrix: M + delta Ars
eigs_Mp = []
maxeig_Mp = []

# Jacobian:
eigs_J = []     # eigenvalues of the jacobian 
eigs_J_died = []        # eigenvalues when not all species survive 
eigs_J_survived = []        # eigenvalues when all species survive 
maxeig_J = []       # max eigenvalue for each run

# final state: 
final_abundances_an = []
n_survives = []     # number of subspecies that survive - mostly a check 
final_abundances_num = []
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
    Avals_cl, trash = np.linalg.eig(A_classic)
    eigs_A_cl.extend(Avals_cl)
    maxeig_A_cl.append(np.max(np.real(Avals_cl)))

    A = A_matrix3(n, C, sigma2, seed)      #random a matrix 
    A_row = np.dot(A, One)
    A_rowsums.extend(A_row)
    Avals, trash = np.linalg.eig(A)
    eigs_A.extend(Avals)
    maxeig_A.append(np.max(np.ma.masked_inside(np.real(Avals), -zest, zest)))
    # endregion

    # region analytical final abundances 
    A_inv = np.linalg.inv(A_classic)
    xf_an_adult = -np.dot(A_inv, Rvec)
    xf_an = np.repeat(xf_an_adult, 3)   # make unscaled
    xf_an = np.multiply(xf_an, zvec)
    final_abundances_an.extend(xf_an)

    A_clxs = np.multiply(np.outer(xf_an_adult, np.ones(s)), A_classic)
    Avals_clxs, trash = np.linalg.eig(A_clxs)
    maxeig_Axs_cl.append(np.max(np.real(np.ma.masked_inside(Avals_clxs, -zest, zest))))
    
    A_row_xs_cl = np.dot(A_classic, xf_an_adult)
    A_rowsums_xs_cl.extend(A_row_xs_cl)

    A_LHxs = np.multiply(np.outer(xf_an, np.ones(n)), A)
    Avals_xs, trash = np.linalg.eig(A_LHxs)
    maxeig_Axs.append(np.max(np.real(np.ma.masked_inside(Avals_xs, -zest, zest))))
    
    A_row_xs = np.dot(A, xf_an)
    A_rowsums_xs.extend(A_row_xs)


    # region jacobian
    Jac = LH_jacobian(A, M, xf_an)
    Jvals, trash = np.linalg.eig(Jac)
    eigs_J.extend(Jvals)
    maxeig_J.append(np.max(np.real(Jvals)))

    Mp = M + np.diag(A_row_xs)
    Mpvals, trash = np.linalg.eig(Mp)
    eigs_Mp.extend(Mpvals)

    #region numerical solution
    # run ODE solver 
    result = lv_LH(x0, t, A, M)         # with scaled M
    xf_num = result[-1, :]
    #if xf.all() >= 0:
    final_abundances_num.extend(xf_num)
    

    n_survive = n
    for species in range(n):
        if xf_an[species] < zest:
            n_survive -= 1
    n_survives.append(n_survive)
    if n_survive < n:
        eigs_J_died.extend(Jvals)
    elif n_survive == n:
        eigs_J_survived.extend(Jvals)
        
    stable_true.append(n_survive==n)


# region analysis 
avg_n_survives = np.average(n_survives)
frac_blk_stable = np.mean(stable_true)
print('K:', K_set, 'n:', n, 'average survived:', '%.3f'%avg_n_survives, 'stability', '%.1f'%(frac_blk_stable*100), '% of runs')
print('min abundance: ', np.min(final_abundances_an), ', max abundabce: ', np.max(final_abundances_an))

# make histograms
nbins = 200
abun_an_mean = np.mean(final_abundances_an); abun_an_std = np.std(final_abundances_an)
#histmin = abun_an_mean - 5*abun_an_std; histmax = abun_an_mean + 5*abun_an_std
abmin = -2; abmax = 3
#histmin = -10; histmax = 10

abun_num_counts, abun_num_be = np.histogram(final_abundances_num, bins=nbins, range = [abmin, abmax])
abun_num_bc = np.real((abun_num_be[:-1] + abun_num_be[1:]) / 2)

abun_an_counts, abun_an_be = np.histogram(final_abundances_an, bins=nbins, range = [abmin, abmax])
abun_an_bc = (abun_an_be[:-1] + abun_an_be[1:]) / 2

# histograms of analytical eigs:
ebins = 100
#emin = np.min(np.real(eigs_J)); emax = np.max(np.real(eigs_J))
emin = -5; emax = 5
Jan_counts, Jan_be = np.histogram(np.real(eigs_J), bins=ebins, range = [emin, emax])
Mpan_counts, Mpan_be = np.histogram(np.real(eigs_Mp), bins=ebins, range = [emin, emax])
Aan_counts, Aan_be = np.histogram(np.real(eigs_A), bins=ebins, range = [emin, emax])

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
#pars_abun_an, covs_abun_an = curve_fit(gaussian, abun_an_bc, abun_an_counts, p0_abun)

#print('pars guess: \n', p0_abun)
#print('pars fit: \n', pars_abun_an)
# endregion analysis

# region plot setup 
fsize = (6,6)

mpar_text = str('$\u03bc_1 =$'+str(mu1)+', $\u03bc_1 =$'+str(mu2)+', $\u03bc_1 =$'+str(mu3)+
                ', \n$f12=$'+str(f12)+', $f13=$'+str(f13)+', $f23=$'+str(f23)+
                ', \n$g21 =$'+str(g21)+', $g31 =$'+str(g21)+', $g32 =$'+str(g32)+
                ' (z1='+str('%.2f'%z1)+', z2='+str('%.2f'%z2)+')')
apar_text = str('n='+ str(n)+', K='+str(K_set)+', '+str(runs)+' runs'+', '+str('%.1f'%(frac_blk_stable*100)) +'% stable')

stability_text = str(str('%.3f'%(frac_blk_stable*100))+' % stable') #, predicted '+str('%.3f'%predict_stable_frac)+'%')

# endregion plot setup 

# region plotting 

# numerical eigenvalues distribution
plt.figure(figsize=fsize)
plt.plot(np.real(eigs_J_survived), np.imag(eigs_J_survived), '.', ms = 4, color = 'C2', label = 'J: stable')
plt.plot(np.real(eigs_J_died), np.imag(eigs_J_died), '.', ms = 4, color = 'C0', label = 'J: unstable')
plt.plot(np.real(eigs_Mp), np.imag(eigs_Mp), '.', ms = 4, color = 'C1', label = "M")
plt.grid()
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('Eigenvalues of J, K='+str(K_set))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.21, apar_text)
plt.legend()

# histograms of real eigenvalues, for analytical solutions 
plt.figure(figsize=fsize)
plt.stairs(Jan_counts, Jan_be, fill=True, alpha = 0.7, label = 'Jacobian')
plt.stairs(Mpan_counts, Mpan_be, fill=True, alpha = 0.5, label = "M'")
#plt.stairs(Aan_counts, Aan_be, fill=True, alpha = 0.5, label = 'A (scaled)')
plt.xlabel('real eigenvalue component')
plt.ylabel('counts')
plt.title('Real components of Analytical Jacobian Eigenvalues')
plt.grid()
plt.legend()


# abundance distribution
plt.figure(figsize=fsize)
plt.stairs(abun_an_counts, abun_an_be, fill=True, alpha = 0.7, label = 'analytical solutions')
plt.stairs(abun_num_counts, abun_num_be, fill=True, alpha = 0.7, label = 'numerical solutions')
#plt.plot(abun_an_bc, gaussian(abun_an_bc, *pars_abun_an), '-', label = 'fit of analytical')
plt.grid()
plt.xlabel('final abundance of species')
plt.ylabel('counts')
plt.title('final abundances from unconstrained cases, K='+str(K_set))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.21, apar_text)
plt.figtext(0.13, 0.85, stability_text)
#plt.figtext(0.13, 0.79, "M eigenvalues: "+str(mpre_vals[0:2]))
#plt.figtext(0.13, 0.79, 'Fit mean:'+str('%0.3f'%(pars_abun_an[1]))+', fit std.dev: '+str('%0.3f'%(pars_abun_an[2]**0.5)))
plt.legend(loc='upper right')
#plt.xlim(-1, 1.5)

# abundance vs A rowsum 
plt.figure(figsize=fsize)
plt.plot(A_rowsums, final_abundances_an, '.', alpha = 0.1)
plt.xlabel('(true) A rowsum')
plt.ylabel('species abundance')
plt.grid()
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.21, apar_text)

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






plt.show()

print('\n')

