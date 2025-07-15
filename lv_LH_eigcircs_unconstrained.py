# this is for many runs without final population constraint. 




# libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.stats import linregress
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import random 
from matplotlib.colors import LogNorm
import math

#region Set Variables 

# stuff that gets changed: 
n = 20
runs = 100

zest = 1e-4      # anything smaller = 0

# A matrix 
random.seed(1)
K_set = 0.5
C = 1

# M matrix
# for z=1, have g-f = muc-mua -- required for current version
muc = -0.5
mua = -0.1
f = 1.9
g = 1.5

# stuff that does not get changed:
s = n/2
sigma2 = K_set**2/n*2
t = np.linspace(0, 500, 1000)
x0 = x0_vec(n, 1)

M = M_matrix(n, muc, mua, f, g)
print('n:', n, 'K:', K_set, ', sigma:', '%.3f'%(sigma2**0.5))
mpre_vals, trash = np.linalg.eig(M)
print('M vals:', mpre_vals[0], mpre_vals[1])
One = np.ones(n)

z_calc = (muc-mua+((muc-mua)**2+4*g*f)**0.5) / (2*g)
print('z=', '%0.3f'%z_calc)
R_val = muc+f       # update for z = 1
# endregion variables

#region make arrays 
# A matrix: 
eigs_A = []     # eigenvalues of A 
maxeig_A = []
A_rowsums = []      # rowsums of A 
A_rowsums_stable = []
A_rs_scaled = []
Ars_max = []        # max rowsum of A 
eigs_Ascaled = []
maxeig_Ascaled = []

# M' matrix: M + delta Ars
eigs_Mp = []
eigs_Mp_died = []
eigs_Mp_survived = []
maxeig_Mp = []
eigs_Mp_unscaled = []
eigs_Mp_unscaled_died = []
eigs_Mp_unscaled_survived = []

# Jacobian:
eigs_J = []     # eigenvalues of the jacobian 
eigs_J_died = []        # eigenvalues when not all species survive 
eigs_J_survived = []        # eigenvalues when all species survive 
maxeig_J = []       # max eigenvalue for each run
maxeig_J_complex = []       # max eigenvalue, storing the complex value
maxeig_J_complex_died = []

eigs_Jan = []
maxeig_Jan = []

# other:
n_survives = []     # number of subspecies that survive 
final_abundances = []
final_abundances_stable = []
survive_true = []

total_pops = []

# endregion arrays 


# region loop 
for run in range(runs):

    #region setup and A
    seed = run
    np.random.seed(seed)
    if run %126 == 0:
        print(run)

    # make A matrix 
    A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
    A_classic = A_matrix(s, C, sigma2, seed, LH = 0)
    A_rows = np.dot(A, One)
    A_rowsums.extend(A_rows)
    Ars_max.append(np.max(A_rows))
    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)
    # endregion

    # region analytical 



    # run ODE solver 
    result = lv_LH(x0, t, A, M)         # with scaled M
    xf = result[-1, :]
    #if xf.all() >= 0:
    final_abundances.extend(xf)

    A_rows_scaled = np.dot(A, xf)
    A_rs_scaled.extend(A_rows_scaled)
    A_scaled = np.multiply(np.outer(xf, np.ones(n)), A)
    Asvals, Asvecs = np.linalg.eig(A_scaled)
    eigs_Ascaled.extend(Asvals)

    Mp = M + np.diag(A_rows_scaled)        # mprime = m + delta
    Mpvals, Mpvecs = np.linalg.eig(Mp)
    eigs_Mp.extend(Mpvals)

    Mp_unscaled = M + np.diag(A_rows)        # mprime = m + delta
    Mpvals_unscaled, Mpvecs_unscaled = np.linalg.eig(Mp_unscaled)
    eigs_Mp_unscaled.extend(Mpvals_unscaled)

    Jac = LH_jacobian_norowsum(xf, A, M)
    Jvals, Jvecs = np.linalg.eig(Jac) 
    eigs_J.extend(Jvals)

    n_survive = n
    for species in range(n):
        if xf[species] < 1e-3:
            n_survive -= 1
    n_survives.append(n_survive)
    if n_survive < n:
        eigs_J_died.extend(Jvals)
        eigs_Mp_died.extend(Mpvals)
        eigs_Mp_unscaled_died.extend(Mpvals_unscaled)
        maxeig_J_complex_died.append(np.max(Jvals))
        #print('seed:', seed, 'max real eig:',np.max(Jvals), 'species left:', n_survive)
    elif n_survive == n:
        final_abundances_stable.extend(xf)
        A_rowsums_stable.extend(A_rows)
        eigs_J_survived.extend(Jvals)
        eigs_Mp_survived.extend(Mpvals)
        eigs_Mp_unscaled_survived.extend(Mpvals_unscaled)

    survive_true.append(n_survive==n)

    Avalsm = np.ma.masked_inside(Avals, -zest, zest)      # nonzero eigenvalues of the two, since the zeros dissapear later 
    Mpvalsm = np.ma.masked_inside(Mpvals, -zest, zest)
    Asvalsm = np.ma.masked_inside(Asvals, -zest, zest)

    if -2.7 > np.max(Mpvalsm) > -2.701:
        print('seed:', seed)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Ascaled.append(np.max(np.real(Asvalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))
    maxeig_J_complex.append(np.max(Jvals))

    total_pops.append(np.sum(xf))

print('mean abundance: ', np.mean(final_abundances))

# region analysis 
avg_n_survives = np.average(n_survives)
pct_blk_stable = n_survives.count(n) / runs * 100
print('K:', K_set, 'n:', n, 'average survived:', '%.3f'%avg_n_survives, 'stability', '%.1f'%pct_blk_stable, '% of runs')

# make histograms
nbins = 100
histmin = 0
histmax = np.max(final_abundances)

abun_counts, abun_be = np.histogram(final_abundances, bins=nbins, range = [histmin, histmax])
abun_bc = np.real((abun_be[:-1] + abun_be[1:]) / 2)
abun_stable_counts, abun_stable_be = np.histogram(final_abundances_stable, 
                                                  bins=nbins, range = [histmin, histmax])
abun_stable_bc = np.real((abun_stable_be[:-1] + abun_stable_be[1:]) / 2)

# fitting:
def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma2: variance of Gaussian
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian

p0_abun_stable = [600, mpre_vals[1]/2, 0.003]
#pars_abun_stable, cov_abun_stable = curve_fit(gaussian, abun_stable_counts, abun_stable_bc, p0=p0_abun_stable)
#print('fit pars:', pars_abun_stable)

pars_abun, cov_abun = curve_fit(gaussian, abun_bc[1:], abun_counts[1:], maxfev = 2000)
pars_abun_stable, cov_abun_stable = curve_fit(gaussian, abun_stable_bc[1:], abun_stable_counts[1:], maxfev = 2000)
#pars_abun, cov_abun = curve_fit(gaussian, abun_bc, abun_counts, p0=p0_abun)

fit_abun_Ars_stable = linregress(A_rowsums_stable, final_abundances_stable)
print('linear fit: m=', fit_abun_Ars_stable.slope, ', b=', fit_abun_Ars_stable.intercept)
print('linear fit R^2 = ', fit_abun_Ars_stable.rvalue**2)

fit_abun_Ars = linregress(A_rowsums, final_abundances)
print('linear fit all: m=', fit_abun_Ars.slope, ', b=', fit_abun_Ars.intercept)
print('linear fit all R^2 = ', fit_abun_Ars.rvalue**2)

f_below_0 = 1 - 0.5*(1+math.erf(-pars_abun[1]/((2*pars_abun[2])**0.5)))
predict_stable_frac = f_below_0**s * 100

print('fit pars:', pars_abun)
print('fit covs:', cov_abun)


print('stable fit pars:', pars_abun_stable)
print('stable fit covs:', cov_abun_stable)

# region plot setup 
fsize = (6,6)

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z_calc)+')')
apar_text = str('n='+ str(n)+', K='+str(K_set)+', '+str(runs)+' runs'+', '+str('%.1f'%pct_blk_stable) +'% stable')

stability_text = str(str('%.3f'%pct_blk_stable)+' % stable, predicted '+str('%.3f'%predict_stable_frac)+'%')


# region plotting 

# eigenvalues distribution
plt.figure(figsize=fsize)
plt.plot(np.real(eigs_J_died), np.imag(eigs_J_died), '.', ms = 4, color = 'C0', label = 'J: unstable')
plt.plot(np.real(eigs_Mp_died), np.imag(eigs_Mp_died), '.', ms = 4, color = 'C1', label = "M': unstable")
plt.plot(np.real(eigs_J_survived), np.imag(eigs_J_survived), '.', ms = 4, color = 'C2', label = 'J: stable')
plt.plot(np.real(eigs_Mp_survived), np.imag(eigs_Mp_survived), '.', ms = 8, color = 'C4', label = "M': stable")
#plt.plot(np.real(eigs_Ascaled), np.imag(eigs_Ascaled), '.', ms = 4, color = 'C6', label = "scaled A")
plt.grid()
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('Eigenvalues of J, K='+str(K_set))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)
plt.legend()

# eigenvalues of M' but not scaling
plt.figure(figsize=fsize)
plt.plot(np.real(eigs_Mp_died), 0*np.real(eigs_Mp_died), '.', ms = 4, color = 'C1', label = "M': unstable")
plt.plot(np.real(eigs_Mp_survived), 0*np.real(eigs_Mp_survived), '.', ms = 4, color = 'C4', label = "M': stable")
plt.plot(np.real(eigs_Mp_unscaled_died), 0*np.real(eigs_Mp_unscaled_died)+1, '.', ms = 4, color = 'C2', label = "M' unscaled: unstable")
plt.plot(np.real(eigs_Mp_unscaled_survived), 0*np.real(eigs_Mp_unscaled_survived)+1, '.', ms = 4, color = 'C5', label = "M' unscaled: stable")
plt.grid()
plt.legend()

# max eig vs number left 
plt.figure(figsize=fsize)
plt.plot(n_survives, maxeig_J, '.')
plt.grid()
plt.xlabel('number of surviving species')
plt.ylabel("J max eigenvalue")
plt.title('Max eig of J vs. surviving species')

# M' vs survival 
plt.figure(figsize=fsize)
plt.plot(n_survives, maxeig_Mp, '.')
#plt.plot(n_survives, maxeig_Ascaled, '.')
plt.grid()
plt.xlabel('number of surviving species')
plt.ylabel("M' max eigenvalue")

# M' vs survival 
plt.figure(figsize=fsize)
plt.plot(survive_true, maxeig_Mp, '.')
plt.grid()
plt.xlabel('all survive?')
plt.ylabel("M' max eigenvalue")

# abundance distribution
plt.figure(figsize=fsize)
plt.stairs(abun_counts, abun_be, fill=True, label = 'all runs')
plt.stairs(abun_stable_counts, abun_stable_be, fill=True, label = 'stable runs')
#plt.plot(abun_bc, abun_counts, '.')
#plt.plot(abun_stable_bc, gaussian(abun_stable_bc, *pars_abun_stable), '-k')
#plt.plot(abun_be, gaussian(abun_be, *pars_abun), '-k')
plt.grid()
plt.xlabel('final abundance of species')
plt.ylabel('counts')
plt.title('final abundances from unconstrained cases, K='+str(K_set))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)
plt.figtext(0.13, 0.85, stability_text)
#plt.figtext(0.13, 0.82, str('full fit mean:'+str('%.3f'%pars_abun[1])+', sigma='+str('%.4f'%(pars_abun[2]**0.5))))
plt.figtext(0.13, 0.79, "M eigenvalues: "+str(mpre_vals[0:2]))
plt.legend(loc='upper right')

# abundance vs A rowsum 
plt.figure(figsize=fsize)
plt.plot(A_rowsums, final_abundances, '.', alpha = 0.1,label = 'all runs')
plt.plot(A_rowsums_stable, final_abundances_stable, '.', alpha = 0.1,label = 'stable runs')
#plt.plot(np.array(A_rowsums), fit_abun_Ars_stable.slope*np.array(A_rowsums)+fit_abun_Ars.intercept, '--', label='fit all runs')
#plt.plot(np.array(A_rowsums_stable), fit_abun_Ars_stable.slope*np.array(A_rowsums_stable)+fit_abun_Ars_stable.intercept, '--', label='fit stable runs')
#plt.plot(A_rs_scaled, final_abundances, '.', label = 'scaled Ars')
plt.xlabel('(true) A rowsum')
plt.ylabel('species abundance')
plt.grid()
plt.legend()
#plt.figtext(0.13, 0.65, str('full fit: y='+str('%.3f'%fit_abun_Ars.slope)+'x+'+str('%.3f'%fit_abun_Ars.intercept)+'\nR^2 = '+str('%.3f'%fit_abun_Ars.rvalue**2)))
#plt.figtext(0.13, 0.59, str('stable fit: y='+str('%.3f'%fit_abun_Ars_stable.slope)+'x+'+str('%.3f'%fit_abun_Ars_stable.intercept)+'\nR^2 = '+str('%.3f'%fit_abun_Ars_stable.rvalue**2)))
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)



plt.show()



