
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

seed =  18 #24 18 658
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
g = 1

p = 40 # number of percentage values run 


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
mvals0, mprevecs = np.linalg.eig(M_pre)

# region run unconstrained case 
result_un = lv_LH(x0, t, A, M_pre)
xf_un = result_un[-1,:]

z = float(xf_un[0]/xf_un[1])
z = 1
xs_final = np.ones(n) 
for i in range(0,n,2):
    xs_final[i] = z
print(xs_final)

diff_from_set = xs_final - xf_un


species_left_un = 0
for i in range(n):
    if xf_un[i] > 1e-3:
        species_left_un += 1 

Jac_pre = LH_jacobian_norowsum(xf_un, A, M_pre)
Mprime = M_pre + np.dot(A, xf_un)

Jvals0, jvecs = np.linalg.eig(Jac_pre)
Mpvals0, mvecs = np.linalg.eig(Mprime)

#print('final pops:\n', xf_un[:]) 
print('Species remain:', species_left_un, ', max rowsum:', A_rs_max)
print('Juvinile fraction:', xf_un[1]/xf_un[0])



pcts = np.linspace(0, 1, p)

max_eigs_J = []
max2_eigs_J = []
max3_eigs_J = []
maxcomplex_eigs_J = []
maxnotMp_eigs_J = []
max_eigs_Mp = []
max_eig_diff = []
min_eig_diff = []
max_scaled_rs = []
species = []

# region adjust toward constraint 
pos = 0
match = 0

snapcount = 0

pct_unphys = []
maxeig_unphys = []
zvals = []
pcts_wz = []

for pct in pcts:

    xs = xf_un + pct*diff_from_set
    #print('on ', pct, ' percent')
    #print(xs)

    A_rows_scaled = np.dot(A, xs)
    M_rows_scaled = np.dot(M_pre, xs)
    scales = -np.divide(np.multiply(A_rows_scaled, xs), M_rows_scaled)
    M = np.multiply(M_pre, np.outer(scales, np.ones(n)))       # newly scaled M
    max_scaled_rs.append(np.max(A_rows_scaled))

    Mprime = M + np.diag(A_rows_scaled)
    Mp_evals, Mp_evecs = np.linalg.eig(Mprime)
    Mp_eigs_mask = np.ma.masked_inside(Mp_evals, -1e-10, 1e-10)
    max_eigs_Mp.append(np.max(np.real(Mp_eigs_mask)))

    
    Jac = LH_jacobian(n, A, M, xs) 
    Jvals, Jvecs = np.linalg.eig(Jac)
    max_eig_J = np.max(np.real(Jvals))
    max_eigs_J.append(max_eig_J)
    max_eig_diff.append(max_eig_J-np.max(np.real(Mp_eigs_mask)))
    min_eig_diff.append(np.min(np.real(Jvals))-np.min(np.real(Mp_eigs_mask)))

    #print(max_eig_J)
    #print(np.real(Jvals))
    #print(np.ma.masked_equal(np.real(Jvals), max_eig_J))
    masked_biggest = np.ma.masked_equal(np.real(Jvals), max_eig_J)
    biggest2 = np.max(masked_biggest)
    masked_2biggest = np.ma.masked_equal(masked_biggest, biggest2)
    biggest3 = np.max(masked_2biggest)
    max2_eigs_J.append(biggest2)
    max3_eigs_J.append(biggest3)

    complex_Jvals = np.ma.masked_equal(Jvals, np.real(Jvals))
    maxcomplex_eigs_J.append(np.max(np.real(complex_Jvals)))



    if np.max(np.diag(M)) > 0 and pos == 0: 
        #print('pct ', pct, ', M has a positive diagonal.')
        pct_unphys.append(pct)
        maxeig_unphys.append(np.max(np.real(Jvals)))
        #pos = 1
        #print(M)
    #if np.min(np.diag(M, 1)) < 0: print('M has a negative f value.')
    #mif np.min(np.diag(M, -1)) < 0: print('M has a negative g value.')

    if np.max(np.real(Jvals)) - np.max(np.real(Mp_eigs_mask)) < 1e-5 and match == 0:
        print('pct', pct,', eigs match')
        match = 1 

    result = lv_LH(x0, t, A, M)
    xf = result[-1, :]
    #print(xf)
    species_left = 0
    #print('first 2:', xf[0], xf[1])
    for i in range(n):
        if xf[i] > 1e-3:
            species_left += 1 
        if i%2 == 0:
            zvals.append(float(xf[i]/xf[i+1]))
            pcts_wz.append(pct)
        #print(zvals[-10:])
    
    species.append(species_left)

    # save eigenvalue snapshots
    if pct >= 0.2 and snapcount < 1: 
        Jvals2, jvecs = np.linalg.eig(Jac)
        Mpvals2, mpvecs = np.linalg.eig(Mprime)
        pct2 = pct
        snapcount += 1
    if pct >= 0.4 and snapcount < 2: 
        Jvals4, jvecs = np.linalg.eig(Jac)
        Mpvals4, mpvecs = np.linalg.eig(Mprime)
        pct4 = pct
        snapcount += 1
    if pct >= 0.6 and snapcount < 3: 
        Jvals6, jvecs = np.linalg.eig(Jac)
        Mpvals6, mpvecs = np.linalg.eig(Mprime)
        pct6 = pct
        snapcount += 1
    if pct >= 0.8 and snapcount < 4: 
        Jvals8, jvecs = np.linalg.eig(Jac)
        Mpvals8, mpvecs = np.linalg.eig(Mprime)
        pct8 = pct
        snapcount += 1
    if pct == 1: 
        Jvals10, jvecs = np.linalg.eig(Jac)
        Mpvals10, mpvecs = np.linalg.eig(Mprime)
        snapcount += 1


# region plot 
plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+
                ', A seed ='+str(seed)+ ', K='+str('%.3f'%K))

fsize = (6, 6)

# region plot snapshots 
plt.figure(figsize = fsize)
plt.plot(np.real(Jvals0), np.imag(Jvals0), 'o', label = 'J')
plt.plot(np.real(Mpvals0), np.imag(Mpvals0), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at fractional shift of 0')
plt.legend()

plt.figure(figsize = fsize)
plt.plot(np.real(Jvals2), np.imag(Jvals2), 'o', label = 'J')
plt.plot(np.real(Mpvals2), np.imag(Mpvals2), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at fractional shift of '+str(pct2))
plt.legend()

plt.figure(figsize = fsize)
plt.plot(np.real(Jvals4), np.imag(Jvals4), 'o', label = 'J')
plt.plot(np.real(Mpvals4), np.imag(Mpvals4), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at fractional shift of '+str(pct4))
plt.legend()

plt.figure(figsize = fsize)
plt.plot(np.real(Jvals6), np.imag(Jvals6), 'o', label = 'J')
plt.plot(np.real(Mpvals6), np.imag(Mpvals6), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at fractional shift of '+str(pct6))
plt.legend()

plt.figure(figsize = fsize)
plt.plot(np.real(Jvals8), np.imag(Jvals8), 'o', label = 'J')
plt.plot(np.real(Mpvals8), np.imag(Mpvals8), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at fractional shift of '+str(pct8))
plt.legend()

plt.figure(figsize = fsize)
plt.plot(np.real(Jvals10), np.imag(Jvals10), 'o', label = 'J')
plt.plot(np.real(Mpvals10), np.imag(Mpvals10), 'o', mfc = 'none', label = "M'")
plt.grid()
plt.xlim(-14.5, 1)
plt.ylim(-.5, .5)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalue spectra at full constraint')
plt.legend()




# region overall plots

plt.figure(figsize = fsize)
plt.plot(pcts, species, 'o')
plt.grid()
plt.xlabel('Percentage shift toward 1')
plt.ylabel('Number of species survived')

plt.figure()
plt.plot(pcts, max_eigs_J, '.--', label = 'Jacobian')
plt.plot(pcts, max2_eigs_J, '.--', label = 'J2')
plt.plot(pcts, max3_eigs_J, '.--', label = 'J3')
plt.plot(pcts, maxcomplex_eigs_J, '.--', label = 'J compelx eig')
plt.plot(pcts, max_eigs_Mp, '--', mfc = 'none', label = 'M prime')
plt.plot(pcts, max_eigs_A, '--', label = 'A')
#plt.plot(pcts, max_scaled_rs, '--', label = 'max scaled A rowsum')
#plt.plot(pct_unphys, maxeig_unphys, '.', color='fuchsia', label= 'unphysical M points')
plt.grid()
plt.xlabel('Percentage shift toward 1')
plt.ylabel('Max eigenvalue')
plt.title('Eigenvalues as constraint is shifted from unconstrained to 1')
plt.figtext(0.13, 0.12, plot_text)
plt.legend()


plt.figure()
plt.plot(pcts, max_eig_diff, '.--', label = "max eig diff")
plt.plot(pcts, min_eig_diff, '.--', label = "min eig diff")
plt.xlabel('Percentage shift toward 1')
plt.ylabel('Difference in max eigenvalues')
plt.title("Difference in min/max real eig of J and M' (J-M')")
plt.figtext(0.13, 0.12, plot_text)
plt.legend()
plt.grid()

# juvinile fractions
plt.figure(figsize=fsize)
plt.plot(pcts_wz, zvals, '.')
plt.grid()
plt.xlabel('fractional shift')
plt.ylabel('juvinile fraction')
plt.ylim(0.8, 1.5)




plt.show()
