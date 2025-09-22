# Run many runs with a certain K, find percent stable using analytical 
# method for each value of K to create S curve. 
print ('\n')
# region libraries 
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
from lv_functions import classic_jacobian
import random 
from matplotlib.colors import LogNorm
import math
import time
# endregion libraries 

#region variables to set 

n = 10
runs_per_K = 1000
num_Ks = 30
Kmin = 0.2; Kmax = 1.5

# M matrix
muc = -0.1
mua = -0.05
f = 0.15
g = 0.1



#endregion set

#region variables that dont change 
s = n
One = np.ones(n)
Ks = np.linspace(Kmin, Kmax, num_Ks)
#Ks = np.linspace(0.3, 0.55, 15)
C = 1 # complexity 

M = M_matrix(n, muc, mua, f, g)
z = (muc-mua+((muc-mua)**2+4*g*f)**0.5) / (2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua        # should match
Rvec = R_a/(1+z) * np.ones(s) 
print('n:', n, ', runs per K: ', runs_per_K)
print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a, '\n')

# endregion variables

#region checks
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if abs(R_c-R_a) > 1e-10:
    raise Exception("error calculating R values.")

#endregion checks

stable_fracs = []

start = time.time()
# region loop for Ks
for K in Ks:
    sigma2 = K**2 / s / C
    isstable = []   # store stability per run to average later 
    for run in range(runs_per_K):
        # make A:
        A_classic = A_matrix(s, C, sigma2, run, LH = 0) 
        # caluclate final abundances: 
        A_inv = np.linalg.inv(A_classic)
        xf = -np.dot(A_inv, Rvec)
        # calculate Jacobian:
        Jac = classic_jacobian(A_classic, xf)
        #check stability: 
        Jvals, scrap = np.linalg.eig(Jac)
        max_eig = np.max(np.real(Jvals))
        if max_eig <= 0: isstable.append(1)
        else: isstable.append(0)
    # calculate stable fraction
    frac = np.mean(isstable)
    stable_fracs.append(frac)
    end = time.time()
    Ktime = end-start

    print('K: ', '%0.3f'%K, ', ', '%0.3f'%(frac*100), '% stable, took', '%0.3f'%(Ktime*1000), 'ms')
    start = time.time()

#endregion loop

#region analysis/fitting 

# estimate 50%:
for i in range(len(Ks)):
    if stable_fracs[i] > 0.5: ind_bef = i
shlope = (stable_fracs[ind_bef]-stable_fracs[ind_bef+1])/(Ks[ind_bef]-Ks[ind_bef+1])
K_50 = (0.5-stable_fracs[ind_bef])/shlope + Ks[ind_bef]

print('\nK of 0.5: ', K_50)
print('sigma of 0.5: ', K_50**2 / s / C)


#region plotting setup 
mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', '+str(runs_per_K)+' runs per K value')
K50_text = str('f=0.5 at at K='+str('%0.3f'%K_50)+'\nwith slope '+str('%0.3f'%shlope)+' / K')
fsize = (6,6)

# region plotting 
plt.figure(figsize = fsize)
plt.plot(Ks, 0.5*np.ones(len(Ks)), '--k', alpha = 0.4)
plt.plot(Ks, stable_fracs)
plt.grid()
plt.xlabel('complexity K')
plt.ylabel('fraction of runs stable')
plt.title('1-stage Stablility by K, n='+str(n)+' (s='+str(s)+')')
plt.figtext(0.13, 0.12, mpar_text)
plt.figtext(0.13, 0.15, apar_text)
plt.figtext(0.5, 0.8, K50_text, fontsize = 14)
plt.ylim(-0.1, 1.1)
plt.xlim(Kmin, Kmax)

plt.show()

print('\n')



