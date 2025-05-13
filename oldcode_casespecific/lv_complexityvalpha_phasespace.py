#region preamble
# For different values of f and sigma, calculate system stability. 
# See if there is interesting divide in the phase space caused by f, or if it is 
# just similar to the classic case. 

#endregion 

#region functions 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import x0_vec
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum

import random 
#endregion


n = 20      # number of species

#region set up variables 
C = 1       # connectedness 
x0 = x0_vec(n)      # initial populations
t = np.linspace(0, 100, 2000)

# values for m
muc = -0.5
mua = -0.5
g = 1
f = 1.5

# lists to append: 
zs_stable = []     # values of f used in M matrix
Ks_stable = []     # complexity of the classical (n/2) A matrix
zs_unstable = []
Ks_unstable = []
zs_zeropop = []
Ks_zeropop = []
Ks_stable_actual = []
Ks_unstable_actual = []
Ks_zeropop_actual = []
maxeig = []

actbigger = 0       # counts how many times the actual complexity is higher than set
One = np.ones(n)        # useful for rowsums 
#endregion

#region set up conditions 
#seed = random.randint(1, 1000)        # for now keeps every A the same but scaled (I think)
seed = 2
print('seed: ', seed)
runs = 500     # number of scenarios to run
xstar = 1      # do you want all of the final populations to be the same?
zmin = 0.5;    zmax = 1.5
Kmin = 0;     Kmax = 4

xs = np.ones(n)


#endregion



for run in range(runs):
    if run%20 == 0:
        print(run)
    random.seed(run)
    z = random.uniform(zmin, zmax)
    K = random.uniform(Kmin, Kmax)

    for i in range(0, n, 2):
        xs[i] = z

    #K = 0.7
    #print('f:', f, ', K:', K)

    sigma2 = K**2/(n/2)     # sigma to give complexity K in the classical case 
    #print('set std:', sigma2, ', Set K:', K)

    #region make matrices
    if seed != 0:
        A = A_matrix(n, C, sigma2, seed, LH=1)
    elif seed == 0: 
        A = A_matrix(n, C, sigma2, random.randint(1, 1000), LH=1)
    #print(A)
    M = M_matrix(n, muc, mua, f, g)
    # scale to have identical rowsums 
    A_rowsums = np.dot(A, One)
    M_rowsums = np.dot(M, One)
    if xstar == 1:
        scales = -np.divide(A_rowsums, M_rowsums)
        M = np.multiply(M, np.outer(scales, One))
    #endregion

    #region calculate actual stability
    '''A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
    #print(A_classic)
    var_act = np.var(np.ma.masked_equal(A, -1), ddof = 1)
    K_actual = (var_act*(n/2))**0.5
    if K_actual > K:
        actbigger += 1
    #print('std act:', std_act, ', actual K:', K_actual)'''
    #endregion

    #region run ODE
    result = lv_LH(x0, t, A, M)
    xf = result[-1, :]
    #endregion

    #region determine stability
    if xstar == 1:
        Jac = LH_jacobian(n, A, M, xs)
    elif xstar == 0:
        Jac = LH_jacobian_norowsum(xf, A, M)
    Jvals, Jvecs = np.linalg.eig(Jac)
    maxeig = np.max(np.real(Jvals))

    if maxeig <= 0:
        zs_stable.append(z)
        Ks_stable.append(K)
        #Ks_stable_actual.append(K_actual)
    elif maxeig > 0:
        zs_unstable.append(z)
        Ks_unstable.append(K)
        #Ks_unstable_actual.append(K_actual)
    
    if np.min(xf) < 1e-5:
        zs_zeropop.append(z)
        Ks_zeropop.append(K)
        #Ks_zeropop_actual.append(K_actual)
    #endregion


print('how many times was teh actual K bigger:', actbigger)
#region figure constants 
mss = 4
if seed != 0:   
    plot_title = str('stability for K, z: '+str(n)+' species, A seed = ' +str(seed))
    plot_title_2 = str('stability for K, z actaul: '+str(n)+' species, A seed = ' +str(seed))
else:
    plot_title = str('stability for K, z: '+str(n)+' species, random As')
    plot_title_2 = str('stability for K, z actual: '+str(n)+' species, random As')

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $g =$'+str(g)+', $f =$'+str(f))
#endregion

#region make figures 
# K vs z
plt.figure(figsize=(8, 6))    # K vs f
plt.plot(zs_unstable, Ks_unstable, 'ob', ms = mss, alpha = 0.5, label = 'unstable scenarios')
plt.plot(zs_zeropop, Ks_zeropop, 'or', ms = mss, alpha = 0.5, label = 'die off of at least 1 species')
plt.plot(zs_stable, Ks_stable, 'o', color='limegreen', ms = mss, alpha = 1, label = 'stable scenarios')
plt.figtext(0.13, 0.12, plot_text)
plt.legend(bbox_to_anchor=(1, 1),  loc='upper right')       
plt.grid()
plt.title(plot_title)
plt.xlabel('z = juvinile fraction')
plt.ylabel('(classical) complexity K')
plt.xlim(zmin-0.4, zmax+0.4)
plt.ylim(Kmin-0.4, Kmax+0.4)


print('Max stable K: ', np.max(Ks_stable))

# actual K vs f
'''plt.figure(figsize=(8, 6))    # K vs f
plt.plot(fs_unstable, Ks_unstable_actual, 'ob', ms = mss, alpha = 0.5, label = 'unstable scenarios')
plt.plot(fs_zeropop, Ks_zeropop_actual, 'or', ms = mss, alpha = 0.5, label = 'die off of at least 1 species')
plt.plot(fs_stable, Ks_stable_actual, 'o', color='limegreen', ms = mss, alpha = 1, label = 'stable scenarios')
plt.figtext(0.13, 0.12, plot_text)
plt.legend(bbox_to_anchor=(1, 1),  loc='upper right')       
plt.grid()
plt.title(plot_title_2)
plt.xlabel('f')
plt.ylabel('(classical) complexity K')
plt.xlim(fmin-0.4, fmax+0.4)
plt.ylim(Kmin-0.4, Kmax+0.4)'''


plt.show()
















