
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math
from scipy.optimize import curve_fit

seed = random.randint(0, 1000)
seed = 922
print('\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 2000)
K_set = 0.5
C = 1

mua = -0.5
f = 1.5
g = 1.2

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
print('sigma2:', sigma2)
K = (sigma2*C*s)**0.5

zest = 1e-5         # anything less is zero



A_classic = A_matrix(s, C, sigma2, seed, LH = 0)
# endregion set variables


#region loop thru muc values
nmucs = 10
njs = 21
mucs = np.linspace(-0.1, -3, nmucs)
js = np.linspace(0, 2, njs)
zs = []

maxeig_mubyj = np.empty((nmucs, njs))

for i in range(nmucs):
    muc = mucs[i]
    z = (muc-mua+((muc-mua)**2 +4*g*f)**0.5)/(2*g)
    zs.append(z)
    R_c = (z*muc+f)/z; R_a = z*g+mua
    Rvec = R_a * np.ones(s)   # M part of equilibrium equation
    print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a)

    # for m matrix:
    M = M_matrix(n, muc, mua, f, g)
    mvals, mvecs = np.linalg.eig(M)


    #region loop thru js
    for k in range(njs): 
        j = js[k]
        A = A_matrix_juvscale(n, C, sigma2, seed, j)
        Aprime = (j*z+1)*A_classic + (j*z-z)*np.identity(s)
        Ap_inv = np.linalg.inv(Aprime)
        xf_an_adult = -np.dot(Ap_inv, Rvec)
        xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
        xf_an[::2] *= z     # scale child 

        Jac = LH_jacobian(A, M, xf_an)
        Jvals, Jvecs = np.linalg.eig(Jac)
        maxeig_mubyj[i,k] = np.max(np.real(Jvals))

    #endregion jloop

#endregion muloop

#region get slopes! 
def linear(x, m, b):
    return x*m + b\
    
slopes = []
intercepts = []
for l in range(nmucs):
    pars, covs = curve_fit(linear, js, maxeig_mubyj[l, :])
    slopes.append(pars[0])
    intercepts.append(pars[1])


mpar_text = str('$\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g))
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))
#endregion slopes 

#region get transition mu/z
eig0_js = []
for l in range(nmucs):
    ind = 0
    for k in range(njs):
        if maxeig_mubyj[l, k] < 0: ind = k
    e1 = maxeig_mubyj[l, ind]; e2 = maxeig_mubyj[l, ind+1]
    j1 = js[ind]; j2 = js[ind+1]
    eig0_j = j1 - e1*(j2-j1)/(e2-e1)
    eig0_js.append(eig0_j)
    




# Max eigenvalue vs muc, for different j
plt.figure()
for l in range(int(nmucs/2)):
    plt.plot(js, maxeig_mubyj[l*2, :], '--.', label='muc='+str('%.2f'%mucs[l*2])+', z='+str('%.2f'%zs[l*2]))
plt.grid()
plt.xlabel('j value')
plt.ylabel('max real eigenvalue of Jacobian')
plt.legend(loc='lower right')
plt.title('max J eigenvalue by j, for changing mu_c')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)


plt.figure()
plt.plot(mucs, slopes, '--.', label='slopes')
plt.plot(mucs, intercepts, '--.', label='intercepts')
plt.grid()
plt.legend()

plt.figure()
plt.plot(mucs, eig0_js, '--.')
plt.plot(zs, eig0_js, '--.')
plt.xlabel('mu_c')
plt.ylabel('j value of transition')
plt.grid()
plt.title('j. value of transition to instability, by mu_c')


fig, ax1 = plt.subplots()
# Bottom x-axis
ax1.plot(mucs, eig0_js, 'b.--', label='mu_c')
ax1.set_xlabel('mu_c')
ax1.set_ylabel('j value of transition to instability')
ax1.tick_params(axis='x', colors='blue')
ax1.grid()

# Top x-axis
ax2 = ax1.twiny()
ax2.plot(zs, eig0_js, 'r.--', label='z')
ax2.set_xlabel('z')
ax2.tick_params(axis='x', colors='red')
#ax2.grid()

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)



plt.show()

