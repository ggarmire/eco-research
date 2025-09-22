
# libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import random 
from matplotlib.colors import LogNorm

#region Set Variables 

# stuff that gets changed: 
n = 50
runs = 500

# A matrix 

K_set = 0.7
C = 1

pct = 0

# M matrix 
muc = -0.8
mua = -0.5
f = 1.5
g = 1.2

z = 1

# stuff that does not get changed:
s = n/2
x0 = x0_vec(n, 1)
sigma2 = K_set**2/n*2
t = np.linspace(0, 300, 600)
M_pre = M_matrix(n, muc, mua, f, g)

One = np.ones(n)

print('n:', n, ', sigma:', '%.3f'%(sigma2**0.5))

#region make arrays 
eigs_J = []     # eigenvalues of the jacobian 

eigs_M = []     # eigenvlaues of M after scaling 
eigs_Mp = []        # eigenvalues of M+delta term 

eigs_A = []     # eigenvalues of A 

A_rowsums = []      # rowsums of A 
Ars_max = []        # max rowsum of A 
Ars_max_scaled = []        # max rowsum of A 

maxeig_J = []       # max eigenvalue for each run
maxeig_A = []
maxeig_Mp = []

n_survives_unconstrianed = []

constraint = np.ones(n)


for i in range(0, n, 2):
    constraint[i] = z

# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %87 == 0:
        print(run)

    # make A matrix 
    A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
    A_rows = np.dot(A, One)

    # find final abundances unconstrained: 
    result_un = lv_LH(x0, t, A, M_pre)     # ODE with unscaled M
    xf_un = result_un[-1,:]
    n_survives_unconstrianed.append((xf_un>1).sum())

    diff_from_set = constraint - xf_un

    xs = xf_un + pct*diff_from_set

    # A specs with cosntraint 
    A_rows_scaled = np.dot(A, xs)
    A_rowsums.extend(A_rows)
    Ars_max_scaled.append(np.max(A_rows_scaled))
    Ars_max.append(np.max(A_rows))

    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)

    # set M
    M_rows = np.dot(M_pre, xs)
    scales = -np.divide(np.multiply(A_rows_scaled, xs), M_rows)
    M = np.multiply(M_pre, np.outer(scales, np.ones(n)))
        
    Mp = M + np.diag(A_rows)        # mprime = m + delta

    Mvals, Mvecs = np.linalg.eig(M)
    Mpvals, Mpvecs = np.linalg.eig(Mp)
    
    eigs_M.extend(Mvals)
    eigs_Mp.extend(Mpvals)

    Jac = LH_jacobian(n, A, M, xs) 
    Jvals, Jvecs = np.linalg.eig(Jac) 
    
    eigs_J.extend(Jvals)
    
    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10)      # nonzero eigenvalues of the two, since the zeros dissapear later 
    Mpvalsm = np.ma.masked_inside(Mpvals, -1e-10, 1e-10)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))


#region figure setup 
fsize = (6, 6)
mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g))




# region plotting 

plt.figure(figsize = fsize)
plt.plot(np.real(eigs_J), np.imag(eigs_J), 'o', ms = 2, label = 'J')
plt.grid()
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalues: fraction shift = '+ str(pct)+', K='+str(K_set)+',n='+str(n)+',z='+str('%.3f'%z))
plt.figtext(0.13, 0.12, mpar_text)
plt.legend()


plt.figure(figsize = fsize)
plt.plot(np.real(eigs_J), np.imag(eigs_J), 'o', ms = 2, color = 'C0', alpha = 0.5, label = 'J')
plt.plot(np.real(eigs_Mp), np.imag(eigs_Mp), 'o', ms = 2, mfc = 'none', color = 'C1', alpha=0.5, label = "M'")
plt.plot(np.real(eigs_A), np.imag(eigs_A), 'o', ms = 2, mfc = 'none', color = 'C2', alpha=0.5, label = "A")

plt.grid()
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalues: fraction shift = '+ str(pct)+', K='+str(K_set)+',n='+str(n)+',z='+str('%.3f'%z))
plt.figtext(0.13, 0.12, mpar_text)
plt.legend()


plt.show()

