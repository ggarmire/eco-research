
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

seed = random.randint(0, 1000)
#seed = 944
print('\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 2000)
K_set = 0.7
C = 1

muc = -1
mua = -0.5
f = 1.5
g = 1.2

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
print('sigma2:', sigma2)
K = (sigma2*C*s)**0.5

z = (muc-mua+((muc-mua)**2 +4*g*f)**0.5)/(2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua
print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a)

zest = 1e-5         # anything less is zero

# for m matrix:
M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")
if abs(R_c-R_a) > 1e-10:
    raise Exception("error calculating R values.")
# endregion set variables 

# region classical case
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
print('max classic eig:', np.max(np.real(Avals_cl)))
# endregion



#region loop
js = np.linspace(0, 1, 11)

xfs_byj = {}
Ars_byj = {}
Jvals_byj = {}
Jvals_byj_imag = {}
Avals_sc_byj = {}
maxeig_J_byj = []
maxeig_A_byj = []
xf_std = []
xf_mean = []

flag02 = 0 
flag05 = 0
flag08 = 0

#region plot setup 

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

#endregion

for j in js: 
    print('j = ', j)
    # region set A matrix 
    A = A_matrix_juvscale(n, C, sigma2, seed, j)
    Avals, Avecs = np.linalg.eig(A)
    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10) 
    A_rowsums = np.dot(A, np.ones(n))
    #endregion

    # region Analytical Final Abundances'
    D = (j*z+1)*np.ones((s,s)) + (z-j*z)*np.identity(s)
    #print('D matrix:\n', D )
    Aprime = np.multiply(D, A_classic)
    Ap_inv = np.linalg.inv(Aprime)
    Rvec = R_a * np.ones(s)   # M part of equilibrium equation
    xf_an_adult = -np.dot(Ap_inv, Rvec)
    xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
    xf_an[::2] *= z     # scale child 

    Jac = LH_jacobian(A, M, xf_an)
    Jvals, Jvecs = np.linalg.eig(Jac)
    Jvals_imag = []
    for k in range(n):
        if abs(Jvals[k].imag) > zest:
            Jvals_imag.append(Jvals[k])

    print('     max eigenvalue of J:', np.max(np.real(Jvals)))
    # endregion

    # region other A stuff 
    A_scaled = np.multiply(np.outer(xf_an, np.ones(n)), A)
    Avals_sc, Avecs_sc = np.linalg.eig(A_scaled)
    Avals_sc_ma = np.ma.masked_inside(Avals_sc, -zest, zest)
    #print('max eig A_scale:', np.max(np.real(Avals_sc_ma)))

    A_rows_scaled = np.dot(A, xf_an)

    Mp = M +np.diag(A_rows_scaled) 

    Mpvals, Mpvecs = np.linalg.eig(Mp)
    #endregion

    xfs_byj['j_{0}'.format(j)] = xf_an
    Ars_byj['j_{0}'.format(j)] = A_rowsums
    Jvals_byj['j_{0}'.format(j)] = Jvals
    Jvals_byj_imag['j_{0}'.format(j)] = Jvals_imag
    Avals_sc_byj['j_{0}'.format(j)] = Avals_sc

    maxeig_J_byj.append(np.max(np.real(Jvals)))
    maxeig_A_byj.append(np.max(np.real(Avals_sc_ma)))
    xf_mean.append(np.mean(xf_an))
    xf_std.append(np.std(xf_an))


    #region plotting in loop
    if (j == 0) or (j ==1) or (j >= 0.2 and flag02 == 0) or (j >= 0.5 and flag05 ==0) or (j >= 0.8 and flag08 ==0):
        if j >= 0.2 and flag02 == 0: flag02 = 1
        elif j >=0.5 and flag05 == 0: flag05 = 1
        elif j >=0.8 and flag08 == 0: flag08 = 1
        plt.figure()
        plt.grid()
        plt.plot(np.real(Jvals), np.imag(Jvals), '.')
        plt.xlabel('real component'); plt.ylabel('imaginary component')
        plt.title('Eigenvalues of J, j='+str(j))
        #plt.xlim((-3, 1)); 
        #plt.ylim((-0.015, 0.015))
        plt.figtext(0.2, 0.15, apar_text)
        plt.figtext(0.2, 0.12, mpar_text)      
    # endregion

#endregion loop


# J eigenvalues, all
plt.figure()
plt.grid()
for j in js: 
    Jvals = Jvals_byj['j_{0}'.format(j)]
    Jvals_imag = Jvals_byj_imag['j_{0}'.format(j)]
    plt.plot(np.real(Jvals), j*np.ones(n), '.', color='C0')
    plt.plot(np.real(Jvals_imag), j*np.ones(len(Jvals_imag)), '.', color='C1')
plt.ylabel('j, scale of inter-species juvenile effects')
plt.xlabel('real components of Jacobian eigenvalues')
plt.title('Distribution of J eigenvalues with changing j')
plt.figtext(0.2, 0.15, apar_text)
plt.figtext(0.2, 0.12, mpar_text)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', label='on real line'),
                   Line2D([0], [0], marker = 'o', color='C1', label='off real line')]
plt.legend(handles=legend_elements, loc = 'upper left')
plt.ylim((-0.15, 1.1))

plt.figure()
plt.grid()
for j in js: 
    xfs = xfs_byj['j_{0}'.format(j)]
    for i in range(s):
        plt.plot(xfs[2*i], j, 'o', mfc = 'none', ms = 3, color = colors[i])
        plt.plot(xfs[2*i+1], j, 'o', ms = 3, color = colors[i])
plt.ylabel('j, scale of inter-species juvenile effects')
plt.xlabel('final abundances (analytically found)')
plt.title('Distribution of final abundances with changing j')
plt.figtext(0.2, 0.15, apar_text)
plt.figtext(0.2, 0.12, mpar_text)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements, loc='lower right')
plt.ylim((-0.15, 1.1))

plt.figure()
plt.grid()
for j in js: 
    Ars = Ars_byj['j_{0}'.format(j)]
    for i in range(s):
        plt.plot(Ars[2*i], j, 'o', mfc = 'none', ms = 3, color = colors[i])
        plt.plot(Ars[2*i+1], j, 'o', ms = 3, color = colors[i])
plt.ylabel('j, scale of inter-species juvenile effects')
plt.xlabel('Row sums of A')
plt.title('Distribution of scaled row sums of A with changing j')
plt.figtext(0.2, 0.15, apar_text)
plt.figtext(0.2, 0.12, mpar_text)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements, loc='lower right')
plt.ylim((-0.15, 1.1))

plt.figure()
plt.plot(xf_std, js, '.', label = 'standard deviation')
plt.plot(xf_mean, js, '.', label = 'mean')
plt.grid()
plt.legend()
plt.title('Mean/Standard deviation of population densities for changing j')
plt.ylabel('j')
plt.xlabel('value')
plt.show()

print('\n')