
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
seed = 875
print('\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 2000)
K_set = 0.3
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

# region classical analog 
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
#print('max classic eig:', np.max(np.real(Avals_cl)))
# endregion


js = np.linspace(0, 1, 11)

for j in js: 
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
    print('max eigenvalue of J:', np.max(np.real(Jvals)))
    print('# of Jvals:', len(Jvals))
    # endregion

    # region other A stuff 
    A_scaled = np.multiply(np.outer(xf_an, np.ones(n)), A)
    Avals_sc, Avecs_sc = np.linalg.eig(A_scaled)
    Avals_sc_ma = np.ma.masked_inside(Avals_sc, -zest, zest)

    A_scaled_cl = np.multiply(np.outer(xf_an_adult, np.ones(s)), A_classic)
    Avals_sc_cl, Avecs_sc_cl = np.linalg.eig(A_scaled_cl)
    print('max eig A_scale_cl:', np.max(np.real(Avals_sc_cl)))
    print('max eig A_scale:', np.max(np.real(Avals_sc_ma)))


    #endregion

    # region run function: 
    result = lv_LH(x0, t, A, M)
    xf_num = result[-1,:]

    #print('final abnundances: \n', xf_num)
    #print('analytical final abundances: \n', xf_an)



    A_rows_scaled = np.dot(A, xf_an)
    #print('A dot xf:', A_rows_scaled)
    A_rows_scaled_cl = np.dot(A_classic, xf_an_adult)
    Mp = M +np.diag(A_rows_scaled) 

    Mpvals, Mpvecs = np.linalg.eig(Mp)
    #print('Mp eigs: ', Mpvals)

    #%% Stats: 
    species_left = 0
    species_stable = 0
    for i in range(n):
        if result[-1, i] > 1e-3:
            species_left+=1
            if abs((result[-1, i]-result[-2, i]) / result[-1, i]) < 0.001:
                species_stable +=1

    print("species remaining:", species_left)



    # region Calculate the Jacobian
    Jac_num = LH_jacobian(A, M, xf_num)
    #print("Jacobian: ", Jac)
    Jnumvals, Jnumvecs = np.linalg.eig(Jac_num)

    #region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))



plot_text2 = str('Max real eig of J: '+ str('%.3f'%(np.max(np.real(Jvals))))) 
               # + '\n Max real eig J (analytical): '+ str('%.3f'%(np.max(np.real(Janvals)))))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.8)


# region figures 

# evolution of populations
plt.figure()
plt.grid()
if species_left == n: title = str('species population over time, j = '+str(j)+' (stable)')
elif species_left < n: title = str('species population over time, j = '+str(j)+' (unstable)')
plt.title(title)
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)
plt.figtext(0.4, 0.6, plot_text2, bbox=box_par)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)
plt.ylim((0, 1.2))

# eigenvalues, analytical
plt.figure()
plt.grid()
plt.plot(np.real(Jvals), np.imag(Jvals), '.', label = 'Analytical J')
plt.plot(np.real(Jnumvals), np.imag(Jnumvals), 'o', mfc = 'none', label = 'Numerical J')
plt.plot(np.real(Mpvals), np.imag(Mpvals), '.', label= "M' eigs")
plt.title('eigenvalues of Jacobian')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.legend()


# final pops- analytical vs numerucal
plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf_an),10),np.linspace(0, 1.1*np.max(xf_an),10),'--', label = 'y=x')
plt.plot(xf_an, xf_num, 'o', label = 'populations')
plt.xlabel('final populations, analytically found')
plt.ylabel('final populations, numerically found')
if species_left == n: plt.title('comparing final populations, stable case')
elif species_left < n: plt.title('comparing final populations, unstable case')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)

# final pops- j=j vs j=1
plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf_an_og),10),np.linspace(0, 1.1*np.max(xf_an_og),10),'--', label = 'y=x')
plt.plot(xf_an_og, xf_an, 'o', label = 'populations')
plt.xlabel('final populations, j=1')
plt.ylabel('final populations, j='+str(j))
plt.title('comparing final populations with j to original case')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)

plt.figure()
plt.grid()
plt.plot(np.real(Jvals_og), np.imag(Jvals_og), 'o', mfc = 'none', color = 'C1', label = 'J, j=1')
plt.plot(np.real(Jvals), np.imag(Jvals), '.', color = 'C0', label = str('J, j='+str(j)))
plt.legend()
plt.title('comparing eigenvalues with j to original case')
plt.xlabel('real component')
plt.ylabel('imaginary component')

'''plt.figure()
plt.grid()
plt.plot(xf_an, np.divide(xf_an-xf, xf_an), '.')
plt.xlabel('analytical solution')
plt.ylabel('fractional difference between analytical and numerical')'''




plt.show()


print('\n')