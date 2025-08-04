
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math

seed = random.randint(0, 1000)
seed = 563 # 432 stable 644 unstable 
print('\n\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 6     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 2000)
K_set = 0.6
C = 1

muc = -1
mua = -0.5
f = 1.5
g = 1.2
z = (muc-mua+((muc-mua)**2 +4*g*f)**0.5)/(2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua
print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a)

# constraint settings 
xstar = 0      #flag: 1 if constraining abundances 

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")
if abs(R_c-R_a) > 1e-10:
    raise Exception("error calculating R values.")
# endregion set variables 

# region set matrices 
A = A_matrix(n, C, sigma2, seed, LH=1)
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
Avals, Avecs = np.linalg.eig(A)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10) 
A_rowsums = np.dot(A, np.ones(n))
print('max A rowsum:', '%.3f'%np.max(A_rowsums),'; real max A eig:', '%.3f'%np.max(np.real(Avalsm)))

# for m matrix:
M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)
if xstar == 1:
    xs = np.ones(n)
    for i in range(0, n, 2):
        xs[i] = z
    
    #print(xs)
    A_rows = np.dot(A, xs)
    M_rows = np.dot(M, xs)
    scales = -np.divide(np.multiply(A_rows, xs), M_rows)
    M = np.multiply(M, np.outer(scales, np.ones(n)))
    if np.max(np.diag(M)) > 0: print('M has a positive diagonal.')
    if np.min(np.diag(M, 1)) < 0: print('M has a negative f value.')
    if np.min(np.diag(M, -1)) < 0: print('M has a negative g value.')


    # alternative 1: scale off diagonals of M
    '''for row in range(0, n, 2):
        M[row][row+1] = -z*muc - A_rows[row]
        M[row+1][row] = 1/z * (-mua - A_rows[row])'''


    Mprime = M + np.diag(A_rows)
    mpvals, mpvecs = np.linalg.eig(Mprime)
print('primary M eigs:', '%.3f'%mvals[0], '%.3f'%mvals[1], '%.3f'%mvals[2], '%.3f'%mvals[3])
# endregion matrices 

# region analytical final abundances
A_inv = np.linalg.inv(A_classic)
print(A_inv)
Rvec = R_a/(1+z) * np.ones(s)   # for in
xf_an_adult = -np.dot(A_inv, Rvec)  # solve classical system
xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
xf_an[::2] *= z     # scale child 

Jac_an = LH_jacobian(A, M, xf_an)
Janvals, Janvecs = np.linalg.eig(Jac_an)
print('max eigenvalue of J_an:', np.max(np.real(Janvals)))

# endregion

# region run function: 
result = lv_LH(x0, t, A, M)
xf = result[-1,:]
#print('xf: ', xf)

A_scaled = np.multiply(np.outer(xf, np.ones(n)), A)
A_rows_scaled = np.dot(A, xf)
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
Jac = LH_jacobian(A, M, result[-1, :])
#print("Jacobian: ", Jac)
Jvals, Jvecs = np.linalg.eig(Jac)

#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))


if xstar == 1:
    plot_text2 = str('Max real eigenvalue of J: '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eigenvalue of Mprime: '+ str('%.3f'%(np.max(np.real(mpvals))))
                    +'\n Max real eigenvalue of A: '+ str('%.3f'%(np.max(np.real(Avals)))))
elif xstar == 0:
    plot_text2 = str('Max real eig J (numerical): '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eig J (analytical): '+ str('%.3f'%(np.max(np.real(Janvals)))))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.8)


# region figures 

# evolution of populations
plt.figure()
plt.grid()
if xstar == 1:
    title = str('Species Population over time, N=2S='+str(n)+', x*=1, z = '+str(z))
elif xstar ==0: 
    if species_left == n: title = str('species population over time, stable case')
    elif species_left < n: title = str('species population over time, unstable case')
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
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

# eigenvalues, analytica
plt.figure()
plt.grid()
plt.plot(np.real(Jvals), np.imag(Jvals), '.')
plt.plot(np.real(Mpvals), np.imag(Mpvals), '.', label= "M' eigs")
plt.title('eigenvalues of Jacobian, numerical')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.legend()

# eigenvaluesm analytical
plt.figure()
plt.grid()
plt.plot(np.real(Janvals), np.imag(Janvals), '.')
plt.title('eigenvalues of Jacobian, analytical')
plt.xlabel('real component')
plt.ylabel('imaginary component')

# eigs, analytical and numerical 
plt.figure()
plt.grid()
plt.title('eigenvalues of Jacobian')
plt.plot(np.real(Jvals), np.imag(Jvals), '.', label = 'numerical')
plt.plot(np.real(Janvals), np.imag(Janvals), '.', label= 'analytical')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)
plt.legend()

# final pops- analytical vs numerucal
plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf),10),np.linspace(0, 1.1*np.max(xf),10),'--', label = 'y=x')
plt.plot(xf_an, xf, 'o', label = 'populations')
plt.xlabel('final populations, analytically found')
plt.ylabel('final populations, numerically found')
if species_left == n: plt.title('comparing final populations, stable case')
elif species_left < n: plt.title('comparing final populations, unstable case')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)

'''plt.figure()
plt.grid()
plt.plot(xf_an, np.divide(xf_an-xf, xf_an), '.')
plt.xlabel('analytical solution')
plt.ylabel('fractional difference between analytical and numerical')'''

plt.figure()
plt.grid()
plt.plot(xf, A_rows_scaled, '.')
plt.xlabel('final abundances')
plt.ylabel('A dot xf')
plt.ylim(-0.8, -0.5)


plt.figure()
plt.grid()
plt.plot(np.real(Avals_cl), np.imag(Avals_cl),'.', label = 'classic')
plt.plot(np.real(Avals), np.imag(Avals), '.', label = 'LH')
plt.plot(np.real(Avals)/2, np.imag(Avals)/2, 'o', mfc = 'none', label = 'LH/2')
plt.ylabel('Imaginary component')
plt.xlabel('Real component')
plt.title('Eigs of (unscaled) A - classic case vs. 2-stage A ')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)
plt.legend()


print('0.5 * Eigs of ALH:\n', Avals[::2]/2)
print('Eigs of Aclassic:\n', Avals_cl)



plt.show()


print('\n\n')