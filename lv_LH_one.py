
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
seed = 464 # 432 stable 644 unstable 
print('\n\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 10, 1000)
K_set = 0.5
C = 1

muc = -1
mua = -0.1
f = 1.5
g = 0.6
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

#print('primary M eigs:', '%.3f'%mvals[0], '%.3f'%mvals[1], '%.3f'%mvals[2], '%.3f'%mvals[3])
# endregion matrices 

# region analytical final abundances
A_inv = np.linalg.inv(A_classic)
#print(A_inv)
Rvec = R_a/(1+z) * np.ones(s)   # for in
xf_an_adult = -np.dot(A_inv, Rvec)  # solve classical system
xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
xf_an[::2] *= z     # scale child 

Jac_an = LH_jacobian(A, M, xf_an)
Jvals, Jvecs = np.linalg.eig(Jac_an)
print('max eigenvalue of J_an:', np.max(np.real(Jvals)))

# endregion

A_scaled = np.dot(np.diag(xf_an), A)
Asvals, Asvecs = np.linalg.eig(A_scaled)


A_rows_scaled = np.dot(A, xf_an)
Mp = M +np.diag(A_rows_scaled) 

Mpvals, Mpvecs = np.linalg.eig(Mp)
print('Mp eig: ', Mpvals[0], ', from M: ', mvals[0]-mvals[1])

# region numerical
result = lv_LH(x0, t, A, M)
xf_num = result[-1,:]


#endregion

#%% Stats: 
species_left = np.sum(xf_an>0)
print("species remaining:", species_left)


#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))



plot_text2 = str('Max real eig J (analytical): '+ str('%.3f'%(np.max(np.real(Jvals)))) +"\nM' eigenvalue: "+str('%.3f'%(mvals[0]-mvals[1])))

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
plt.plot(np.real(Jvals), np.imag(Jvals), '.', label= 'Jacobian eigs')
plt.plot(np.real(Mpvals), np.imag(Mpvals), 'o', mfc = 'none', label= "M' eigs")
plt.plot(np.real(Asvals), np.imag(Asvals), 'o', mfc = 'none', label = 'X.A eigs')
plt.title('eigenvalues of Jacobian, numerical')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.legend()


# final pops- analytical vs numerical
plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf_num),10),np.linspace(0, 1.1*np.max(xf_num),10),'--', label = 'y=x')
plt.plot(xf_an, xf_num, 'o', label = 'populations')
plt.xlabel('final populations, analytically found')
plt.ylabel('final populations, numerically found')
if species_left == n: plt.title('comparing final populations, stable case')
elif species_left < n: plt.title('comparing final populations, unstable case')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)



#print('0.5 * Eigs of ALH:\n', Avals[::2]/2)
#print('Eigs of Aclassic:\n', Avals_cl)



plt.show()


print('\n\n')