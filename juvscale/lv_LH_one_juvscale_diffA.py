
## This one specifically applies j to all juvenile interactions - including diagonal blocks.



#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale2
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import classic_jacobian
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
K_set = 1
C = 1

muc = -1
mua = -0.5
f = 1.5
g = 1.2

j = 0.5      # scales the effects of juviniles

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
print('sigma2:', sigma2)
K = (sigma2*C*s)**0.5

z = (muc-mua+((muc-mua)**2 +4*g*f)**0.5)/(2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua
print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a)

zest = 1e-5         # anything less is zero

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")
if abs(R_c-R_a) > 1e-10:
    raise Exception("error calculating R values.")
# endregion set variables 

# region set matrices 
A = A_matrix_juvscale2(n, C, sigma2, seed, j)
A_og = A_matrix(n, C, sigma2, seed, LH=1)
A_cl = A_matrix(s, C, sigma2, seed, LH=0)
#print('A: \n', A)
Avals, Avecs = np.linalg.eig(A)
Avalsm = np.ma.masked_inside(Avals, -zest, zest) 
Avals_cl, Avecs_cl = np.linalg.eig(A_cl)
Avals_og, Avec_og = np.linalg.eig(A_og)
Avals_ogm = np.ma.masked_inside(Avals_og, -zest, zest) 
A_rowsums = np.dot(A, np.ones(n))

print(f"max eig: {np.max(np.real(Avalsm)):.3f}, " 
    f"max eig j=1: {np.max(np.real(Avals_ogm)):.3f}, " 
    f"max eig classic: {np.max(np.real(Avals_cl)):.3f}")
# for m matrix:
M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)

# endregion

# region Analytical Final Abundances'
Rvec = R_a * np.ones(s)   # M part of equilibrium equation

# classic case: 
Ainv_cl = np.linalg.inv(A_cl)
xf_cl = -np.dot(Ainv_cl, Rvec)
Jac_cl = classic_jacobian(A_cl, xf_cl)
Jvals_cl, trash = np.linalg.eig(Jac_cl)
print('classic abundances: \n', xf_cl)
print(Jvals_cl)

# j=1 case: 
xf_og_adult = -np.dot(Ainv_cl, 1/(1+z)*Rvec)
xf_og = np.repeat(xf_og_adult, 2)   # make unscaled
xf_og[::2] *= z     # scale child 
#print(xf_og)
Jac_og = LH_jacobian(A_og, M, xf_og)
Jvals_og, trash = np.linalg.eig(Jac_og)
print('Jvasl og:', Jvals_og)

# j = j case
xf_adult = -np.dot(Ainv_cl, 1/(1+j*z)*Rvec)
xf = np.repeat(xf_adult, 2)   # make unscaled
xf[::2] *= z     # scale child 
Jac = LH_jacobian(A, M, xf)
Jvals, trash = np.linalg.eig(Jac)

print(f"max J eig: {np.max(np.real(Jvals)):.3f}, " 
    f"j=1: {np.max(np.real(Jvals_og)):.3f}, " 
    f"classic: {np.max(np.real(Jvals_cl)):.3f}")
print(f"max og/classic: {np.max(np.real(Jvals_og))/np.max(np.real(Jvals_cl)):.3f}")

# endregion

# region run function: 
result = lv_LH(x0, t, A, M)
xf_num = result[-1,:]

#print('final abnundances: \n', xf_num)
#print('analytical final abundances: \n', xf)


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
Jvals_num, trash = np.linalg.eig(Jac_num)

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
plt.ylim((-0.1, 1.2))

# eigenvalues, analytical
plt.figure()
plt.grid()
plt.plot(np.real(Jvals), np.imag(Jvals), '.', label = 'J, j='+str(j))
plt.plot(np.real(Jvals_og), np.imag(Jvals_og), 'o', ms = 9, mfc = 'none', label = 'J, j=1')
plt.plot(np.real(Jvals_cl), np.imag(Jvals_cl), 'o', ms = 6, mfc = 'none', label= "J, classical")
plt.title('eigenvalues of Jacobian')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.legend()

# eigenvalues of Aprime 
plt.figure()
plt.plot(np.real(Avals), np.imag(Avals), '.', label='A')
plt.plot(np.real(Avals_og), np.imag(Avals_og), 'o', mfc = 'none', label = 'A, j=1')
plt.plot(np.real(Avals_cl), np.imag(Avals_cl), 'o', mfc = 'none', label= "A, classical")
plt.grid()
plt.legend()
plt.xlabel('real component')
plt.ylabel('imaginary componenet')


# final pops- analytical vs numerucal
plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf),10),np.linspace(0, 1.1*np.max(xf),10),'--', label = 'y=x')
plt.plot(xf, xf_num, 'o', label = 'populations')
plt.xlabel('final populations, analytically found')
plt.ylabel('final populations, numerically found')
if species_left == n: plt.title('comparing final populations an/num (stable case)')
elif species_left < n: plt.title('comparing final populations an/num (unstable case)')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)

# final pops- j=j vs j=1
plt.figure()
plt.grid()
plt.plot(np.linspace(np.min(xf_og), 1.1*np.max(xf_og),10), (z+1)/(z*j+1)*np.linspace(np.min(xf_og), 1.1*np.max(xf_og),10),'--', label = 'y=(z+1)/(jz+1)x')
plt.plot(xf_og, xf, 'o', label = 'populations')
plt.xlabel('final populations, j=1')
plt.ylabel('final populations, j='+str(j))
plt.title('comparing final populations with j to original case')
plt.legend(loc = 'lower right')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)



plt.show()


print('\n')