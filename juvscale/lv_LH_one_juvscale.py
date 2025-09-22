
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
seed = 464 # 432 stable 644 unstable 
print('\n\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 1000)
K_set = 0.5
C = 1

muc = -0.8
mua = -0.3
f = 1.5
g = 1

j = 0.5
j1 = 0.5; j2 = 1; j3 = 1; j4 = 1.3
#j1 = 1; j2 = 1/j; j3 = j; j4 = 1
jmat = [[j1, j2],[j3, j4]]
print('j matrix: \n', jmat)

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5

zest = 1e-5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")

# endregion set variables 

# region set matrices 
A = A_matrix_juvscale(n, C, sigma2, seed, jmat)
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
A_inv_cl = np.linalg.inv(A_classic)
Avals, Avecs = np.linalg.eig(A)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10) 
A_rowsums = np.dot(A, np.ones(n))
print('max A rowsum:', '%.3f'%np.max(A_rowsums),'; real max A eig:', '%.3f'%np.max(np.real(Avalsm)))

# for m matrix:
M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)

#print('primary M eigs:', '%.3f'%mvals[0], '%.3f'%mvals[1], '%.3f'%mvals[2], '%.3f'%mvals[3])# endregion matrices 


# region numerical
result = lv_LH(x0, t, A, M)
xf_num = result[-1,:]
species_left = np.sum(xf_num>zest)

z_num = np.divide(result[:,::2], result[:, 1::2])
print('numerical z=', np.mean(z_num[-1, :]))
#endregion

#region analytical 
res_forz = lv_LH(x0, np.linspace(0, 500, 1000), A, M)
z = np.mean(np.divide(res_forz[-1, ::2], res_forz[-1, 1::2]))

R = (g*z+mua) / (jmat[1][0]*z+jmat[1][1])
Rvec = R*np.ones(s)

xf_an_adult = -np.dot(A_inv_cl, Rvec)
xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
xf_an[::2] *= z     # scale child 

#endregion 


#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

#mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z=)')

apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))



#plot_text2 = str('Max real eig J (analytical): '+ str('%.3f'%(np.max(np.real(Jvals)))) +"\nM' eigenvalue: "+str('%.3f'%(mvals[0]-mvals[1])+', R='+str('%.3f'%R_a)))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.8)


# region figures 

# evolution of populations
fig = plt.figure() 
gs=fig.add_gridspec(2, hspace=0.1, height_ratios=[3, 1])
axs = gs.subplots(sharex=True, sharey=False)
axs[0].grid(); axs[1].grid()

if species_left == n: title = str('species population over time, stable case')
elif species_left < n: title = str('species population over time, unstable case')
axs[0].set_title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(n):
    if i%2 == 0:
        axs[0].plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        axs[0].plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
        axs[1].plot(t, z_num[:, int(i/2)], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))
    else:
        axs[0].plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        axs[0].plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
axs[0].set_ylabel('Population density')
axs[1].set_ylabel('juvenile fraction z')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)
#plt.figtext(0.4, 0.6, plot_text2, bbox=box_par)
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
axs[0].legend(handles=legend_elements)


'''# eigenvalues, analytica
plt.figure()
plt.grid()
plt.plot(np.real(Jvals), np.imag(Jvals), '.', label= 'Jacobian eigs')
plt.plot(np.real(Mpvals), np.imag(Mpvals), 'o', mfc = 'none', label= "M' eigs")
plt.plot(np.real(Asvals), np.imag(Asvals), 'o', mfc = 'none', label = 'X.A eigs')
plt.title('eigenvalues of Jacobian, numerical')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.legend()

'''
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