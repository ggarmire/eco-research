print('\n')
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix3
from lv_functions import M_matrix3
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math

seed = random.randint(0, 1000)
#seed = 287
print("seed: ", seed)


#region initial conditions 

# values to set 
n = 21     # number of species 
x0 = x0_vec(n, 1)
t = np.linspace(0, 100, 2000)
K_set = 0.7
C = 1

mu1 = -0.5; mu2 = -0.6; mu3 = -0.7
f12 = 1.4; f13 = 1.5; f23 = 1.6
g21 = 0.9; g31 = 1; g32 = 1.1
# constraint settings 
xstar = 1       #flag: 1 if constraining abundances 

# values that dont get set 
s = int(n / 3)
sigma2 = K_set**2 / s / C
K = (sigma2*C*s)**0.5

# checks:
if n % 3 != 0:
    raise Exception("n is not a multiple of 3.")
if K!=K_set:
  raise Exception("K set does not match K.")

# region set matrices 
A = A_matrix3(n, C, sigma2, seed)
A_classic = A_matrix(s, C, sigma2, seed, LH=0)
Avals, Avecs = np.linalg.eig(A)

M = M_matrix3(n, mu1, mu2, mu3, f12, f13, f23, g21, g31, g32)
#print(M)

A_rowsums = np.dot(A, np.ones(n))
print('max A rowsums:', np.max(A_rowsums))

result = lv_LH(x0, t, A, M)

xf = result[-1, :]
print(xf)

i_live = 0
while xf[i_live] < 1e-5:
    i_live += 3

z1 = xf[i_live]/xf[i_live+2]; z2 = xf[i_live+1]/xf[i_live+2]
zvec = []
for i in range(s):
    zvec.extend([z1, z2, 1])

R = g31*z1 + g32*z2 + mu3
Rvec = R*np.ones(s)/(1+z1+z2)
print('R:', R)

if (R - 1/z2 * (g21*z1 + mu2*z2 + f23))/R > 1e-5:
    raise Exception("issue in the Rs.")

print('z1:', z1, ', z2:', z2)

# solve for analytical soln and jacobian:
A_inv = np.linalg.inv(A_classic)
xf_adult_an = -np.dot(A_inv, Rvec)
print('xf classical: ', xf_adult_an)
xf_an = np.repeat(xf_adult_an, 3)   # make unscaled
xf_an = np.multiply(xf_an, zvec)

print('analytical xf: ', xf_an)

#%% Stats: 


species_left = 0
species_stable = 0
for i in range(n):
    if result[-1, i] > 1e-3:
        species_left+=1
        if abs((result[-1, i]-result[-2, i]) / result[-1, i]) < 0.001:
            species_stable +=1

print("species remaining:", species_left, "sepcies stable: ", species_stable)

# region Calculate the Jacobian
Jac = LH_jacobian(A, M, xf_an) 

#print("Jacobian: ", Jac)
Jvals, Jvecs = np.linalg.eig(Jac)

#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

#plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', A seed ='+str(seed)+ ', K='+str('%.3f'%K))
plot_text = str('text')
if xstar == 1:
    plot_text2 = str('Max real eigenvalue of J: '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eigenvalue of A: '+ str('%.3f'%(np.max(np.real(Avals)))))
elif xstar == 0:
    plot_text2 = str('Max real eigenvalue of J: '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eigenvalue of A: '+ str('%.3f'%(np.max(np.real(Avals)))))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.5)


# region figures 

plt.figure()
plt.grid()

title = str('Species Population over time, N=3S='+str(n)+', x*=1, z = '+str(z1))
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(n):
    if i%3 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/3)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/3)], ms = 3, markevery = (i, 20))       # child (empty)
    elif i%3==1:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/3)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/3)], ms = 3, markevery = (i, 20))     # adult (full)
    elif i%3==2:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/3)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/3)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.13, 0.12, plot_text)
plt.figtext(0.4, 0.6, plot_text2, bbox=box_par)
#plt.figtext(0.5, 0.80, plot_text3)
#plt.figtext(0.5, 0.76, plot_text4)
plt.semilogx

legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)


# plot the analytical vs numerical final pops: 
plt.figure()
plt.plot(np.linspace(0, 1.1*np.max(xf),10),np.linspace(0, 1.1*np.max(xf),10),'--', label = 'y=x')
plt.plot(xf_an, xf, 'o', label = 'populations')
plt.grid()
plt.show()

print('\n')