
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
import random, math, sys

sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/Lotka-Voltera_Life-History/numerical_studies/')

from functions.lv_functions import A_matrix, M_matrix, x0_vec


seed = random.randint(0, 1000)
seed = 272 # 432 stable 644 unstable 
print('\n\n')
print("seed: ", seed)

random.seed(1)
np.random.seed(1)


def derivative(x, t, M, A, B):
    dxdt = B + np.dot(M, x) + np.multiply(x, np.dot(A, x))
    for i in range(len(x0)):
            if x[i] <= 0: 
                dxdt[i] = 0
                x[i] = 0
    return dxdt

def lv_LH_quad(x0, t, A, M, omega): 
    result = integrate.odeint(derivative, x0, t, args = (M, A, B))
    return result

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 50, 1000)
K_set = 0.5
C = 1

muc = -0.03
mua = -0.05
f = -0.04
g = -0.05 

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5
zest = 1e-5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")



# M matrix :
M = M_matrix(n, muc, mua, f, g)
    
A = A_matrix(n, C, sigma2, seed, LH=1)

#B = 2*np.ones(n); Btext = str('resource: 1 for all')
B = np.random.uniform(0.1, 0.6,n); Btext = str('resource: uniform(0.1, 0.6)')

B = np.zeros(n); B[2], B[3] = [1, 1]; Btext = str('resource: 1 for one species only')


print('B:', B)


result = lv_LH_quad(x0, t, A, M, B)
xf_num = result[-1, :]
print(xf_num)
Z_num = np.divide(result[:,::2], (result[:, 1::2]+result[:, ::2]))
print('mean final z: ', np.mean(Z_num[-1, :]), '; std: ', np.std(Z_num[-1, :]))

#endregion

#%% Stats: 

species_left = np.sum(xf_num>zest)
print("species remaining:", species_left)


#region plot setup 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g))

apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))
plot_text2 = str(str(species_left)+str(' species left'))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.8)


# region figures 

# evolution of populations
plt.figure()
plt.grid()
if species_left == n: title = str('species population over time, stable case')
elif species_left < n: title = str('species population over time, unstable case')
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 10))       # child (empty)
        plt.plot(t, Z_num[:, int(i/2)], '--', mfc = 'none', color=colors[math.floor(i/2)], lw = 1, markevery = (i, 20))

  #      plt.plot(t, z_num[:, int(i/2)], '*', mfc = 'none', color=colors[math.floor(i/2)], ms = 5, markevery = (i, 20))
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 10))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.2, 0.80, apar_text)
plt.figtext(0.2, 0.83, mpar_text)
plt.figtext(0.2, 0.77, Btext)
plt.figtext(0.4, 0.6, plot_text2, bbox=box_par)
legend_elements = [Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='C0', markersize=3, label='child'),
    Line2D([0], [0], marker='o', linestyle='None', color = 'C0', markersize=3, label='adult'),
    Line2D([0], [0], linestyle='--', color='C0', linewidth=1, label='z')]

plt.legend(handles=legend_elements, loc='best')
plt.ylim(0, 1.2)


#print('0.5 * Eigs of ALH:\n', Avals[::2]/2)
#print('Eigs of Aclassic:\n', Avals_cl)



plt.show()


print('\n\n')