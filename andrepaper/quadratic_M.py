
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
import random, math, sys

sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/Lotka-Voltera_Life-History/numerical_studies/')

from functions.lv_functions import A_matrix, A_matrix_upd, M_matrix, M_matrix_rand, lv_LH, LH_jacobian, x0_vec


seed = random.randint(0, 1000)
seed = 705 # 432 stable 644 unstable 
print('\n\n')
print("seed: ", seed)

random.seed(1)

def lv_LH_cutoff(x0, t, A, M, alpha): 
    def derivative(x, t, M, A, alpha):
        alphp = np.dot(alpha, x)
        dxdt = []
        for i in range(len(x0)):
            if x[i] <= 0:
                x[i] = 0
        Apart = np.multiply(x, np.dot(A, x))
        dxdtvec = np.dot(M, x) + Apart
        for i in range(len(x0)):
            if alphp[i] >7.6: dxdt.append(dxdtvec[i])
            elif alphp[i] <=7.6: dxdt.append(M[i][i]*x[i]+Apart[i])
            if x[i] <= 0:
                  dxdt[i] = 0
        return dxdt
    result = integrate.odeint(derivative, x0, t, args = (M, A, alpha))
    return result

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 100, 1000)
K_set = 1
C = 1

muc = -0.5
smuc = -0.5
mua = -0.6
smua = -0.6
f = 1.3
sf = 0.3
g = 1.2
sg = 0.3

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
K = (sigma2*C*s)**0.5
zest = 1e-5

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")
#if abs(R_c-R_a) > 1e-10:
#    raise Exception("error calculating R values.")
# endregion set variables 

# region set matrices 
A = A_matrix(n, C, sigma2, seed, LH=1)
alpha = np.clip(A, 0, None)
A_classic = A_matrix_upd(int(n/2), C, sigma2, seed, LH=0)
Avals, Avecs = np.linalg.eig(A)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10) 
A_rowsums = np.dot(A, np.ones(n))
print('max A rowsum:', '%.3f'%np.max(A_rowsums),'; real max A eig:', '%.3f'%np.max(np.real(Avalsm)))

# for m matrix:

M = M_matrix_rand(n, muc, smuc, mua, smua, f, sf, g, sg, seed=2)
mvals, mvecs = np.linalg.eig(M)

# region numerical
result = lv_LH_cutoff(x0, t, A, M, alpha)
xf_num = result[-1,:]
z_num = np.divide(result[:,::2], result[:, 1::2])
Z_num = np.divide(result[:,::2], (result[:, 1::2]+result[:, ::2]))

alphsum = []
for i in range(len(t)):
    alphp = np.dot(alpha, result[i,:])
    alphsum.append(np.sum(alphp))

#endregion
for i in range (n):
    print('std of x[i]: ', np.std(result[i, -20:-1]))






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
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
        plt.plot(t, Z_num[:, int(i/2)], '*', mfc = 'none', color=colors[math.floor(i/2)], ms = 5, markevery = (i, 20))
  #      plt.plot(t, z_num[:, int(i/2)], '*', mfc = 'none', color=colors[math.floor(i/2)], ms = 5, markevery = (i, 20))
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



plt.figure()
plt.grid()
plt.plot(t, alphsum)

#print('0.5 * Eigs of ALH:\n', Avals[::2]/2)
#print('Eigs of Aclassic:\n', Avals_cl)



plt.show()


print('\n\n')