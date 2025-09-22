#region preamble
# For different values of f and sigma, calculate system stability. 
# See if there is interesting divide in the phase space caused by f, or if it is 
# just similar to the classic case. 

#endregion 

#region functions 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import x0_vec
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import csv

import random 
#endregion


n = 50      # number of species

#region set up variables 
C = 1       # connectedness 
x0 = x0_vec(n)      # initial populations
t = np.linspace(0, 100, 2000)

# values for m
muc = -0.5      
mua = -0.5
g = 1.5
f = 1

M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)
print('max eig of M: ', np.max(np.real(mvals)))

actbigger = 0       # counts how many times the actual complexity is higher than set
One = np.ones(n)        # useful for rowsums 
#endregion

#region set up conditions 
#seed = random.randint(1, 1000)        # for now keeps every A the same but scaled (I think)
seed = 1
print('seed: ', seed)
runs = 100     # number of scenarios to run
xstar = 1      # do you want all of the final populations to be the same?
Ks = np.linspace(0.1, 2, runs)
Ks_actual = np.zeros(runs)
maxeigs = np.zeros(runs)
alpha = 1.667
Ks_stable = []
maxeig_stable = []
Ks_unstable = []
maxeig_unstable = []
n_species_stable = []
n_species_unstable = []

#endregion
nums = []

xs = np.ones(n)
for i in range(0, n, 2):
    xs[i] = alpha
#print(xs)

'''csv_name = str('Kvsstablilty_Aseed'+str(seed)+'_n'+str(n)+'_muc'+str(muc)+'_mua'+str(mua)+'_f'+str(f)+'_g'+str(g)+'.csv')
print(csv_name)

with open(csv_name, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['K', 'maxeig', 'speciesleft']
    writer.writerow(header)'''


    

for run in range(runs):
    if n > 20 and run%20 ==0:
        print(run)
    K = Ks[run]
    sigma2 = K**2/n * 2     # sigma to give complexity K in the classical case 
    #print('set std:', sigma2, ', Set K:', K)

    #region make matrices
    if seed != 0:
        A = A_matrix(n, C, sigma2, seed, LH=1)
        A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
    elif seed == 0: 
        #num = np.random.randint(1, 1000)
        num = run
        #print(num)
        A = A_matrix(n, C, sigma2, run, LH=1)
        A_classic = A_matrix(int(n/2), C, sigma2, num, LH=0)
        nums.append(num)

    
    

    #print(A)
    M = M_matrix(n, muc, mua, f, g)
    # scale to have identical rowsums 
    A_rowsums = np.dot(A, One)
    M_rowsums = np.dot(M, One)
    if xstar == 1:
        A_rows = np.dot(A, xs)
        M_rows = np.dot(M, xs)
        scales = -np.divide(np.multiply(A_rows, xs), M_rows)
        M = np.multiply(M, np.outer(scales, One))
    #endregion

    #region calculate actual stability
    var_actual = np.var(np.ma.masked_equal(A_classic, -1), ddof = 1)
    K_actual = (var_actual * n/2 * C)**0.5
    Ks_actual[run] = K_actual
    #endregion

    #region run ODE
    result = lv_LH(x0, t, A, M)
    xf = result[-1, :]

    species_left = 0
    for i in range(n):
        if xf[i] > 1e-3:
            species_left+=1
    species_left = species_left/2

    #endregion

    #region determine stability
    if xstar == 1:
        Jac = LH_jacobian(n, A, M, xs)
    elif xstar == 0:
        Jac = LH_jacobian_norowsum(xf, A, M)
    Jvals, Jvecs = np.linalg.eig(Jac)
    maxeig = np.max(np.real(Jvals))
    maxeigs[run] = maxeig
    #endregion

    if maxeig <= 0:
        Ks_stable.append(K)
        maxeig_stable.append(maxeig)
        n_species_stable.append(species_left)
    elif maxeig > 0:
        Ks_unstable.append(K)
        maxeig_unstable.append(maxeig)
        n_species_unstable.append(species_left)

        '''with open(csv_name, newline='') as file:
            writer = csv.writer(file)
            writer.writerow([K, maxeig, species_left])'''




#region save data 



#endregion

#region make figures
mss = 4
if seed != 0:   
    plot_title = str('Max real eigenvalue for K: '+str(n/2)+'*2 species, A seed = ' +str(seed)+', z='+str(alpha))
else:
    plot_title = str('Max real eigenvalue for K: '+str(int(n/2))+'*2 species, random As')

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $g =$'+str(g))


# K vs max eig
plt.figure(figsize=(8, 6))    # K vs f
plt.plot(Ks_stable, maxeig_stable, 'og', ms = mss, alpha = 0.5, label = 'stable scenarios')
plt.plot(Ks_unstable, maxeig_unstable, 'ob', ms = mss, alpha = 0.5, label = 'unstable scenarios')
plt.figtext(0.13, 0.12, plot_text)
plt.legend(bbox_to_anchor=(1, 1),  loc='upper right')       
plt.grid()
plt.title(plot_title)
plt.xlabel('(classical) complexity K')
plt.ylabel('Maximum real eigenvalue')
#plt.xlim(0.2, 2)
#plt.ylim(-2, 12)
#plt.ylim(-0.78, 0.78)


# K vs species left 
plt.figure(figsize=(8, 6))    
plt.plot(Ks_stable, n_species_stable, 'og', ms = mss, alpha = 0.5, label = 'stable scenarios')
plt.plot(Ks_unstable, n_species_unstable, 'ob', ms = mss, alpha = 0.5, label = 'unstable scenarios')
plt.figtext(0.13, 0.12, plot_text)
plt.legend(bbox_to_anchor=(1, 1),  loc='upper right')       
plt.grid()
plt.title(plot_title)
plt.xlabel('(classical) complexity K')
plt.ylabel('number of species remaining ')
#plt.xlim(fmin-0.4, fmax+0.4)
#plt.ylim(-2, 3)


plt.figure(figsize=(8,6))   # histogram of A seeds 
plt.hist(nums, bins=20, range =(0, 1000))



'''# actual K vs f 
plt.figure(figsize=(8, 6))    # K vs f
plt.plot(Ks_actual, stables, 'og', ms = mss, alpha = 0.5, label = 'stable scenarios')
plt.plot(Ks_actual, unstables, 'ob', ms = mss, alpha = 0.5, label = 'unstable scenarios')
plt.figtext(0.13, 0.12, plot_text)
plt.legend(bbox_to_anchor=(1, 1),  loc='upper right')       
plt.grid()
plt.title(plot_title)
plt.xlabel('actual (classical) complexity K')
plt.ylabel('Maximum real eigenvalue')
#plt.annotate(ann_text, (max_K, maxrealeig), va='top')
#plt.xlim(fmin-0.4, fmax+0.4)
#plt.ylim(Kmin-0.4, Kmax+0.4)'''


plt.show()
















