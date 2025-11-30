# goal here is to replicate https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007827. 
# see functions as well! 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import sys 
sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic')


from funcs.lv_patches_functions import A_matrix_patches, D_matrix_patches, dxdt_multipatch, Jacobian_multipatch
from funcs.gglyapunov import LE_lead, LE_spec

start = time.time()

#seed = np.random.randint(0, 1000)
seed = 1
print('seed:', seed)
# set up variables 
s = 200
M = 8
# A variables:
C = 1./8.
Amean = 0.3 * s
Asigma = 0.45 * np.sqrt(s)
d = 1e-3
Nc = 1e-15
rho = 0.95

t_initial_int = 2500
t_LE = 100
t_warm_2 = 5
nLEs = 10
ds = 5

# set A matrix up with multiple patches
A = A_matrix_patches(s, M, C, Amean, Asigma, rho, seed=seed) 
B = np.full((s,M), 1)
D = D_matrix_patches(s, M, d)   

x0_seed = 1
np.random.seed(x0_seed)
x0 = np.abs(np.random.normal(1, 0.2, (s,M)).flatten())
#print(x0)
#x0 = np.load('m8s200seed86_xseed3_out.npy')
#print(len(x0))

# region initial integration 
tspan = [0, t_initial_int]
sol =  solve_ivp(dxdt_multipatch, tspan, x0, method='RK45', rtol=1e-7, atol=1e-7, args=(B,A,D,Nc))
tsol = sol.t
result = sol.y
print('solved initial integration.')
timed_initial_int = time.time()
xf_init_int = result[:, -1].reshape(s, M)       # abundances after the initial integration - reshape on patches
survived = np.where(np.max(xf_init_int, axis=1) > Nc)[0]        # where any patch has above Nc of species
s_new = len(survived)
print('number survive: ', s_new)

# region reduce system to survivors
x0_reduced = xf_init_int[survived, :].flatten()     # back to 1D array 

A_reduced = A[np.ix_(survived, survived, np.arange(M))]
D_reduced = D[survived, :, :]
B_reduced = B[survived, :]

J_indexing = np.hstack([np.arange(i*M, (i+1)*M) for i in range(s_new)])       # has to be funny bc J is s*Mxs*M
def dxdt_reduced(t, x, B, A, D, Nc):
    return dxdt_multipatch(t, x, B, A, D, Nc)
def Jacobian_reduced(t, x, B, A, D, Nc):
    J_old = Jacobian_multipatch(t, x, B, A, D, Nc)
    return J_old[np.ix_(J_indexing, J_indexing)]

# region calculate LEs in 2nd interval 
tspan_calc = [0, t_LE]
sol_calc = solve_ivp(dxdt_reduced, tspan_calc, x0_reduced, method='RK45', rtol=1e-8, atol=1e-8, args=(B_reduced,A_reduced,D_reduced,Nc))
tsol_calc = sol_calc.t
result_calc = sol_calc.y
print('solved LE timespan.')

'''LE1 = LE_lead(dxdt_reduced, Jacobian_reduced, x0_reduced, tend=t_LE+t_warm_2, twarm=t_warm_2, ds=1,p=(B_reduced, A_reduced, D_reduced, Nc))
print("Leading LE:", LE1)
#print('leading LE: ', LE1, ', took ', end-start)'''

start = time.time()
LEs, t_time, LE_time = LE_spec(dxdt_reduced, Jacobian_reduced, x0_reduced, t_warm_2, t_LE+t_warm_2, ds, p=(B_reduced, A_reduced, D_reduced, Nc), nLE = nLEs, LEseq=True) 

end = time.time()
print('LEs: ', LEs, ', took ', end-start)


# region results 


#np.save('m8s200seed86_xseed5_out.npy', xf_num)
#np.save('m8s200seed86_xseed3_LE50.npy', LEs)


result3D = result.reshape((result.shape[1], s, M))
result3D_mask = np.ma.masked_less(result3D, Nc)
mean_patch = np.mean(result3D, axis=1)
std_patch = np.std(result3D, axis=1)

species_left = np.sum(result>0, axis = 0) * 1/M

print('numerical species left:', np.sum(xf_init_int>Nc))

# region plot setup things 
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

patch_m1s = mean_patch - std_patch
patch_p1s = mean_patch + std_patch

# region plotting 
fig, ax = plt.subplots()
for i in range(s*M):
    ax.plot(tsol, result[i,:], '-', alpha = 0.5)
for j in range(s_new*M):
    ax.plot(tsol_calc+t_initial_int, result_calc[j,:], '-')
#    plt.plot(tsol[-1], xf_an[i], '.')
ax.grid()
ax.set_yscale('log')

fig, ax = plt.subplots()
ax.plot(tsol, species_left)
ax.grid()
ax.set_yscale('log')

index = np.arange(nLEs) + 1
fig, ax = plt.subplots()
ax.plot(index, LEs, '.')
ax.grid()

fig, ax = plt.subplots()
for i in range(nLEs):
    ax.plot(t_time, LE_time[:,i], '--')
ax.grid()
plt.show()