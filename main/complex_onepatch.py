# replicate the single patch dynamics outlined in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007827
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from funcs.lv_patches_functions import A_matrix_patches, dxdt_singlepatch

seed = np.random.randint(0, 1000)
print('seed:', seed)
# set up variables 
s = 250
M = 1
# A variables:
C = 1./8.
Amean = 0.3 
Asigma = 0.45
Nc = 1e-15

# set A matrix up with 1 patch! 
A = A_matrix_patches(s, M, C, Amean, Asigma, rho=1, seed=seed)  # correlation doesnt matter
B = np.full((s,M), 1)
x0 = np.abs(np.random.normal(1, 0.2, (s,M)).flatten())
tspan = [0, 2000]

sol =  solve_ivp(dxdt_singlepatch, tspan, x0, method='RK45', rtol=1e-9, atol=1e-9, args=(B,A,Nc))
tsol = sol.t
ressol = sol.y

# region results 
xf_num = ressol[:, -1]
mean_abun = np.mean(ressol, axis=0)
std_abun = np.std(ressol, axis=0)

A_mat = np.squeeze(A[:, :, 0]) 
xf_an = np.linalg.solve(np.eye(s)+A_mat, B)
xf_an_mean = np.mean(xf_an)
xf_an_std = np.std(xf_an)
print('analytical species left:', np.sum(xf_an>Nc))
print('numerical species left:', np.sum(xf_num>Nc))



# region plotting 
fig, ax = plt.subplots()
for i in range(s):
    ax.plot(tsol, ressol[i,:], '-')
#    plt.plot(tsol[-1], xf_an[i], '.')
ax.grid()
ax.set_yscale('log')

plt.figure()
plt.plot(tsol, mean_abun, label = 'mean abundance')
plt.fill_between(tsol, mean_abun-std_abun, mean_abun+std_abun, color='blue', alpha=0.3, label='one std')
plt.grid()


plt.show()