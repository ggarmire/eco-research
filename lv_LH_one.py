
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 
import math

seed = 203
#seed = np.random.randint(low=0, high = 1000)

print("seed: ", seed)

#%% initial conditions and such 
n = 20     # number of species 
#x0 = np.random.uniform(low=0.1, high = 0.5, size=(n))
x0 = np.ones(n)

for i in range(n):
    x0[i] += np.random.normal(loc=0, scale=0.5)


t = np.linspace(0, 400, 1000)


sigma2 = .9**2/n
C = 1

A = A_matrix(n, C, sigma2, seed=203, LH=1)
K = (C*sigma2*n)**0.5
print("complexity: ", K)
#print("A:")
#print(A)
Avals, Avecs = np.linalg.eig(A)

A_rowsums = np.dot(A, np.ones(n))
print('A rowsums:', A_rowsums)
print('wrogn:', np.dot(np.ones(n), A))


# for m matrix:
muc = .5
mua = .5

f = .5
g = .5

M = M_matrix(n, muc, mua, f, g)
M_rowsums = np.dot(M, np.ones(n))

for i in range(n):
    for j in range(n):
        if i%2 ==0:
            M[i][j] = -M[i][j]*A_rowsums[i]/M_rowsums[i] 
        if i%2 ==1:
            M[i][j] = -M[i][j]*A_rowsums[i]/M_rowsums[i]  


print("new M rowsums:", np.dot(M, np.ones(n)))

# run function here: 
result = lv_LH(x0, t, A, M)
xf = result[-1, :]



print("final populations: ")
print(xf)

#%% Stats: 

species_left = 0
species_stable = 0
z = np.zeros(int(n/2))
for i in range(n):
    if result[-1, i] > 1e-3:
        species_left+=1
        if abs((result[-1, i]-result[-2, i]) / result[-1, i]) < 0.001:
            species_stable +=1
    if i%2 == 0:
        z[int(i/2)]= (result[-1,i]/(result[-1,i]+result[-1,i+1]))

#print("juvinile fractions: ", z)

#print("tfinal: ", t[-1], ", species remaining:", species_left, "sepcies stable: ", species_stable)

#%% Calculate the Jacobian 

Jac = A+M
for i in range(n):
    Jac[i][i] += A_rowsums[i]

#print("Jacobian: ", Jac)

Jvals, Jvecs = np.linalg.eig(Jac)
#print('Eigenvalues of Jacobian: \n', Jvals)
print('Max real eigenvalue of A:', np.max(np.real(Avals)))
print('Max real eigenvalue of Jac:', np.max(np.real(Jvals)))




#%% Plotting: 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.figure()

plt.grid()
plt.title("Species Population over time")
for i in range(n):
    if i%2 == 0:
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = 10)       # child (empty)
    else:
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = 10)     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')

plt.ylim(0.9*np.min(result), 1.1*np.max(result))


plt.show()