
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 
import math

seed = 1
seed = np.random.randint(low=0, high = 1000)
np.random.seed(seed)
print("seed: ", seed)

#%% initial conditions and such 
n = 10     # number of species 
#x0 = np.random.uniform(low=0.1, high = 0.5, size=(n))
x0 = np.ones(n)

for i in range(n):
    x0[i] += np.random.normal(loc=0, scale=0.2)


t = np.linspace(0, 10, 1000)
sigma2 = 0.5
C = 0.1

A = A_matrix(n, C, sigma2, seed=1, LH=1)
K = (C*sigma2*n)**0.5
print("complexity: ", K)
#print("A:")
print(A)

A_rowsums = np.zeros(n)
for i in range(n):
    #print(A[i, :])
    for j in range(n):
        A_rowsums[j] += A[j][i]

print('rowsums:', A_rowsums)


# for m matrix:
muc = 0.5
mua = 0.5
#f = np.random.uniform(low=0.1, high = 0.9)
#g = np.random.uniform(low=0.1, high = 0.9)
f = 0.5
g = 0.5
M = M_matrix(n, muc, mua, f, g)
print('M: ')
print(M)

for i in range(n):
    for j in range(n):
        if i%2 ==0:
            M[i][j] = -M[i][j]*A_rowsums[i]/(muc+f) 
        if i%2 ==1:
            M[i][j] = -M[i][j]*A_rowsums[i]/(mua+g) 


print("new M:", M)

# run function here: 
result = lv_LH(x0, t, A, M)
xf = result[-1, :]



#print("final populations: ")
#print(xf)

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

print("tfinal: ", t[-1], ", species remaining:", species_left, "sepcies stable: ", species_stable)

#%% Calculate the Jacobian 
delt = np.dot(A, result[-1, :])
#print(delt)

Jac = np.zeros((n,n))
for i in range(n):

    for k in range(n):
        Jac[i][k] = M[i][k] + xf[i]*A[i][k]
        if(i==k):
            Jac[i][k] += delt[i]

#print("Jacobian: ", Jac)

Jvals, Jvecs = np.linalg.eig(Jac)
#print('Eigenvalues of Jacobian: ', Jvals)





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

plt.ylim(-.1, max(1.1, 1.1*np.max(result)))


plt.show()