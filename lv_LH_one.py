
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
import random 
import math

seed = 5
seed = np.random.randint(low=0, high = 1000)
np.random.seed(seed)
print("seed: ", seed)

#%% initial conditions and such 
n = 10     # number of species 
#x0 = np.random.uniform(low=0.1, high = 0.5, size=(n))
x0 = 0.5*np.ones(n)


t = np.linspace(0, 70, 1000)
sigma2 = 0.5
C = 0.5 

A = A_matrix(n, C, sigma2, seed=1, LH=1)
K = (C*sigma2*n)**0.5
print("complexity: ", K)
for r in A: 
    print(r)


# for m matrix:
muc = 0
mua = 0
f = np.random.uniform(low=0.1, high = 0.9)
g = np.random.uniform(low=0.1, high = 0.9)

M = M_matrix(n, muc, mua, f, g)

Avals, Avecs = np.linalg.eig(A)
Mvals, Mvecs = np.linalg.eig(M)

print("max eigenvalue of A: ", np.max(np.real(Avals)))
print ("eigenvalues of M: ", Mvals)

# run function here: 
result = lv_LH(x0, t, A, M)

print("final populations: ")
print(result[-1, :])

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

print("juvinile fractions: ", z)

print("tfinal: ", t[-1], ", species remaining:", species_left, "sepcies stable: ", species_stable)



#%% Plotting: 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.figure()

plt.grid()
plt.title("Species Population over time")
for i in range(n):
    if i%2 == 0:
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 5, markevery = 10)       # child (empty)
    else:
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 5, markevery = 10)     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.ylim(-.1, max(1.1, 1.1*np.max(result)))


plt.show()