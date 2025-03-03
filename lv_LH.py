
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_matrices import A_matrix
from lv_matrices import M_matrix
import random 
import math

seed = 513
seed = np.random.randint(low=0, high = 1000)
np.random.seed(seed)
print("seed: ", seed)

#%% initial conditions and such 
n = 6     # number of species 

x0 = np.random.uniform(low=0.1, high = 1, size=(n))
#x0 = 0.5*np.ones(n)
C = 0.5    # connectedness|
sigma2 = 0.5;       ## variance in off diagonals of interaction matrix

t_end = 50     # length of time 
Nt = 10*t_end

K = (C*sigma2*n)**0.5
#print("x0: ", x0)
print("complexity: ", K)

# for m matrix:
muc = 0
mua = 0
f = 0.5
g = 0.5

A = A_matrix(n, C, sigma2, seed, LH=1) 
M = M_matrix(n, muc, mua, f, g, seed)

Avals, Avecs = np.linalg.eig(A)
Mvals, Mvecs = np.linalg.eig(M)
'''print ("A: ")
for r in A:
    print(r)'''
print("max eigenvalue of A: ", np.max(np.real(Avals)))
print ("eigenvalues of M: ", Mvals)

#print ("M: ")
#for r in M:
#    print(r)
#A= [[0, 1], [-1, 0]]

def derivative(x, t, M, A):
    for i in range(0, n):
        if x[i] <=0:
            x[i] = 0
    dxdt = np.dot(M, x) + np.multiply(x, np.dot(A, x))
    for i in range(0, n):
       if x[i]<=0:
            dxdt[i] == 0
    return dxdt

t = np.linspace(0, t_end, Nt)
result = integrate.odeint(derivative, x0, t, args = (M, A))

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




#if count !=0: print("final populations: ", result[-1, :])
#print("min value in x: ", np.min(result))
#print(t)

#%% Print results:
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
plt.legend()

plt.show()


