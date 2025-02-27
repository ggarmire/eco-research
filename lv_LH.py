
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_matrices import A_matrix
from lv_matrices import M_matrix
import random 

seed = 1
np.random.seed(seed)


#%% initial conditions and such 
n = 10     # number of species 

#x0 = np.random.uniform(low=0.1, high = 1, size=(n))
x0 = 0.5*np.ones(n)
C = 0.1    # connectedness|
sigma2 = 0.5;       ## variance in off diagonals of interaction matrix

t_end = 50     # length of time 
Nt = 100*t_end

K = (C*sigma2*n)**0.5
print("x0: ", x0)
print("complexity: ", K)

# for m matrix:
muc = 0.5
mua = 0.5*muc
f = 0.1
g = 0.1

A = A_matrix(n, C, sigma2, seed, LH=1) 
M = M_matrix(n, muc, mua, f, g, seed)

evals, evecs = np.linalg.eig(A)
print ("A: ")
for r in A:
    print(r)
print("max eigenvalue: ", np.max(np.real(evals)))

print ("M: ")
for r in M:
    print(r)
#A= [[0, 1], [-1, 0]]

def derivative(x, t, r, A):
    for i in range(0, n):
        if x[i] <=0:
            x[i] = 0
    dxdt = np.dot(M, x) + np.multiply(x, np.dot(A, x))
    for i in range(0, n):
       if x[i]<=0:
            dxdt[i] == 0
    return dxdt

t = np.linspace(0, t_end, Nt)
result = integrate.odeint(derivative, x0, t, args = (r, A))
#print(np.max(result))
#print(result) 

count = 0
for num in result[-1, :]: 
    if num > 1e-3:
        count+=1
    


print("tfinal: ", t[-1], ", species remaining:", count)
if count !=0: print("final populations: ", result[-1, :])
#print("min value in x: ", np.min(result))
#print(t)
plt.figure()

plt.grid()
plt.title("Species Population over time")
for i in range(n):
    plt.plot(t, result[:, i], 'x')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.ylim(-.2, 1.2)
plt.legend()

plt.show()


