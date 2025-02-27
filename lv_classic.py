
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_matrices import A_matrix
import random 

seed = 11
np.random.seed(seed)


#%% initial conditions and such 
n = 4      # number of species 
#x0 = np.random.uniform(low=0.1, high = 1, size=(n))
x0 = np.ones(n)
#r = np.random.uniform(low=0, high=0.01, size=n)
r= np.ones(n)
#r = [1, -1]
C = 0.5     # connectedness|
sigma2 = 0.1;       ## variance in off diagonals of interaction matrix

t_end = 30     # length of time 
Nt = 1000

K = (C*sigma2*n)**0.5
print("x0: ", x0)
print("r: ", r)
print("complexity: ", K)



A = A_matrix(n, C, sigma2, seed, LH=0) #- np.identity(n)

evals, evecs = np.linalg.eig(A)
print("min eigenvalue: ", np.min(np.real(evals)))
#A= [[0, 1], [-1, 0]]

def derivative(x, t, r, A):
    for i in range(0, n):
        if x[i] <=0:
            x[i] = 0
    dxdt = np.multiply(r, x) - np.multiply(x, np.dot(A, x))
    for i in range(0, n):
       if x[i]<=0:
            dxdt[i] == 0
    return dxdt

t = np.linspace(0, t_end, Nt)
result = integrate.odeint(derivative, x0, t, args = (r, A))
#print(np.max(result))
print(result)

print("tfinal: ", t[-1], "xfinal: ", result[-1, :])
#print(t)
plt.figure()

plt.grid()
plt.title("odeint method")
for i in range(n):
    plt.plot(t, result[:, i], 'x')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.ylim(0, 200)
plt.legend()

plt.show()


