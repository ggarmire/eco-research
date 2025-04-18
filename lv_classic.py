
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import x0_vec
import random 

#seed = 36
seed = random.randint(0, 1000)
print("seed: ", seed)

#%% initial conditions and such 
n = 10     # number of species 
x0 = x0_vec(n)

r = np.ones(n)
C = 1    # connectedness|
sigma2 = .9**2/n       ## variance in off diagonals of interaction matrix

t_end = 30     # length of time 
Nt = 1000

K = (C*sigma2*n)**0.5
#print("x0: ", x0)
#print("r: ", r)
print("complexity: ", K)



A = A_matrix(n, C, sigma2, seed, LH=0) 
print(A)
        
evals, evecs = np.linalg.eig(A)

print("max eigenvalue: ", np.max(np.real(evals)))

def derivative(x, t, r, A):
    for i in range(0, n):
        if x[i] <=0:
            x[i] = 0
    dxdt = np.multiply(r, x) + np.multiply(x, np.dot(A, x))
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
    

'''print("tfinal: ", t[-1], ", species remaining:", count)
print("final populations: ", result[-1, :])
print("min value in x: ", np.min(result))'''
#print(t)
plt.figure()
plt.grid()
plt.title("Species Population over time")
for i in range(n):
    plt.plot(t, result[:, i], 'o', ms = 3, mfc = 'none', markevery = 10)
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.ylim(-.1, max(1.1, 1.1*np.max(result)))

plt.figure()
plt.grid()
plt.title("eigenvalues of A")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(np.real(evals), np.imag(evals), 'o')


plt.show()


