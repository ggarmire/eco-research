print('\n')
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import x0_vec
from lv_functions import lv_classic
import random 


seed = random.randint(0, 1000)
seed = 563
print("seed: ", seed)

xstar = 1

#%% initial conditions and such 
n = 10     # number of species 
#x0 = x0_vec(n, 1)
x0 = x0_vec(n, 1)
K_set = 1.4


C = 1    # connectedness|
sigma2 = K_set**2/n       ## variance in off diagonals of interaction matrix

t_end = 50     # length of time 
Nt = 2000

K = (C*sigma2*n)**0.5

#print("complexity: ", K)



A = A_matrix(n, C, sigma2, seed, LH=0) 
print('sigma2:', sigma2)# 
#print('A: \n', A)

Avals, Avecs = np.linalg.eig(A)
print('max eig A:', np.max(np.real(Avals)))

if xstar == 1:
    r = -np.dot(A, np.ones(n))
elif xstar == 0:
    r = np.ones(n)
#print('max rowsum:', np.max(-r))
#print(r)

xf_an = -np.dot(np.linalg.inv(A), r)
#print('xf_an: ', xf_an)

Jac = np.multiply(np.outer(xf_an, np.ones(n)), A)
#print('Jac: \n', Jac)

evals, evecs = np.linalg.eig(Jac)
print('max eigenvalue:', np.max(np.real(evals)))

t = np.linspace(0, t_end, Nt)
result = lv_classic(x0, t, A, r)

#print(np.max(result))
#print(result) 

count = 0
for num in result[-1, :]: 
    if num > 1e-3:
        count+=1
    

print("tfinal: ", t[-1], ", species remaining:", count)
'''print("final populations: ", result[-1, :])
print("min value in x: ", np.min(result))'''
#print(t)
plt.figure()
plt.grid()
if xstar == 1:
    plt.title("Species Population over time: classic case, x*=1")
elif xstar != 1:
    plt.title("Species Population over time: classic case, x*/=1")
for i in range(n):
    plt.plot(t, result[:, i], 'o', ms = 3, mfc = 'none', markevery = 10)
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
#plt.ylim(-.1, 4)

plt.figure()
plt.grid()
plt.title("eigenvalues of Jac")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(np.real(evals), np.imag(evals), 'o')

plt.figure()
plt.grid()
plt.plot(np.linspace(0, 1.1*np.max(xf_an),10),np.linspace(0, 1.1*np.max(xf_an),10),'--', label = 'y=x')
plt.plot(xf_an, result[-1, :], '.')
plt.xlabel('final populations, analytically found')
plt.ylabel('final populations, numerically found')
if num == n: plt.title('comparing final populations, stable case')
elif num < n: plt.title('comparing final populations, unstable case')




plt.show()


print('\n')