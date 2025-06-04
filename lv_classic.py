
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import x0_vec
from lv_functions import lv_classic
import random 


seed = random.randint(0, 1000)
seed = 576
print("seed: ", seed)

xstar = 1

#%% initial conditions and such 
n = 10     # number of species 
x0 = x0_vec(n, 1)

K_set = 1.1


C = 1    # connectedness|
sigma2 = K_set**2/n       ## variance in off diagonals of interaction matrix

t_end = 50     # length of time 
Nt = 2000

K = (C*sigma2*n)**0.5

#print("complexity: ", K)



A = A_matrix(n, C, sigma2, seed, LH=0) 
A_offdiags = np.ma.masked_equal(A, -1)
variance = np.var(A_offdiags, ddof = 1)
K_act = (n*variance)**0.5
print(K_act)


if xstar == 1:
    r = -np.dot(A, np.ones(n))
elif xstar == 0:
    r = np.ones(n)
print('max rowsum:', np.max(-r))
print(r)

evals, evecs = np.linalg.eig(A)
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
plt.title("Species Population over time: classic case, x*=1")
for i in range(n):
    plt.plot(t, result[:, i], 'o', ms = 3, mfc = 'none', markevery = 10)
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
#plt.ylim(-.1, 4)

plt.figure()
plt.grid()
plt.title("eigenvalues of A")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(np.real(evals), np.imag(evals), 'o')


plt.show()


