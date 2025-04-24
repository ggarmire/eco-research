
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import x0_vec
from lv_functions import lv_classic
import random 

seed = 1
print("seed: ", seed)

# LH stuff 
muc = -0.5
mua = -0.5
f = 1
g = 1

z = 0.7

xstar = 0



#%% initial conditions and such 
n = 2     # number of species 
x0 = np.ones(n)
x0_LH = x0_vec(n*2)
print(x0_LH)
for i in range(n):
    x0[i] = x0_LH[int(i*2)]+x0_LH[int(i*2)+1]
    print(x0[i])
print(x0)


r_one = (muc+f)*z/(1+z) + (g+mua)/(1+z)
r = (1+z)*r_one*np.ones(n) 

K_set = 0.8
C = 1    # connectedness|
sigma2 = K_set**2/n       ## variance in off diagonals of interaction matrix

t_end = 20     # length of time 
Nt = 1000

K = (C*sigma2*n)**0.5

#print("complexity: ", K)



A = A_matrix(n, C, sigma2, seed, LH=0) 
print(A)
Avals, Avecs = np.linalg.eig(A)
print('max real eig: ', np.max(np.real(Avals)))

A_offdiags = np.ma.masked_equal(A, -1)
variance = np.var(A_offdiags, ddof = 1)
K_act = (n*variance)**0.5
print(K_act)


if xstar == 1:
    r = -np.dot(A, np.ones(n))
elif xstar == 0:
    r = r_one * np.ones(n)



evals, evecs = np.linalg.eig(A)

t = np.linspace(0, t_end, Nt)
result = lv_classic(x0, t, A, r)

#print(np.max(result))
#print(result) 

count = 0
for num in result[-1, :]: 
    if num > 1e-3:
        count+=1


plot_text2 = str('Max real eigenvalue of A: '+ str('%.5f'%(np.max(np.real(Avals)))))

print("tfinal: ", t[-1], ", species remaining:", count)
'''print("final populations: ", result[-1, :])
print("min value in x: ", np.min(result))'''
#print(t)
plt.figure()
plt.grid()
plt.title("Species Population over time: classic case, x*/=1")
for i in range(n):
    plt.plot(t, result[:, i], 'o', ms = 3, mfc = 'none', markevery = 10)
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.figtext(0.3, 0.84, plot_text2)
#plt.ylim(-.1, 4)

plt.figure()
plt.grid()
plt.title("eigenvalues of A")
plt.xlabel('real component]')
plt.ylabel('imaginary component')
plt.plot(np.real(evals), np.imag(evals), 'o')



plt.show()


