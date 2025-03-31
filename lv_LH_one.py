
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
from lv_functions import x0_vec
import random 
import math

seed = 398


#%% initial conditions and such 
n = 20     # number of species 
x0 = x0_vec(n)
#x0 = np.ones(n)
print('x0: ', x0)

t = np.linspace(0, 50, 1000)

sigma2 = 1.1**2/n
C = 1

A = A_matrix(n, C, sigma2, seed, LH=1)
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)

K = (C*sigma2*n)**0.5
print("complexity: ", K)
#print("A:"); print(A)
Avals, Avecs = np.linalg.eig(A_classic)

A_rowsums = np.dot(A, np.ones(n))
#print('A rowsums:', A_rowsums)

# for m matrix:
muc = -0.5
mua = -0.5
f = 0.51
g = 1

M = M_matrix(n, muc, mua, f, g)
#print('M:'); print(M)
M_rowsums = np.dot(M, np.ones(n))
for i in range(n):
    for j in range(n):
            M[i][j] == M[i][j]
            #M[i][j] = -M[i][j]*A_rowsums[i]/M_rowsums[i] 


# run function here: 
result = lv_LH(x0, t, A, M)



#result = integrate.odeint(derivative, x0, t, args = (M, A))



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

print("tfinal: ", t[-1], ", species remaining:", species_left, "sepcies stable: ", species_stable)

#%% Calculate the Jacobian
Jac = LH_jacobian(n, A, M) 
Jac2 = LH_jacobian_norowsum(result[-1, :], A, M)
#print("Jacobian: ", Jac)
Jvals, Jvecs = np.linalg.eig(Jac)
Jvals2, Jvecs2 = np.linalg.eig(Jac2)
#print('Eigenvalues of Jacobian: \n', Jvals)
print('Max real eigenvalue of A:', np.max(np.real(Avals)))
print('Max real eigenvalue of Jac:', np.max(np.real(Jvals)))
print('Max real eigenvalue of Jac2:', np.max(np.real(Jvals2)))




#%% Plotting: 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plt.figure()

plt.grid()
plt.title("Species Population over time: f=0.49, x*/=1")
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 10))       # child (empty)
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 10))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')

legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)

#plt.ylim(-0.1, 6)

#plt.ylim(min(0, np.min(result)-0.1), 1.1*np.max(result))


plt.show()