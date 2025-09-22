#region preamble 
print('\n')
import numpy as np
import matplotlib.pyplot as plt 
from lv_functions import A_matrix

# this code is intended to replicate the results of "Effect of population abundances on the stability of large random ecosystems"
# https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.022410

# endregion

seed = np.random.randint(0, 1000)
seed = 373
print(seed)

# region variables
s = 200
# A matrix
mud = -1
sigma2d = 0
K_set = 0.35
mu = 0

r = 0.1
sigma2 = K_set**2/s
K = (s*sigma2)**0.5
print('complexity K=', K)
# final abundances
mux = 1
sigmax = 0.1

Aeigs = []
Meigs = []
Meigsr = []

#endregion 

# region make arrays
#A = np.empty((s, s))
A = A_matrix(s, 1, sigma2, seed, LH=0)
xf = np.random.uniform(0.1, 2, s)
xf_setR = -np.dot(np.linalg.inv(A), r*np.ones(s))

'''for i in range(s):
    for j in range(s):
        if j == i: A[i][j] = np.random.normal(mud, sigma2d**0.5)
        elif j!= i: A[i][j] = np.random.normal(mu, sigma2**0.5)'''


M = np.dot(np.diag(xf), A)
Mr = np.dot(np.diag(xf_setR), A)


# endregion

Avals, no = np.linalg.eig(A)
Mvals, no = np.linalg.eig(M)
Mvalsr, no = np.linalg.eig(Mr)
Aeigs.extend(Avals)
Meigs.extend(Mvals)
Meigsr.extend(Mvalsr)
print('max eig Rset: ', np.max(np.real(Mvalsr)))

apar_text = str('seed ='+str(seed)+', n='+ str(s)+' species, K='+str(K_set)+', R='+str(r))
eigtext = str('max eig with set R: '+str('%0.4f'%np.max(np.real(Mvalsr))))

# region plot
plt.figure()
#plt.plot(np.real(Aeigs), np.imag(Aeigs), '.', alpha =0.8)
plt.plot(np.real(Meigsr), np.imag(Meigsr), '.', alpha =0.8, label = 'uniform R_eff')
plt.plot(np.real(Meigs), np.imag(Meigs), '.', alpha =0.8, label = 'random x*')
plt.grid()
plt.legend(loc = 'upper right')
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Jacobian Eigenvalues in classic GLV case')
plt.figtext(0.13, 0.15, apar_text)
plt.figtext(0.13, 0.12, eigtext)
plt.show()
