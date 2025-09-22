# region preamble
print('\n')
# This is a classical GLV system but structured as a predator prey model, to begin 
# replicating Andre de Roos paper. 


# region libraries
import numpy as np
import matplotlib.pyplot as plt
import random, math, sys

sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/Lotka-Voltera_Life-History/numerical_studies/')

from lv_functions import A_matrix, A_matrix_predprey
from lv_functions import lv_classic
from lv_functions import x0_vec

# endregion

# region input variables 
seed = np.random.randint(0, 1000)
print('seed:', seed)

# system variables:
s = 50      # number of species: s ==n for 1 stage case
K = 3     # complexity 

C = 1       # connectedness

xf = x0_vec(s, 2)

x0 = x0_vec(s, 1)



# variables for evolution:
tsteps = 1000
tend = 100

# region variables you dont change:
sigma2 = K**2 / s / C       # variance in off diagonals 
t = np.linspace(0, tend, tsteps)

# endregion

#region set up system
A_pp = A_matrix_predprey(s, C, sigma2, seed=seed, LH=0)
Avals, no = np.linalg.eig(A_pp)

rvec = -np.dot(A_pp, xf)

# region run simulation 
result = lv_classic(x0, t, A_pp, rvec)


# region plot results 

#print(t)
plt.figure()
plt.grid()
plt.title("Species Population over time: classic case, pred/prey")
for i in range(s):
    plt.plot(t, result[:, i], '-', ms = 2)
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
#plt.ylim(-.1, 4)


plt.figure()
plt.grid()
plt.plot(np.real(Avals), np.imag(Avals), 'o')
plt.xlabel('real value')
plt.ylabel('imaginary value')
plt.title('A eigenvalues')

plt.show()
print('\n')