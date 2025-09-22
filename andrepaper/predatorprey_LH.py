# region preamble
print('\n')
# This is a classical GLV system but structured as a predator prey model, to begin 
# replicating Andre de Roos paper. 


# region libraries
import numpy as np
import matplotlib.pyplot as plt
import random, math, sys

sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/Lotka-Voltera_Life-History/numerical_studies/')

from lv_functions import A_matrix_predprey
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import M_matrix
from lv_functions import LH_jacobian

# endregion

# region input variables 
seed = np.random.randint(0, 1000)
#seed = 703
print('seed:', seed)

# system variables:
s = 20      # number of species: n = 2s for 2 life stages
n = 2*s
K = 1     # complexity 

C = 1       # connectedness

muc = -0.5
mua = -0.5
f = 1.5 
g = 1 

x0 = x0_vec(n, 1)



# variables for evolution:
tsteps = 1000
tend = 100

# region variables you dont change:
sigma2 = K**2 / s / C       # variance in off diagonals 
t = np.linspace(0, tend, tsteps)
z = (muc - mua + (((mua-muc)**2) +4*g*f)**0.5) / (2*g)
R = muc + f/z
Ra = mua +g*z
print('z: ', z, ', Rc: ', R, ', Ra:', Ra)

# endregion

#region set up system
A_pp = A_matrix_predprey(n, C, sigma2, seed=seed, LH=1)
A_cl = A_matrix_predprey(s, C, sigma2, seed=seed, LH=0)
print(A_pp)
Avals, no = np.linalg.eig(A_pp)

# M matrix: 
M = M_matrix(n, muc, mua, f, g)
print(M)

# analytical soln 
Rvec = R/(1+z)*np.ones(s)
xstar_adult = -np.dot(np.linalg.inv(A_cl), Rvec)
xstar = np.repeat(xstar_adult, 2)   # make unscaled
xstar[::2] *= z     # scale child 


# region run simulation 
result = lv_LH(x0, t, A_pp, M)
xf = result[-1, :]

J = LH_jacobian(A_pp, M, xstar)
Jvals, no = np.linalg.eig(J)

# region plot results 
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

#print(t)
plt.figure()
plt.grid()
plt.title("Species Population over time: LH case, pred/prey")
for i in range(s):
    plt.plot(0, result[0, 2*i], '--', ms = 2, color=colors[math.floor(i/2)%10])
    plt.plot(t, result[:, 2*i], '--', ms = 2, color=colors[math.floor(i/2)%10], markevery = (2*i, 20))       # child (empty)
    plt.plot(0, result[0, 2*i+1], '-', ms = 2, color=colors[math.floor(i/2)%10])
    plt.plot(t, result[:, 2*i+1], '-', ms = 2, color=colors[math.floor(i/2)%10], markevery = (2*i, 20))     # adult (full)

plt.xlabel('Time t, [days]')
plt.ylabel('Population density')



plt.figure()
plt.grid()
plt.plot(np.real(Avals), np.imag(Avals), 'o', label = 'Avals')
plt.plot(np.real(Jvals), np.imag(Jvals), 'o', label = 'Jvals')
plt.plot()
plt.xlabel('real value')
plt.ylabel('imaginary value')
plt.title('eigenvalues of A and J')
plt.legend()

plt.figure()
plt.grid()
plt.plot(xstar, xf, '.')
plt.plot(xf, xf, '-')
plt.xlabel('analytical ')
plt.ylabel('numerical ')
plt.title('analytical vs numerical final abundacnes')

plt.show()
print('\n')