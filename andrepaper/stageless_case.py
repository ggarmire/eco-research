
# this code doenst work. everyone dies! 
# supposed to be an exact replica of Andre's code. 


#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
import random, math, sys

sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/Lotka-Voltera_Life-History/numerical_studies/')

seed = np.random.randint(0, 1000)
print('seed:', seed)
np.random.seed(seed)
# functions for derivatives:
def bates(n, N):
    x = np.zeros(N)
    for i in range(n):
        x += np.random.uniform(0, 1, N)
    xvec = 1./n * x
    return xvec

def xi(n, N, w25):
    return np.multiply(np.ones(N)+ 0.2*(bates(3, N)-0.5), w25)

def psi_func(x):
    if (-0.5 <= x) and (-1./6. > x): return 6*(x+0.5)**2
    elif (-1./6. <=x) and (1./6. > x): return -12*(x+0.5)**2+12*(x+0.5)-2
    elif (1./6. <= x) and (0.5 > x): return 6*(x-0.5)**2
    else: return 0

def psi_mat(ni, ci, ri):
    n= len(ni)
    psi = np.zeros((n,n))
    for i in range(n):
        for k in range(n):
            psi[i][k] = psi_func((ni[k]-ci[i])/ri)
    return psi

def Evec(psi, x):
    E = np.zeros(psi.shape[0])
    for i in range(psi.shape[0]):
        for k in range(i):
            E[i] += psi[i][k] * x[k]
    return E

def lv_derivative_limsum(C, t, *p):
    n = C.shape[0]
    P, delta, psi, alpha, H, gamma, T, mu = p
    E = Evec(psi, C)
    F0 = P/(delta+alpha[0]*C[0])
    F = np.divide(E, (H+E))
    F[0] = F0
    dx = np.multiply(gamma, np.multiply(F, C)) - np.multiply(T+mu, C)

    for i in range(n):
        for k in range(i+1, n):
            dx[i] -= C[i]*alpha[k]*psi[k][i]*C[k]/(H[k]+E[k])
    return dx

# constants 
wmax = 10**4; wmin = 10**(-8)     # grams
logw = 12

# establish variables
N = 100
np.random.seed(1)
zest = 10**(-4)

ni = np.sort(np.random.uniform(0, 1, N))
wi = np.multiply(np.power(wmax, ni),  np.power(wmin, 1-ni))
ci = np.zeros(N)
ri = 1./12.

for i in range(N): ci[i] = np.random.uniform(ni[i]-2.5/12., ni[i]-0.5/12.)
psi = psi_mat(ni, ci, ri)

P = 60.
delta = 2.

# random variables: 
w25 = np.power(wi, -0.25)
alpha = xi(3, N, w25)
gamma = 0.6 * xi(3, N, w25)
T = 0.1 * xi(3, N, w25)
mu = 0.015 * xi(3, N, w25)

H = np.random.uniform(0.5, 2.5, N)

p = (P, delta, psi, alpha, H, gamma, T, mu)

# initial conditions: 

x0 = 0.5* np.ones(N)

t = np.linspace(0, 50, 500)

result = integrate.odeint(lv_derivative_limsum, x0, t, args= p)


xf = result[-1, :]
species_left = np.sum(xf>zest)
print('species left:', species_left)
print('species 1 population:' , xf[0])


# plot 
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

# evolution of populations
plt.figure()
plt.grid()
title = str('species population over time, stageless')
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(N):
    plt.plot(t, result[:, i], 'o', ms = 3, mfc = 'none', markevery = 20)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.ylim(-0.2, 4)

plt.figure()
plt.hist(ni, bins=15, histtype='step')
plt.hist(ci, bins=15, histtype='step')
plt.hist(w25,bins=15, histtype='step')


plt.show()
