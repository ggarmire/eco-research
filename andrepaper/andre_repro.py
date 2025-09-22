# region preamble
# this is a reproduction of andres paper 

# libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
from scipy import integrate

# functions
def psi_func(x):
    if (-1/2<=x) and (x<-1/6):
        px = 6*(x+0.5)**2
    elif (-1/6<=x) and (x<1/6):
        px = -12*(x+1/2)**2+12*(x+1/2)-2
    elif (1/6<=x) and (x<1/2):
        px = 6*(x-0.5)**2
    else: px = 0
    
    return px
        
def psi_mat(n, nis, cis, ri):
    s = int(n/2)
    psi = np.zeros((s,s))
    for i in range(s):
        for j in range(s):
            x = (nis[j]-cis[i])/ri
            psi[i][j] = psi_func(x)
    #psi = np.repeat(np.repeat(psi, 2, 0), 2, 1)
    return psi


def derivative(x, t, q, phi, His, alphs, gams, Ts, mus, psi):
    s = len(alphs)
    J = x[::2]; A = x[1::2]
    C = J + A
    E = np.zeros(s)
    for i in range(s):
        for k in range(i): E[i]+=psi[i][k]*(phi*J[k]+(2-phi)*A[k])
    M = np.zeros(s)
    for i in range(s):
        for l in range(i+1, s): M[i]+= alphs[l]*psi[l][i]*(q*J[l]+(2-q)*A[l])/(His[k]+E[k])
    Fis = np.divide(E, (E+His))
    dxdt = []
    for i in range(s):
        mi = max(gams[i]*q*Fis[i]-Ts[i], 0)
        bi = max(gams[i]*(2-q)*Fis[i]-Ts[i], 0)
        num = np.divide(q*J+(2-q)*A , (E+His))
        Mi = M[i]
        dxdt.append(bi*A[i] - (mi+mus[i])*J[i] -phi*J[i]*Mi)
        dxdt.append(mi*J[i] - mus[i]*A[i] -(2-phi)*A[i]*Mi)
    return dxdt


# endregion


# region set variables

# set variables
wmax = 10**4; wmin = 10**(-8)          # grams
logw = math.log10(wmax/wmin)
n = 200      # number of species
s = int(n/2)

# for random variables: 
alph0  = 1; sigalph = 0.1
gam0 = 0.6; siggam = 0.1
T0 = 0.1; sigT = 0.1
mu0 = 0.015; sigmu = 0.1

q = 0.7
phi = 1.8
#endregion

# region set up species specs

nis = np.sort(np.random.uniform(0,1,s))

cis = np.zeros(s)
wis = np.zeros(s)
for i in range(s): 
    cis[i] = np.random.uniform(nis[i]-2.5/logw, nis[i]-0.5/logw)
    wis[i] = (wmax)**nis[i] * (wmin)**(1-nis[i])
ri = 1/logw
His = np.random.uniform(0.5, 2.5, s)

psi = psi_mat(n, nis, cis, ri)
#print(psi)
nionzero = np.count_nonzero(psi)
print('frac of nonzeros in psi: ', nionzero/s**2)

# random variables from bates distribution:
w25 = np.power(wis, -0.25)
halfarr = 0.5*np.ones(s)
alphs = alph0*np.multiply((1+2*sigalph*(np.random.uniform(0, 1,s)-halfarr)), w25)
gams = gam0*np.multiply((1+2*siggam*(np.random.uniform(0, 1,s)-halfarr)), w25)
Ts = T0*np.multiply((1+2*sigT*(np.random.uniform(0, 1,s)-halfarr)), w25)
mus = mu0*np.multiply((1+2*sigmu*(np.random.uniform(0, 1,s)-halfarr)), w25)


# region integrate 
x0 = np.ones(n)
t = np.linspace(0, 1000, 1000)

result = integrate.odeint(derivative, x0, t, args = (q, phi, His, alphs, gams, Ts, mus, psi))



colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

# evolution of populations
plt.figure()
plt.grid()
title = str('Species Population over time, N=2S='+str(n))
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)%10], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)%10], ms = 3, markevery = (i, 20))       # child (empty)
  #      plt.plot(t, z_num[:, int(i/2)], '*', mfc = 'none', color=colors[math.floor(i/2)], ms = 5, markevery = (i, 20))
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)%10], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)%10], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)

plt.show()