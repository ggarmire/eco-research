
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
import random 

muc = 0
mua = 0
f = 0.7
g = 0.3

x0 = [0.5, 0.25]

def derivative(x, t, muc, mua, f, g):
    if x[0] < 0: x[0] == 0
    if x[1] < 0: x[1] == 0
    nc = x[0]
    na = x[1]
    dnc = muc*nc +f*na-nc**2
    dna = mua*na +g*nc-nc**2
    if x[0] ==0: dnc ==0
    if x[1] ==0: dna ==0
    return [dnc, dna]

t_end = 100
Nt = 2000
t = np.linspace(0, t_end, Nt)
result = integrate.odeint(derivative, x0, t, args = (muc, mua, f, g))

z = result[-1, 0] / (result[-1, 0]+result[-1,1])

print('final pops:', result[-1, :])
print('juvinile fraction: ', z)
plt.figure()

plt.grid()
plt.title("Species Population over time")
plt.plot(t, result[:, 0], '.')
plt.plot(t, result[:, 1], '.')
plt.xlabel('Time t')
plt.ylabel('Population')
#plt.ylim(-.1, max(1.1, 1.1*np.max(result)))
plt.legend()

plt.show()


