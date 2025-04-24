
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from lv_functions import A_matrix
import random 


x0 = 0.6
r = 2
a = -1
def derivative(x, t, r, a):
    dx = r*x + a*x**2
    return dx

t_end = 100
Nt = 2000
t = np.linspace(0, t_end, Nt)
result = integrate.odeint(derivative, x0, t, args = (r, a))


plt.figure()

plt.grid()
plt.title("Species Population over time")
plt.plot(t, result, '.')
plt.xlabel('Time t')
plt.ylabel('Population')

#plt.show()

X = [1, 2, 3, 4]
Y = [2, 1, 2, 1]
print(np.outer(X, np.ones(4)))
print(np.divide(Y, X))

