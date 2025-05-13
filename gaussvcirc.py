
#region libraries 
import math
import numpy as np
import matplotlib.pyplot as plt

'''

from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
import random 
import pandas as pd '''


# Comparing the edge of the circle to the probability of a gaussian draw less than the circle edge for the leading eigenvalue.

#region establish constants 

# A matrix: 
C = 1   # derivations done with this. dont change
sigma_a = 0.3     # used in drawing random entries of A
S = 10    # number of distinct species. A is 2s x 2s matrix.

# M matrix: 
f = 1.5; g = 1; muc = -0.5; mua = -0.5      # all used to find the 2nd eigenvalue, lambda.


#region calculate other variables that come up 
K = sigma_a * (C*S)**0.5
print('sigma_a:', sigma_a)
lambda_p = (mua*muc - f*g)/((muc+f)*(mua+g)) - 1        # nonzero eigenvalue of M'=M+delta
print('l2p:', lambda_p)

mean_g = lambda_p
sigma_g = (lambda_p**2 *4*(S-1)*sigma_a**2)**0.5
#print(sigma_g)



# region characterize distribution edges 

# circle:
C_right = -2 + 2*K      # right edge of the circle 
C_left = -2 - 2*K      # right edge of the circle 

# Gaussian 

def Y_func(x, l2p, sigma_a, s): 
    sg = abs(2*l2p*(s-1)**0.5 * sigma_a)
    E = 1/(2**0.5 * sg) * (x - 2*l2p)
    return 0.5 * (1+ math.erf(E))

def P_func(x, l2p, sigma_a, s): 
    sg = abs(2*l2p*(s-1)**0.5 * sigma_a)
    E = 1/(2**0.5 * sg) * (x - 2*l2p)
    Y = 0.5 * (1+ math.erf(E))
    return 1 - Y**s

print('Y of 0:', Y_func(0, lambda_p, sigma_a, S))
print('P of 0:', P_func(0, lambda_p, sigma_a, S))



runs = 100


# region dependence on S
Ss = 6* np.linspace(1, 100, 100)
sigma_g_s = []
P_circle_s = []
Y_circle_s = []
P_zero_s = []
Y_zero_s = []
K_ofS = []

for i in range(runs):
    Si = Ss[i]
    sgs = lambda_p*(4*(Si-1)*sigma_a**2)**0.5
    sigma_g_s.append(sgs)
    P0i = P_func(0, lambda_p, sigma_a, Si)
    Y0i = Y_func(0, lambda_p, sigma_a, Si)
    P_zero_s.append(P0i)
    Y_zero_s.append(Y0i)

    xci = -2 + 2*sigma_a * Si**0.5
    #print(Y_func(xci, lambda_p, sigma_a, Si))
    Pci = P_func(xci, lambda_p, sigma_a, Si)
    Yci = Y_func(xci, lambda_p, sigma_a, Si)
    Y_circle_s.append(Yci)
    P_circle_s.append(Pci)

    K_ofS.append((Si)**0.5 * sigma_a)


# region dependence on sigma_a
sigma_as = np.linspace(0.01, 0.3, runs)
sigma_g_sigma = []
P_circle_sigma = []
Y_circle_sigma = []
P_zero_sigma = []
Y_zero_sigma = []
K_ofsigma = []

for j in range(runs):
    sigma_ai = sigma_as[j]
    sgs = lambda_p*(4*(S-1)*sigma_ai**2)**0.5
    sigma_g_s.append(sgs)
    P0i = P_func(0, lambda_p, sigma_ai, S)
    Y0i = Y_func(0, lambda_p, sigma_ai, S)
    P_zero_sigma.append(P0i)
    Y_zero_sigma.append(Y0i)

    xci = -2 + 2*sigma_ai * S**0.5
    #print(Y_func(xci, lambda_p, sigma_ai, S))
    Pci = P_func(xci, lambda_p, sigma_ai, S)
    Yci = Y_func(xci, lambda_p, sigma_ai, S)
    Y_circle_sigma.append(Yci)
    P_circle_sigma.append(Pci)

    K_ofsigma.append((S)**0.5 * sigma_ai)


# region dependence on lps
lps = np.linspace(-20, 0, runs)
sigma_g_l = []
P_circle_l = []
Y_circle_l = []
P_zero_l = []
Y_zero_l = []
K_ofl = []

print(S, sigma_a)
for k in range(runs):
    lpi = lps[k]
    print(lpi)
    sgl = lpi*(4*(S-1)*sigma_a**2)**0.5
    sigma_g_s.append(sgl)
    P0i = P_func(0, lpi, sigma_a, S)
    Y0i = Y_func(0, lpi, sigma_a, S)
    P_zero_l.append(P0i)
    Y_zero_l.append(Y0i)

    xci = -2 + 2*sigma_a * S**0.5
    #print(Y_func(xci, lambda_p, sigma_ai, S))
    Pci = P_func(xci, lpi, sigma_a, S)
    Yci = Y_func(xci, lpi, sigma_a, S)
    Y_circle_l.append(Yci)
    P_circle_l.append(Pci)

    K_ofl.append((S)**0.5 * sigma_a)


# region plot setup 



# text
box_par = dict(boxstyle='square', facecolor='white')
text_vars_s = str("$\lambda_2'=$"+ str(lambda_p) + '\n $\sigma_a =$'+ str(sigma_a))
text_vars_sigma = str("$\lambda_2'=$"+ str(lambda_p) + '\n $S =$'+ str(S))
text_vars_l = str('$\sigma_a =$'+ str(sigma_a) + '\n $S =$'+ str(S))

# for circle plot:
def gaussian(x, A, mu, sigma2):
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian


x = np.linspace(mean_g-10*sigma_g, mean_g+10*sigma_g, 300)
A = abs(1/ ((2*3.14159)**0.5 * sigma_g))
y = np.linspace(0, A, 10)



# region plots 

# dependence on S
plt.figure()
plt.plot(Ss, P_zero_s, '.', label = 'Probability of instability')
plt.plot(Ss, P_circle_s, '.', label = 'Probability of surpassing circle')
plt.plot(Ss, K_ofS, '-', alpha = 0.3, label = 'K')
plt.grid()
plt.xlabel('S=# of distinct species')
plt.ylabel('Probability')
plt.legend(loc='lower right')
plt.figtext(0.15, 0.75, text_vars_s, bbox=box_par)

# dependence on sigma_a
plt.figure()
plt.plot(sigma_as, P_zero_sigma, '.', label = 'Probability of instability')
plt.plot(sigma_as, P_circle_sigma, '.', label = 'Probability of surpassing circle')
plt.plot(sigma_as, K_ofsigma, '-', alpha = 0.3, label = 'K')
plt.grid()
plt.figtext(0.15, 0.75, text_vars_sigma, bbox=box_par)
plt.xlabel('$\sigma_a$ used in A matrix')
plt.ylabel('Probability')
plt.legend(loc='lower right')

# dependence on lp
plt.figure()
plt.plot(lps, P_zero_l, '.', label = 'Probability of instability')
plt.plot(lps, P_circle_l, '.', label = 'Probability of surpassing circle')
#plt.plot(lps, K_ofl, '-', alpha = 0.3, label = 'K')
plt.grid()
plt.figtext(0.15, 0.75, text_vars_l, bbox=box_par)
plt.xlabel("$\lambda_2'=$ nonzero eigenvalue of M' ")
plt.ylabel('Probability')
plt.legend(loc='upper center')



# circle edges 

# plot the gaussian 

plt.figure()
plt.plot(x, gaussian(x, A, mean_g, sigma_g**2), '-')
plt.plot(C_right*np.ones(10), y, '-')
plt.plot(C_left*np.ones(10), y, '-')
plt.grid()
plt.title('gauss of J vs edge of circle')

plt.show()
