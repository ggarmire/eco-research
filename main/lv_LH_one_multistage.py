
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix3
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math

seed = random.randint(0, 1000)
seed = 400
print('\n\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
s = 10
n2 = 2*s
n3 = s*3

K_set = 0.5
C = 1
sigma2 = K_set**2 / (s*C)
print("complexity: ", K_set, ', sigma2: ', sigma2)

# three stage variables 
mu1 = -0.5; mu2 = -0.6; mu3 = -0.7
f12 = 0.9; f13 = 1.1; f23 = 1.2
g21 = 0.9; g31 = 1; g32 = 1.2

r3 = 1.5

z32 = (r3*g21-mu3*g21+f23*g31)/(r3*g31-mu2*g31+g32*g21)
z31 = 1/r3 * (mu1 + f12*z32 + f13)
print('r3: ', r3, ', z:', z32, z31)


# two stage variables 
muc = -0.8; mua = -0.3
f = 1.5; g = 1
z2 = (muc-mua+((mua-muc)**2 +4*g*f)**0.5)/(2*g)

r2 = muc + f/z2
print('2 stage z = ', z2)
print('r2 = ', r2)




'''if abs(r3 - r2) > 1e-6:
    raise Exception("2 stage and 3 stage r dont match")'''
r = r2
print(r)

rvec = r * np.ones(s)
# endregion variables

# region 1 stage solution

A1 = A_matrix(s, C, sigma2, seed, LH = 0)

xf1 = -np.dot(np.linalg.inv(A1), rvec)

J1 = np.dot(np.diag(xf1), A1)
J1vals, trash = np.linalg.eig(J1)
print('max eigenvalue for 1 stage: ', np.max(np.real(J1vals)))

# endregion

# region 2 stage solution
A2 = A_matrix(n2, C, sigma2, seed, LH=1) 
xf2 = np.repeat(xf1, 2)   # make unscaled
J2 = np.dot(np.diag(xf2), A2)
J2vals, trash = np.linalg.eig(J2)
print('max eigenvalue for 2 stage:', np.max(np.real(J2vals)))

# endregion


# region 3 stage solution 
A3 = A_matrix3(n3, C, sigma2, seed)
xf3 = np.repeat(xf1, 3)   # make unscaled
J3 = np.dot(np.diag(xf3), A3)
J3vals, trash = np.linalg.eig(J3)
print('max eigenvalue for 2 stage:', np.max(np.real(J3vals)))

# endregion

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.5)
plt_text = str('r1 = '+str('%0.2f'%r)+'\nr2 = '+str('%0.2f'%r2)+', z2 = '+str('%0.2f'%z2)+'\nr3 = '+str('%0.2f'%r3)+', z3 = '+str('%0.2f'%z31)+', '+str('%0.2f'%z32))

plt.figure()
plt.plot(np.real(J1vals), np.imag(J1vals), '.', label = '1 stage')
plt.plot(np.real(J2vals), np.imag(J2vals), '.', label = '2 stage')
plt.plot(np.real(J3vals), np.imag(J3vals), '.', label = '3 stage')
plt.plot(2*np.real(J1vals), 2*np.imag(J1vals), 'o', mfc = 'None', label = '2*1 stage')
plt.plot(3*np.real(J1vals), 3*np.imag(J1vals), 'o', mfc = 'None', label = '3*1 stage')
plt.figtext(0.15, 0.75, plt_text, bbox=box_par)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.title('Eigenvalues of Jacobian, multi-stage structure')
plt.grid()
plt.legend()

plt.show()