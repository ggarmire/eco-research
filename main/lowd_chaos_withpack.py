# import libraries
import numpy as np
import matplotlib.pyplot as plt
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ



#define variables 
t0 = 0
dt = 0.01

global Aex, rex
rex = [1, 0.72, 1.53, 1.27]
Aex = [[1, 1.09, 1.52, 0], [0, 1, 0.44, 1.36], [2.33, 0, 1, 0.47], [1.21, 0.51, 0.35, 1]]
x0 = np.random.uniform(0.2, 0.8, 4)
n = 4


# function:
def fglv(x, t):
    n = len(x)
    rx = np.multiply(rex, x)
    dxdt = np.multiply(rx, (np.ones(n) - np.dot(Aex, x)))
    for i in range(0, len(x0)):
        if x[i] <= 0:
            dxdt[i] == 0
    return dxdt

# jacobian:

def jac(x, t):
    Ax = np.dot(Aex, x)
    xA = np.dot(np.diag(x), Aex)
    Jac = np.diag(rex) - np.diag(np.multiply(rex, Ax)) - np.dot(np.diag(rex), Ax)
    return Jac

glvsys = ContinuousDS(x0, t0, fglv, jac, dt)
glvsys.forward(10**5, False)

LCE, history = LCE(glvsys, 4, 0, 10**5, True)
print('LCE: ', LCE)

LCE_est = []
for i in range(n):
    LCE_est.append(np.mean(history[i, -100:-1]))
print('LCEs: ', LCE_est)


# Compute CLV
CLV, traj, checking_ds = CLV(glvsys, 3, 0, 10**5, 10**5, 10**5, True, check = True)



# Check CLV
LCE_check = np.zeros((glvsys.dim,))
for i in range(len(CLV)):
    W = CLV[i]
    init_norm = np.linalg.norm(W, axis = 0)
    W = checking_ds.next_LTM(W)
    norm = np.linalg.norm(W, axis = 0)
    checking_ds.forward(1, False)
    LCE_check += np.log(norm / init_norm) / checking_ds.dt
LCE_check = LCE_check / len(CLV)

print("Average of first local Lyapunov exponent: {:.3f}".format(LCE_check[0]))
print("Average of second local Lyapunov exponent: {:.3f}".format(LCE_check[1]))
print("Average of third local Lyapunov exponent: {:.3f}".format(LCE_check[2]))



# Plot of LCE
plt.figure(figsize = (10,6))
plt.plot(history[:5000])
plt.xlabel("Number of time steps")
plt.ylabel("LCE")
plt.title("Evolution of the LCE for the first 5000 time steps")
plt.grid()
plt.show()

