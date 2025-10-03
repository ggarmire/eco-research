# functions for lyapunov things. 

# LE spectrum:

import numpy as np
from scipy import integrate
import math


def LE_spec(deriv, jac, x0, twarm, tsim, dt, ds, p):
    n = len(x0)     # dimension of system, i.e. number of LEs that will be computed 
    # times:
    print(type(p), len(p), p)
    t_warmup = np.arange(0, twarm, dt)
    t = np.arange(twarm, tsim, dt)
    ntwarm = t_warmup.shape[0]
    nt = t.shape[0]
    Q = np.eye(n)
    # warmup:
    print('warming up...')
    warmup = integrate.odeint(deriv, x0, t_warmup, args=p)
    for i in range(ntwarm):
        xt = warmup[i, :]
        D = jac(xt, *p)
        Qt = np.dot(D, Q)
        if i%ds == 0:       # only do ONS every ds timesteps 
            Q, R = np.linalg.qr(Qt)
    # perform LE computation
    print('computing LEs...')
    gamma =np.zeros(n)
    xw = warmup[-1, :]
    xts = integrate.odeint(deriv, xw, t, args=p)
    for i in range(nt): 
        xt = xts[i, :]
        D = jac(xt, *p)
        Qt = np.dot(D, Q)
        if i%ds == 0:       # only do ONS every ds timesteps 
            Q, R = np.linalg.qr(Qt)
            gamma += np.log(np.abs(np.diag(R)))    
    LE= np.multiply(1/tsim, gamma)
    return LE
        












