# functions for lyapunov things. 

# LE spectrum:

import numpy as np
from scipy.integrate import solve_ivp
import math



def LE_spec(deriv, jac, x0, twarm, tend, ds, p, nLE=None, LEseq=False):
    '''
    deriv: derivative of system, should be f(t,x,p)
    jac: jacobian of system, should be J(t,x,p)
    x0: starting state vector
    tend: total time to run (inc. warmup)
    twarm: warm up time
    ds: renormalization interval 
    p: arguments for deriv, jac
    nLE: number of LEs to solve, i.e. number of orthonormal vectors to use 
    result: if True, outputs x, t from solver. 
    '''
    n = len(x0)     # dimension of system, i.e. number of LEs that will be computed 
    if nLE == None: nLE = n
    # times:
    nint_warm = int(twarm/ds)
    nint_sim = int((tend-twarm)/ds)
    t_warmup = np.linspace(0, twarm, nint_warm)
    t_sim = np.linspace(twarm, tend, nint_sim)
    if LEseq: 
        LE_time = []
        t_run = 0
        t_time = []

    # functions: 
    def dQdt(t, Q, x):
        Qq = np.reshape(Q, (n,nLE))
        dQq = np.dot(jac(t, x, *p), Qq)
        return dQq.flatten()
    
    def dYdt(t, Y, p):
        x = Y[:n]; Q = Y[n:]
        dY = np.append(deriv(t, x, *p), dQdt(t, Q, x))
        return dY

    # warmup:
    Qstart = np.random.uniform(0, 1, (n,nLE))
    Q, R = np.linalg.qr(Qstart)
    print('warming up system')
    Y = np.append(x0, Q)
    # evolve forward by step
    for i in range(nint_warm-1): 
        t1 = t_warmup[i]; t2 = t_warmup[i+1]
        if i%10 == 0: print('time = ', t1)
        sol = solve_ivp(dYdt, [t1, t2], Y, method='RK45', rtol=1e-9, atol=1e-9, args=(p,))
        xnew = sol.y[:n, -1]
        Qtemp = sol.y[n:, -1].reshape((n,nLE))
        Q, R = np.linalg.qr(Qtemp)
        Y = np.append(xnew, Q.flatten())
        

    gamma = np.zeros(nLE)

    # find LEs: 
    print('finding LEs')
    for i in range(nint_sim-1): 
        t1 = t_sim[i]; t2 = t_sim[i+1]
        if i%10 == 0: print('time = ', t1)
        sol = solve_ivp(dYdt, [t1, t2], Y, method='RK45', rtol=1e-9, atol=1e-9, args=(p,))
        xseg = sol.y[:n, :]
        tseg = sol.t
        xnew = sol.y[:n, -1]
        Qtemp = sol.y[n:, -1].reshape((n,nLE))
        Q, R = np.linalg.qr(Qtemp)
        # fix signs in Q: 
        Rdiag = np.diag(R)
        signs = np.sign(Rdiag)
        signs[signs==0] = 1
        Q = Q * signs[np.newaxis, :]
        R = R * signs[:, np.newaxis]
        gamma += np.log(np.abs(Rdiag))
        Y = np.append(xnew, Q.flatten())
        if LEseq: 
            t_run += ds
            LE_time.append(gamma / (t_run))
            t_time.append(t_run)
    T = ((nint_sim - 1) * (t_sim[1] - t_sim[0]))
    LE = gamma / T
    t = np.append(t_warmup, t_sim)
    if LEseq: 
        LE_time = np.array(LE_time)
        return LE, t_time, LE_time
    else:
        return LE
   
def LE_lead(deriv, jac, x0, twarm, tend, ds, p):
    # should be very similar to full spectrum, but using 1 leading vector rather than Q matrix
    '''
    deriv: derivative of system, should be f(t,x,p)
    jac: jacobian of system, should be J(t,x,p)
    x0: starting state vector
    tend: total time to run (inc. warmup)
    twarm: warm up time
    ds: renormalization interval 
    p: arguments for deriv, jac
    '''
    n = len(x0)     # dimension of system, i.e. number of LEs that will be computed 
    # times:
    nint_warm = int(twarm/ds)
    nint_sim = int((tend-twarm)/ds)
    t_warmup = np.linspace(0, twarm, nint_warm)
    t_sim = np.linspace(twarm, tend, nint_sim)
    def dVdt(t, V, x):
        dV = np.dot(jac(t,x,*p), V)
        return dV
    
    def dZdt(t, Z, p):
        x = Z[:n]; V = Z[n:]
        dZ = np.append(deriv(t, x, *p), dVdt(t, V, x))
        return dZ
    
    # warmup:
    V = np.random.uniform(0, 1, n)
    V /= np.linalg.norm(V)
    print('warming up system')
    Z = np.append(x0, V)

    # evolve forward by step
    for i in range(nint_warm-1): 
        t1 = t_warmup[i]; t2 = t_warmup[i+1]
        if i%5 == 0: print('time = ', t1)
        sol = solve_ivp(dZdt, [t1, t2], Z, method='RK45', rtol=1e-9, atol=1e-9, args=(p,))
        xnew = sol.y[:n, -1]
        Vtemp = sol.y[n:, -1]
        V = Vtemp/np.linalg.norm(Vtemp)
        Z = np.append(xnew, V)

    gamma1 = 0
    # find LEs: 
    print('finding LE')
    for i in range(nint_sim-1): 
        t1 = t_sim[i]; t2 = t_sim[i+1]
        if i%5 == 0: print('time = ', t1)
        sol = solve_ivp(dZdt, [t1, t2], Z, method='RK45', rtol=1e-9, atol=1e-9, args=(p,))
        xnew = sol.y[:n, -1]
        Vtemp = sol.y[n:, -1]
        Vnorm = np.linalg.norm(Vtemp)
        gamma1 += np.log(Vnorm)
        V = Vtemp/Vnorm
        Z = np.append(xnew, V)
    print('tsim=', t_sim)
    print('len t_sim=', len(t_sim))
    T = ((nint_sim - 1) * (t_sim[1] - t_sim[0]))
    LE1 = gamma1 / T
    return LE1

