## Matrices! 

#%% libraries
import numpy as np
import random 
from scipy import integrate
from decimal import Decimal, getcontext

#%% initial conditions 
def x0_vec(n, seed):
    np.random.seed(seed)
    x0 = np.random.normal(loc=1, scale=0.1, size=n)
    for i in range(n):
        while x0[i] <= 0: 
            x0[i] = np.random.normal(loc=1, scale=0.1)
    return x0

#%% Matrices 
def A_matrix(n, C, sig2, seed, LH):

    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    sig = sig2**0.5
    A = np.zeros((n, n))
    random.seed(seed)
    np.random.seed(seed)
    a = -1
    if LH ==1:
        for i in range (0, n, 2):   # set each block at once
            A[i][i] = a
            A[i+1][i] = a
            A[i][i+1] = a
            A[i+1][i+1] = a
            for j in range(0, n, 2):
                num = random.random()
                if A[i][j] == 0:
                    if num < C:
                        val = np.random.normal(0, sig)
                        A[i][j] = val
                        A[i+1][j+1] = val
                        A[i][j+1] = val
                        #A[i][j+1] = A[i][j]
                        A[i+1][j] = val
    
    if LH == 0:
        for i in range (0,n):
            A[i][i] = -1
            for j in range(0,n):
                num = random.random()
                if A[i][j] == 0:
                    if num < C:
                        A[i][j] = np.random.normal(0, sig)
                        

    #   print(r)
    return A

def A_matrix3(n, C, sig2, seed):

    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    sig = sig2**0.5
    A = np.zeros((n, n))
    random.seed(seed)
    np.random.seed(seed)

    for i in range (0, n, 3):   # set each block at once
        A[i][i] = -1; A[i+1][i] = -1; A[i+2][i] = -1
        A[i][i+1] = -1; A[i+1][i+1] = -1; A[i+2][i+1] = -1
        A[i][i+2] = -1; A[i+1][i+2] = -1; A[i+2][i+2] = -1
        for j in range(0, n, 3):
            num = random.random()
            if A[i][j] == 0:
                if num < C:
                    val = np.random.normal(0, sig)
                    A[i][j] = val; A[i][j+1] = val; A[i][j+2] = val
                    A[i+1][j] = val; A[i+1][j+1] = val; A[i+1][j+2] = val
                    A[i+2][j] = val; A[i+2][j+1] = val; A[i+2][j+2] = val

    return A


def M_matrix(n, muc, mua, f, g):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    
    M = np.zeros((n, n))


    for i in range (0, n-1):
        if i % 2 == 0:
            M[i][i] = muc
            M[i+1][i+1] = mua
            M[i][i+1] = f   
            M[i+1][i] = g
        
    return M


def M_matrix3(n, muc, mua, f, g):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    
    M = np.zeros((n, n))
    for i in range (0, n-1):
        if i % 3 == 0:
            M[i][i] = muc
            M[i+1][i+1] = muc
            M[i+2][i+2] = mua
            M[i][i+1] = 0
            M[i][i+2] = f
            M[i+1][i] = g
            M[i+2][i+1] = g
        
    return M

#M = M_matrix3(6, -0.5, -0.6, 1, 2)
#print(M)

#%% ODE solvers 
'''def lv_LH(x0, t, A, M): 
    def derivative(x, t, M, A):
        dxdt = np.dot(M, x) + np.multiply(x, np.dot(A, x))
        for i in range(0, len(x0)):
            if x[i] <= 0:
                x[i] == 0
                dxdt[i] == 0
        return dxdt
    result = integrate.odeint(derivative, x0, t, args = (M, A))
    return result'''

def lv_LH(x0, t, A, M): 
    def derivative(x, t, M, A):
        for i in range(len(x0)):
            if x[i] <= 0:
                x[i] = 0
        dxdt = np.dot(M, x) + np.multiply(x, np.dot(A, x))
        for i in range(len(x0)):
            if x[i] <= 0:
                  dxdt[i] = 0
        return dxdt
    result = integrate.odeint(derivative, x0, t, args = (M, A))
    return result

def lv_classic(x0, t, A, r): 
    
    def derivative(x, t, r, A):
        
        for i in range(0, len(x0)):
            if x[i] <=0:
                x[i] = 0

        dxdt = np.multiply(r, x) + np.multiply(x, np.dot(A, x))
        for i in range(0, len(x0)):
            if x[i] <= 0:
                dxdt[i] == 0
        return dxdt
    
    result = integrate.odeint(derivative, x0, t, args = (r, A))
    return result


#%% Jacobian
def LH_jacobian(A, M, xf):
    n = len(xf)
    delta = np.diag(np.dot(A, xf))
    Ax = np.multiply(np.outer(np.ones(n), xf), A)
    J = M + delta + Ax
    return J



