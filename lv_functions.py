## Matrices! 

#%% libraries
import numpy as np
import random 
from scipy import integrate
from decimal import Decimal, getcontext


#%% A Matrix
def A_matrix(n, C, sig2, seed, LH):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    
    A = np.zeros((n, n))
    random.seed(seed)
    np.random.seed(seed)
    if LH ==1:
        for i in range (0, n, 2):   # set each block at once
            A[i][i] = -1
            A[i+1][i] = -1
            A[i][i+1] = -1
            A[i+1][i+1] = -1
            for j in range(0, n, 2):
                num = random.random()
                if A[i][j] == 0:
                    if num < C:
                        A[i][j] = np.random.normal(0, sig2**0.5)
                        A[i+1][j] = A[i][j]
                        A[i][j+1] = np.random.normal(0, sig2**0.5)
                        #A[i][j+1] = A[i][j]
                        A[i+1][j+1] = A[i][j+1]
    
    if LH == 0:
        for i in range (0,n):
            A[i][i] = -1
            for j in range(0,n):
                num = random.random()
                if A[i][j] == 0:
                    if num < C:
                        A[i][j] = np.random.normal(0, sig2**0.5)
                        

    #   print(r)
    return A

'''A = A_matrix(4, 0.5, 0.2, 7, LH=1)
for r in A:
    print(r)

print(np.max(A))'''


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

#M = M_matrix(6, 0.1, 0.2, 0.3, 0.4)
'''for r in M:
    print(r)'''

def lv_LH(x0, t, A, M): 
    
    def derivative(x, t, M, A):
        
        for i in range(0, len(x0)):
            if x[i] <=0:
                x[i] = 0
        dxdt = np.dot(M, x) + np.multiply(x, np.dot(A, x))
        return dxdt
    
    result = integrate.odeint(derivative, x0, t, args = (M, A))
    return result

def lv_classic(x0, t, A, r): 
    
    def derivative(x, t, r, A):
        
        for i in range(0, len(x0)):
            if x[i] <=0:
                x[i] = 0
        dxdt = np.multiply(r, x) + np.multiply(x, np.dot(A, x))
        return dxdt
    
    result = integrate.odeint(derivative, x0, t, args = (r, A))
    return result





