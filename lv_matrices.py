## Matrices! 

#%% libraries
import numpy as np
import random 

#%% A Matrix
def A_matrix(n, C, sig2, seed, LH):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    
    A = np.zeros((n, n))
    random.seed(seed)

    for i in range (0, n):
        for j in range (0, n):
            num = random.random()   # random number between 0 and 1. 
            if i==j:
                A[i][j] = -1        # Set diagonals to -1
            elif num < C:
                A[i][j] = np.random.normal(0, sig2**0.5)        # drawn from normal distribution with std=sigma
            else:
                A[i][j] = 0     # for connectedness
            if LH==1:
                if j == i+1 or j==i-1:
                    A[i][j] = -1
    #for r in A:
    #   print(r)
    return A

A = A_matrix(5, 0.5, 1, 10, LH=0)
for r in A:
    print(r)

print(np.max(A))


def M_matrix(n, muc, mua, f, g, seed):
    # n = number of species 
    # C = connectedness (prob of nonzero a_ij)
    #sig2 = sigma^2 of normal distribution of nonzero a_ij's
    
    M = np.zeros((n, n))
    random.seed(seed)

    for i in range (0, n-1):
        if i % 2 == 0:
            M[i][i] = -muc
            M[i+1][i+1] = -mua
            M[i][i+1] = f
            M[i+1][i] = g
        
    return M

'''M = M_matrix(6, 0.25, 0.27, 0.1, 0.2, 10)
for r in M: 
    print(r)'''


