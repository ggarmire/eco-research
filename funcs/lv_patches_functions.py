# functions for https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007827
import numpy as np

# region generate initial conditions (random vector)
def x0_vector(s, M, seed): 
    rng = np.random.default_rng(seed)
    x0vec = np.abs(rng.normal(1, 0.2, (s,M)).flatten())
    return x0vec

# region generate matrices 
def A_matrix_patches(s, M, C, mean, sigma, rho, seed):
    # returns sxsxM array of interaction terms - 
    ''' 
    s: number of species 
    M: number of patches
    C: connectedness: probability of nonzero interaction between any two species
    mean: center of (nonzero) Aijs 
    sigma: std of (nonzero) Aijs
    rho: correlation between patches
    seed: seed random number generator for repeatability 
    ''' 
    rng = np.random.default_rng(seed)
    corr = np.full((M,M), rho)      # matrix of correlation values between patches 
    np.fill_diagonal(corr, 1)       # correlation is 1 between element and itself
    mask = rng.random((s,s)) < C        # C is fraction of nonzero entries in A 
    A = np.zeros((s,s,M))
    for i in range(s):      # made choice to sample diagonal elements the same way,since it is not specified in paper and self interactions are encompassed in another term.
        for j in range(s):
            if not mask[i,j]: continue
            # correlated off-diagonals:
            aij_flux = rng.multivariate_normal(np.zeros(M), corr)       # M numbers centered at 0, correlation rho
            A[i,j,:] = mean/s + sigma/np.sqrt(s) * aij_flux     # variables centered at mu/S
    return(A)

def D_matrix_patches(s, M, d):
    '''create migration matrix (sxMxM) between patches: all the same value everywhere in model
    s: number of species 
    M: number of patches
    d: dispersal rate 
    '''
    Dval = d/(M-1)      # simply the same value everywhere in D! diagonal elements dont matter
    D = np.full((s,M,M), Dval)
    for i in range(s):
        np.fill_diagonal(D[i], 0)
    return D

# endregion

# region derivatives 

def dxdt_multipatch(t, x, B, A, D, Nc):
    '''Derivative from model: evolves system
    x = current state (populations)
    B, A, D: input matrices above 
    Nc: cutoff for populations: if species is below Nc in each patch, species is considered extinct. 
    '''
    x = np.reshape(x, B.shape)
    Ax = np.einsum('iju, ju->iu', A, x)
    Dx = np.einsum('iuv, iv->iu', D,x) - x*np.einsum('iuv->iu', D)
    dxdt = x * (B - x - Ax) + Dx   # differential equation for multi-patch formulation
    extinct = np.all(x <= Nc, axis=1)
    dxdt[extinct,:] = 0
    x[extinct,:] = 0
    return dxdt.flatten()

def dxdt_singlepatch(t, x, B, A, Nc):
    ''' Derivative for single patch, for validation of model in paper.
    '''
    x = np.reshape(x, B.shape)
    Ax = np.einsum('iju, ju->iu', A, x) # same as above, but where there is only 1 patch it is simpler 
    dxdt = x * (B - x - Ax)      
    dxdt = np.where(x <= Nc, 0.0, dxdt)      
    return dxdt.flatten()

# region Jacobian 
def Jacobian_multipatch(t, x, B, A, D, Nc):
    '''Jacobian of model: used to evolve orthonormal system. sM x sM matrix.
    Must have same inputs as derivative for LE solver. 
    '''
    s, M = B.shape
    x = np.reshape(x, (s,M))
    J = np.zeros((s*M, s*M))
    OneM = np.ones(M)
    for u in range(M):
        for v in range(M):
            if v == u:      # on diagonal blocks come from same-patch interactions 
                Au = A[:,:,u]
                xu = x[:,u]
                Bu = B[:,u]
                Du = D[:,u,:]
                diaguu = Bu - 2*xu - np.dot(Au, xu) - np.dot(Du, OneM) + D[:,u,u]
                uvblock = -np.dot(np.diag(xu), Au) + np.diag(diaguu)
            elif v != u:        # off diagonal blocks only come from migration terms
                uvblock = np.diag(D[:,u,v])
            J[u*s:(u+1)*s, v*s:(v+1)*s] = uvblock
    return J


# endregion