#region preamble 
print('\n')
import numpy as np
import matplotlib.pyplot as plt 

# this code is intended to replicate the results of "Effect of population abundances on the stability of large random ecosystems"
# https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.022410

# endregion



# region variables
s = 20
runs = 200

r_set = 1
# A matrix
mud = -1
sigma2d = 0
K_set = 0.7
mu = 0
sigma2 = K_set**2/s

K = (s*sigma2)**0.5
print('complexity K=', K)

One = np.ones(s)

# final abundances
mux = 1
sigmax = 0.1

Aeigs = []
Meigs = []
Meigs_r = []
rs = []
rones = []
xfs_r = []

#endregion 
for run in range(runs):
    # region make arrays
    A = np.empty((s, s))
    xf = np.random.uniform(0.1, 2, s)

    for i in range(s):
        for j in range(s):
            if j == i: A[i][j] = np.random.normal(mud, sigma2d**0.5)
            elif j!= i: A[i][j] = np.random.normal(mu, sigma2**0.5)

    M = np.dot(np.diag(xf), A)

    # endregion

    Avals, no = np.linalg.eig(A)
    Mvals, no = np.linalg.eig(M)
    Aeigs.extend(Avals)
    Meigs.extend(Mvals)
    rs.extend(-np.dot(A, xf))
    rones.extend(-np.dot(A, One))

    xf_r = -r_set*np.dot(np.linalg.inv(A), One)
    xfs_r.extend(xf_r)
    Mvals_r, no = np.linalg.eig(np.dot(np.diag(xf_r), A))
    Meigs_r.extend(Mvals_r)



range = [min(np.min(rs), np.min(rones)), max(np.max(rs), np.max(rones))]

r_counts, r_be = np.histogram(rs, 100, range)
rone_counts, rone_be = np.histogram(rones, 100, range)


# region plot
plt.figure()
plt.plot(np.real(Aeigs), np.imag(Aeigs), '.', alpha =0.5)
plt.plot(np.real(Meigs), np.imag(Meigs), '.', alpha =0.5)
plt.plot(np.real(Meigs_r), np.imag(Meigs_r), '.', alpha =0.5)
plt.grid()

plt.figure()
plt.stairs(r_counts, r_be, fill=True)
plt.stairs(rone_counts, rone_be, fill=True, alpha=0.5)
plt.grid()


plt.show()
