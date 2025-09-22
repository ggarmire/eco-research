
# libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
import random 
from matplotlib.colors import LogNorm

#region Set Variables 

# stuff that gets changed: 
n = 50
runs = 500

# A matrix 

K_set = 0.
C = 1

# M matrix 
muc = -0.5
mua = -0.5
f = 1
g = 1

z = 1.224744871391589
z=1

xstar = 0       # 1 constrains the final pops 
zrand = 0      # 1 randomizes juvinile fractions
runode = 1      # 1 runs the ODE solver for each A matrix (not necessary for xstar=0)


# stuff that does not get changed:
s = n/2
x0 = x0_vec(n, 1)
sigma2 = K_set**2/n*2
xs = np.ones(n)
for i in range(0, n, 2):
    xs[i] = z
'''if zrand == 1:  
    np.random.seed(4)
    xs = np.random.uniform(0.8, 2, n)'''
t = np.linspace(0, 300, 600)
M_pre = M_matrix(n, muc, mua, f, g)

One = np.ones(n)

print('n:', n, ', sigma:', '%.3f'%(sigma2**0.5))

print('xs:', xs)

#region make arrays 
eigs_J = []     # eigenvalues of the jacobian 
eigs_J_died = []        # eigenvalues when not all species survive 


eigs_M = []     # eigenvlaues of M after scaling 
eigs_Mp = []        # eigenvalues of M+delta term 
eigs_A = []     # eigenvalues of A 

A_rowsums = []      # rowsums of A 
Ars_max = []        # max rowsum of A 
Ars_max_unphysical = []        # max rowsum of A 
Ars_max_scaled = []        # max rowsum of A 

Ars_max = []        # max rowsum of A 
Ars_max_unphysical = []        # max rowsum of A 

Ars_max_physical_stable = []        # max rowsum of A 
Ars_max_unphysical_stable = []        # max rowsum of A 
Ars_max_physical_unstable = []        # max rowsum of A 
Ars_max_unphysical_unstable = []        # max rowsum of A 

maxeig_J_physical_stable = []
maxeig_J_unphysical_stable = []
maxeig_J_physical_unstable = []
maxeig_J_unphysical_unstable = []

maxeig_J = []       # max eigenvalue for each run

maxeig_A = []
maxeig_M = []
maxeig_Mp = []

maxeig_J_complex = []       # max eigenvalue, storing the complex value
maxeig_J_complex_died = []

n_survives = []     # number of subspecies that survive 
n_survives_noscale = []     # number of subspecies that survive 
# endregion


# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %87 == 0:
        print(run)

    if zrand == 1:  
        xs = np.random.uniform(0.8, 2, n)

    # make A matrix 
    A = A_matrix(n, C, sigma2, seed, LH=1)      #random a matrix 
    A_rows = np.dot(A, One)
    A_rows_scaled = np.dot(A, xs)
    A_rowsums.extend(A_rows)
    Ars_max_scaled.append(np.max(A_rows_scaled))
    Ars_max_run = np.max(A_rows)
    Ars_max.append(Ars_max_run)


    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)

    # scale M matrix for equal row sums 
    if xstar == 0:
        M = M_pre
    elif xstar == 1:
        #normal scaling: scale each row of M
        M_rows = np.dot(M_pre, xs)
        scales = -np.divide(np.multiply(A_rows_scaled, xs), M_rows)
        M = np.multiply(M_pre, np.outer(scales, np.ones(n)))

    # calculate jacobian given final values 
    if xstar == 1:
        Jac = LH_jacobian(n, A, M, xs) 
        Jvals, Jvecs = np.linalg.eig(Jac) 
    
    stable = 1
    # run ODE solver 
    ranode = 0
    if xstar == 0 or runode == 1:
        ranode = 1

        result_noscale = lv_LH(x0, t, A, M_pre)     # ODE with unscaled M
        xf_noscale = result_noscale[-1, :]      

        result = lv_LH(x0, t, A, M)         # with scaled M
        xf = result[-1, :]


        # metrics after solving 
        if xstar == 0:
            #Jac = LH_jacobian_norowsum(xf_noscale, A, M)
            Jac = LH_jacobian(xf_noscale, A, M)
            Jvals, Jvecs = np.linalg.eig(Jac) 
        n_survive = n
        n_survive_noscale = n
        for species in range(n):
            if xf[species] < 1e-3:
                n_survive -= 1
            if xf_noscale[species] < 1e-3:
                n_survive_noscale -= 1
        n_survives.append(n_survive)
        n_survives_noscale.append(n_survive_noscale)
        if n_survive < n:
            eigs_J_died.extend(Jvals)
            maxeig_J_complex_died.append(np.max(Jvals))
            stable = 0
    
    if xstar == 0:      
        Mp = M + np.diag(np.dot(A, xf_noscale))        
    elif xstar == 1:
        Mp = M + np.diag(A_rows_scaled)

    Mvals, Mvecs = np.linalg.eig(M)
    Mpvals, Mpvecs = np.linalg.eig(Mp)
    
    eigs_M.extend(Mvals)
    eigs_Mp.extend(Mpvals)


    #print('size of jacobian: ', np.size(Jac))
    eigs_J.extend(Jvals)

    maxeig_J_run = np.max(Jvals).real

    unphysical = 0
    if np.max(np.diag(M)) > 0 or np.min(np.diag(M, 1)) < 0 or np.min(np.diag(M, -1)) < 0:
    #if np.min(np.diag(M, 1)) < 0 or np.min(np.diag(M, -1)) < 0:
        unphysical = 1

    if maxeig_J_run > 0: 
        stable == 0

    #print(Ars_max_run, maxeig_J_run, n_survive)
    if unphysical == 0:
        if stable == 1:
            Ars_max_physical_stable.append(Ars_max_run)
            maxeig_J_physical_stable.append(maxeig_J_run)
        elif stable == 0: 
            Ars_max_physical_unstable.append(Ars_max_run)
            maxeig_J_physical_unstable.append(maxeig_J_run)
    elif unphysical == 1:
        if stable == 1:
            Ars_max_unphysical_stable.append(Ars_max_run)
            maxeig_J_unphysical_stable.append(maxeig_J_run)
        elif stable == 0: 
            Ars_max_unphysical_unstable.append(Ars_max_run)
            maxeig_J_unphysical_unstable.append(maxeig_J_run)

    

    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10)      # nonzero eigenvalues of the two, since the zeros dissapear later 
    Mpvalsm = np.ma.masked_inside(Mpvals, -1e-10, 1e-10)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))
    maxeig_J_complex.append(np.max(Jvals))

    if Ars_max_run > 0 and stable == 0:
        print('seed: ', seed, ', max rs:', Ars_max_run)


        

# region sort eigenvalues

eigs_real_axis = []     # eigenvalues of J on the real line
eigs_complex = []       # eigenvalues not on the real line

eigs_real_axis_M = [] 
eigs_real_axis_Mp = [] 
eigs_real_axis_A = [] 


nzeros = 0
for i in range(len(eigs_J)):
    if abs(eigs_J[i].imag) <= 1e-7:
        nzeros += 1
        eigs_real_axis.append(eigs_J[i].real)
    elif abs(eigs_J[i].imag) > 1e-7:
        eigs_complex.append(eigs_J[i])
    if abs(eigs_M[i].imag) <= 1e-7:
        eigs_real_axis_M.append(eigs_M[i].real)
    if abs(eigs_Mp[i].imag) <= 1e-7:
        eigs_real_axis_Mp.append(eigs_Mp[i].real)
    if abs(eigs_A[i].imag) <= 1e-7:
        eigs_real_axis_A.append(eigs_A[i].real)
    
#print(nzeros)
eigs_abs = np.abs(eigs_complex)

r_mean = np.mean(np.real(eigs_complex))
i_mean = np.mean(np.imag(eigs_complex))


# region make histograms 

# settings:
nbins = 70
histmin = min(np.min(np.real(eigs_Mp)), np.min(np.real(eigs_A)), np.min(np.real(eigs_J)))
histmax = max(np.max(np.real(eigs_Mp)), np.max(np.real(eigs_A)), np.max(np.real(eigs_J)))
histrange = (histmin, histmax)

# J eigs: real axis 
J_counts, J_be = np.histogram(np.real(eigs_J), bins=nbins, range = histrange)
J_bc = np.real((J_be[:-1] + J_be[1:]) / 2)
# M eigs: real axis 
M_counts, M_be = np.histogram(np.real(eigs_M), bins=nbins, range = histrange)
M_bc = np.real((M_be[:-1] + M_be[1:]) / 2)
# A eigs: real axis 
A_counts, A_be = np.histogram(np.real(eigs_A), bins=nbins, range = histrange)
A_bc = np.real((A_be[:-1] + A_be[1:]) / 2)
# M prime eigs: real axis 
Mp_counts, Mp_be = np.histogram(np.real(eigs_Mp), bins=nbins, range = histrange)
Mp_bc = np.real((Mp_be[:-1] + Mp_be[1:]) / 2)

print('counts J: ', np.sum(J_counts), 'A: ', np.sum(A_counts), ', Mp: ', np.sum(Mp_counts))

bw = J_be[2] - J_be[1]      # bin width






# region fitting 

# analytical eigenvalues of M
m1 = 1
m2 = z*(z+1)/2 * (muc*mua - f*g)/((muc*z+f)*(mua+g*z)) 

# set function 
def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma2: variance of Gaussian
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian

# find bin that contains 0 and circle edges
b1_circ = np.digitize(-2 - 2.5*K_set, J_bc)
b2_circ = np.digitize(-2 + 2.5*K_set, J_bc)


b0 = np.digitize(0, Mp_bc)

# fit J with a gauss, away from the circle 
fitjx = []; fitjy = []
fitjx.extend(J_bc[0:b1_circ]); fitjx.extend(J_bc[b2_circ:-1])
fitjy.extend(J_counts[0:b1_circ]); fitjy.extend(J_counts[b2_circ:-1])

# fit M prime with a gaussian, away from 0
fitmx = []; fitmy = []
fitmx.extend(Mp_bc[0:b0-2]); fitmx.extend(Mp_bc[b0+2:-1]); 
fitmy.extend(Mp_counts[0:b0-2]); fitmy.extend(Mp_counts[b0+2:-1]); 

# M' parameters for fit guessing:
l2p = m2 - (z+1)/2
s2M = l2p**2 * 4*(s-1)*sigma2
print('l2p', l2p, 's', s, 'sigma2', sigma2, 's2m:', s2M)
A_guess = s * runs / ((2*3.14159*s2M)**0.5) * bw


p0_Mp = [A_guess, 2*l2p, s2M]
print('pars guess for M prime:', p0_Mp)

# fit Mprime and J with the same guess 
#pars_Mp, cov_Mp = curve_fit(gaussian, fitmx, fitmy, p0_Mp, maxfev=5000)
#pars_J, cov_J = curve_fit(gaussian, fitjx, fitjy, p0_Mp, maxfev=5000)
#print('fit parameters of M prime:', pars_Mp)
#print('fit parameters of J gauss:', pars_Mp)



#region plotting setup 

# texts 
Jeigs_title = str('Eigenvalues of J: n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
maxrs_title = str('Max Eigenvalues vs. rowsum: n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
if zrand == 1: maxrs_title = str('Max Eigenvalues vs. rowsum: n='+str(int(n/2))+'*2, random constraint, K='+str(K_set))
Jhist_title = str('Real eigenvalues of J: : n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+'; '+str('%.0f'%(np.max(nzeros/n/runs*100)))+'% on x=0')
#Jfit_text = str('J fit mean: '+str('%0.3f'%pars_J[1])+', sigma^2: '+str('%0.3f'%pars_J[2])+', A= '+str('%0.3f'%pars_J[0]))
#Mfit_text = str("M' fit mean: "+str('%0.3f'%pars_Mp[1])+', sigma^2: '+str('%0.3f'%pars_Mp[2])+', A= '+str('%0.3f'%pars_Mp[0]))

box_par = dict(boxstyle='square', facecolor='white')
text_vars = str("$\lambda_2'=$"+ str(m2-1) + '\n $\sigma_a =$'+ str(sigma2**0.5)+'\n s = '+str(n/2))
#text_fitpars_M = str('M fit mean: '+str('%0.3f'%parsgm[1])+', sigma^2: '+str('%0.3f'%parsgm[2])+', A= '+str('%0.3f'%parsgm[0]))

text_species_survive = str('n='+str(n)+', K='+str(K_set))


max_ax = np.max(np.imag(eigs_J)) + 3
min_ax = np.min(np.imag(eigs_J)) - 3

fsize = (6,6)


min_jvals = np.min(maxeig_J)
max_jvals = np.max(maxeig_J)
yjvals = np.linspace(min_jvals-0.2, max_jvals+0.2, 5)



#region plotting 

# fig: distribution of all eigenvalues 
plt.figure(figsize=fsize)
plt.grid()
plt.title(Jeigs_title)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.plot(np.real(eigs_J), np.imag(eigs_J), 'o', ms=2, color ='C0', alpha=0.5, label='J')
plt.plot(np.real(eigs_A), np.imag(eigs_A), 'o', ms=2, color = 'C2', alpha=0.5, label="A")
plt.plot(np.real(eigs_Mp), np.imag(eigs_Mp), 'o', ms=2, color = 'C1', alpha=0.5, label="M'")

#plt.plot(np.real(eigs_J_died), np.imag(eigs_J_died), 'o', ms=2, label='some species died')
#plt.plot(np.real(eigs_Mp), np.imag(eigs_Mp), 'o', ms=2, alpha=.3, label = "M'")
#plt.plot(np.real(eigs_A), np.imag(eigs_A), 'o', ms=2, alpha=.3, label = 'A')
plt.figtext(0.13, 0.12, mpar_text)
plt.legend()
#plt.figtext(0.13, 0.86, plot_text_2)
#plt.xlim([-15, 3])
#plt.ylim([-9, 9])
#plt.tight_layout()


#figure: compex max eig 
plt.figure(figsize=fsize)
plt.plot(np.real(maxeig_J_complex), np.imag(maxeig_J_complex), '.', label = 'all survive')
plt.plot(np.real(maxeig_J_complex_died), np.imag(maxeig_J_complex_died), '.', label ='some died')
plt.grid()
plt.legend()
plt.title('max Eigenvalue of J, in complex plane')
plt.xlabel('real component')
plt.ylabel('imaginary component')


# fig: histogram of the eigenvalues on the real axis. im(eig)=0
plt.figure(figsize=fsize)
plt.stairs(J_counts, J_be, fill = True, alpha=0.8, label = 'J')
plt.stairs(A_counts, A_be, alpha=0.8, label = 'A')
plt.stairs(Mp_counts, Mp_be, alpha=0.8, label = "M'")
plt.title(Jhist_title)
plt.figtext(0.13, 0.66, mpar_text)
'''if xstar == 1 and zrand == 0:
    plt.plot(fitmx, gaussian(fitmx, *pars_Mp), '-', label="fit of M'")
    plt.figtext(0.13, 0.63, Jfit_text)
    plt.figtext(0.13, 0.60, Mfit_text)'''
plt.xlabel('real component of eigenvalue')
plt.ylabel('counts')
plt.legend()
plt.grid()

# region plots for xstar = 1
if xstar == 1:
    #figure: max eig of A/Mp vs max eig of J
    plt.figure(figsize=fsize)
    plt.plot(yjvals, yjvals, '--', label= 'x=y')
    plt.plot(maxeig_A, maxeig_J, '.', label = 'A')
    plt.plot(maxeig_Mp, maxeig_J, '.', label = "M'")
    plt.grid()
    plt.legend(loc='upper left')
    plt.xlabel(' Max (nonzero) eigenvalue of A, Mprime')
    plt.ylabel('Max eigenvalue of J')
    plt.figtext(0.15, 0.15, text_vars, bbox=box_par)


    '''#figure: max eig of A vs max eig ofMp
    plt.figure(figsize=fsize)
    plt.plot(maxeig_A, maxeig_Mp, '.')
    plt.grid()
    plt.xlabel(' Max (nonzero) eigenvalue of A')
    plt.ylabel('Max eigenvalue of Mprime')
    plt.figtext(0.15, 0.15, text_vars, bbox=box_par)'''

    #figure: max rowsum of A vs max nonzro eigenvalue of J'
    plt.figure(figsize=fsize)
    plt.plot(np.linspace(np.min(Ars_max), np.max(Ars_max), 3), 0*np.linspace(np.min(Ars_max), np.max(Ars_max), 3), '--k')
    plt.plot(0*np.linspace(np.min(maxeig_Mp), np.max(maxeig_Mp), 3), np.linspace(np.min(maxeig_Mp), np.max(maxeig_Mp), 3), '--k')
    #plt.plot(Ars_max, maxeig_J, '.', color='C0', label = "J eigenvalues")
    plt.plot(Ars_max_physical_unstable, maxeig_J_physical_unstable, '.', color = 'C0',  label = 'physical M, unstable: '+str(len(Ars_max_physical_unstable)))
    plt.plot(Ars_max_physical_stable, maxeig_J_physical_stable, '.', color = 'C2', label = 'physical M, stable: '+str(len(Ars_max_physical_stable)))
    plt.plot(Ars_max_unphysical_unstable, maxeig_J_unphysical_unstable, '.', color = 'C3', label = 'unphysical M, unstable: '+str(len(Ars_max_unphysical_unstable)))
    plt.plot(Ars_max_unphysical_stable, maxeig_J_unphysical_stable, '.', color = 'C4', label = 'unphysical M, stable: '+str(len(Ars_max_unphysical_stable)))
    plt.grid()
    plt.title(maxrs_title)
    plt.legend(loc='upper left')
    plt.xlabel(' Max rowsum of A')
    plt.ylabel("Max real eigenvalue of J'")


if ranode == 1 or xstar == 0:
    plt.figure(figsize=fsize)
    plt.plot(n_survives, maxeig_J, '.')
    plt.grid()
    plt.xlabel('number of subspecies with nonzero final population')
    plt.ylabel('Max eigenvalue of Jacobian')

    plt.figure(figsize=fsize)
    plt.hist2d(n_survives_noscale, n_survives, bins=n, norm=LogNorm(), cmap='viridis')
    plt.colorbar()
    plt.grid()
    plt.title('Species survival with/without final populations constrained ')
    plt.xlabel('Number of species remaining, without constraint')
    plt.ylabel('Number of species remaining, with constraint')
    plt.figtext(0.13, 0.12, text_species_survive)



if xstar == 0:
    plt.figure(figsize=fsize)
    plt.plot(n_survives, Ars_max, 'o')
    plt.xlabel('Number of subspecies remaining')
    plt.ylabel('Max rowsum in A')
    plt.grid()


plt.show()
#endregion

 


    


