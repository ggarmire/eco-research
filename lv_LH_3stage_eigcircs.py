
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from lv_functions import A_matrix
from lv_functions import A_matrix3
from lv_functions import M_matrix3
from lv_functions import lv_LH
from lv_functions import x0_vec
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
import random 

#region variables to change
K_set = 0.5
muc = -0.5
mua = -0.5
f = 1.5
g = 1

z = 0.7

xstar = 1
n = 48

runs = 100
#endregion

random.seed(1)
#region set up other variables
s = n/3
x0 = x0_vec(n)
C = 1
sigma2 = K_set**2/n*3
print('n:', n, ', sigma:', '%.0f'%(sigma2**0.5))
M_pre = M_matrix3(n, muc, mua, f, g)
xs = np.ones(n)
for species in range(0, n, 3):  
    xs[species] = z
    xs[species+1] = z
print(xs)

t = np.linspace(0, 200, 200)

eigs = []
eigs_died = []

eigs_M = []
eigs_Mp = []
eigs_A = []

A_rowsums = []
Ars_max = []


maxeig_J = []
maxeig_A = []
maxeig_M = []
maxeig_Mp = []

maxeig_J_complex = []
maxeig_J_complex_died = []

n_survives = []
final_match = []


#endregion 


# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %232 == 0:
        print(run)

    # make matrices 
    A = A_matrix3(n, C, sigma2, seed) 
    A_rows = np.dot(A, xs)
    A_rowsums.extend(A_rows)
    Ars_max.append(np.max(A_rows))
    if xstar == 0:
        M = M_pre
    elif xstar == 1:
        M_rows = np.dot(M_pre, xs)
        scales = -np.divide(np.multiply(A_rows, xs), M_rows)
        M = np.multiply(M_pre, np.outer(scales, np.ones(n)))
    Mp = M + np.diag(A_rows)
    Mpvals, Mpvecs = np.linalg.eig(Mp)

    Mvals, Mvecs = np.linalg.eig(M)
    
    eigs_M.extend(Mvals)
    eigs_Mp.extend(Mpvals)

    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)
    if xstar == 1:
        Jac = LH_jacobian(n, A, M, xs) 
    #elif xstar == 0:        # in this case you actually need to run ODE solver and get approx. final state
    
    result = lv_LH(x0, t, A, M)
    xf = result[-1, :]
    match = 1
    if xstar == 0:
        Jac = LH_jacobian_norowsum(xf, A, M)
    n_survive = n
    for species in range(n):
        if abs(xf[species] - xs[species]) > 1e-3:
            match = 0 
        if xf[species] < 1e-3:
            n_survive -= 1
    n_survives.append(n_survive)
    final_match.append(match)


    Jvals, Jvecs = np.linalg.eig(Jac)   

    eigs.extend(Jvals)
    if n_survive < n:
        eigs_died.extend(Jvals)
        maxeig_J_complex_died.append(np.max(Jvals))

    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10)
    Mpvalsm = np.ma.masked_inside(Mpvals, -1e-10, 1e-10)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))
    maxeig_J_complex.append(np.max(Jvals))

    if np.max(Jvals) > 0:
        print('seed:', seed, 'max real eig:',np.max(Jvals), 'species left:', n_survive, 'match?', match)

    
#endregion

# region post analysis 
eigs_real = np.real(eigs)       # eigs of J
eigs_imag = np.imag(eigs)

eigs_real_axis = []     # eigenvalues of J on the real line
eigs_complex = []       # eigenvalues not on the real line

eigs_real_axis_M = [] 
eigs_real_axis_Mp = [] 
eigs_real_axis_A = [] 


nzeros = 0
for i in range(len(eigs_imag)):
    if abs(eigs_imag[i]) <= 1e-7:
        nzeros += 1
        eigs_real_axis.append(eigs[i].real)
    elif abs(eigs_imag[i]) > 1e-7:
        eigs_complex.append(eigs[i])
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
#print(r_mean, ', ', i_mean)


# fit histograms: 

# histograms: 
nbins1 = 70
nbins2 = 70
histrange = (np.min(eigs_real), max(np.max(eigs_real), np.max(eigs_real_axis_M)))

def gaussian_box(x, A, mu, sigma, h, c, w):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma**2))
    boxcar = h * ((x >= c-w/2) & (x <= c+w/2)).astype(float)
    return gaussian + boxcar

def gaussian(x, A, mu, sigma2):
    """
    A: Amplitude of Gaussian
    mu: Mean of Gaussian
    sigma: Std dev of Gaussian
    h = height of box 
    c = center of box 
    w = width of box 
    """
    gaussian = A * np.exp(-((x - mu)**2) / (2 * sigma2))
    return gaussian
    


counts, bin_edges = np.histogram(np.real(eigs_real_axis), bins=nbins1, range = histrange)
bin_centers = np.real((bin_edges[:-1] + bin_edges[1:]) / 2)
print('bin width = ', bin_edges[1]-bin_edges[0])

mp_counts, mp_bin_edges = np.histogram(np.real(eigs_Mp), bins=nbins1, range = histrange)
mp_bin_centers = (mp_bin_edges[:-1] + mp_bin_edges[1:]) / 2

b1 = np.digitize(-2 - 3*K_set, bin_edges)
b2 = np.digitize(-2 + 2.5*K_set, bin_edges)

binwidth_mp = mp_bin_edges[2] - mp_bin_edges[1]
bend_mp = int((-0.1 - bin_edges[0])/binwidth_mp)
bstart_mp = int((0.1 - bin_edges[0])/binwidth_mp)+2
print(bend_mp, bstart_mp)

#print(b1, b2)

r1 = bin_centers[:b1]; c1 = counts[:b1]
r2 = bin_centers[b2:]; c2 = counts[b2:]
fitx = []; fity = []
fitx.extend(r1); fitx.extend(r2)
fity.extend(c1); fity.extend(c2)

fitmx = []; fitmy = []
fitmx.extend(mp_bin_centers[0:bend_mp]); fitmx.extend(mp_bin_centers[bstart_mp:-1]); 
fitmy.extend(mp_counts[0:bend_mp]); fitmy.extend(mp_counts[bstart_mp:-1]); 

# M' parameters for fit guessing:
l2p = z*(z+1)/2 * (muc*mua - f*g)/((muc*z+f)*(mua+g*z)) - (z+1)/2
s2M = l2p**2 * 4*(s-1)*sigma2
print('l2p', l2p, 's', s, 'sigma2', sigma2, 's2m:', s2M)
A_guess = s * runs / ((2*3.14159*s2M)**0.5) * binwidth_mp



p01g = [A_guess, 2*l2p, s2M]
print('pars guess:', p01g)

p01gm = p01g
    


parsg, covg = curve_fit(gaussian, fitx, fity, p01g, maxfev=5000)
print('K: ', K_set, ', pars of gauss of J: ', parsg)

pars_mp, covs_mp = curve_fit(gaussian, fitmx, fitmy, p01g, maxfev=5000)
print('pars of M fit:', pars_mp)

# count number in circle
m1 = 1
m2 = z*(z+1)/2 * (muc*mua - f*g)/((muc*z+f)*(mua+g*z)) 


counts2, bin_edges2 = np.histogram(np.real(eigs_complex), bins=nbins2)

total_circ = np.sum(counts)
#print('total circ = ', total_circ)

#endregion

#region plotting setup 
plot_title = str('Eigenvalues of J: n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
plot_title2 = str('Max Eigenvalues vs. rowsum: n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
hist1_title = str('real component of eigs ON the real axis:'+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
hist2_title = str('real component of eigs OFF the real axis:'+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))

plot_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+'; '+str('%.0f'%(np.max(nzeros/n/runs*100)))+'% on x=0')
plot_text_2 = str('circle centered at real component ='+str(r_mean))
text_fitpars = str('J fit mean: '+str('%0.3f'%parsg[1])+', sigma^2: '+str('%0.3f'%parsg[2])+', A= '+str('%0.3f'%parsg[0]))
text_fitparsM = str("M' fit mean: "+str('%0.3f'%pars_mp[1])+', sigma^2: '+str('%0.3f'%pars_mp[2])+', A= '+str('%0.3f'%pars_mp[0]))

box_par = dict(boxstyle='square', facecolor='white')
text_vars = str("$\lambda_2'=$"+ str(m2-1) + '\n $\sigma_a =$'+ str(sigma2**0.5)+'\n s = '+str(n/2))
#text_fitpars_M = str('M fit mean: '+str('%0.3f'%parsgm[1])+', sigma^2: '+str('%0.3f'%parsgm[2])+', A= '+str('%0.3f'%parsgm[0]))


hist2_text = str('mean at '+ str('%0.3f'%r_mean)+ ', '+str(total_circ)+' counts')

max_ax = np.max(eigs_imag) + 3
min_ax = np.min(eigs_imag) - 3

fsize = (6,6)


min_jvals = np.min(maxeig_J)
max_jvals = np.max(maxeig_J)
yjvals = np.linspace(min_jvals-0.2, max_jvals+0.2, 5)


#endregion



#region plotting 


# fig 1: distribution of all eigenvalues 
plt.figure(figsize=fsize)
plt.grid()
plt.title(plot_title)
plt.xlabel('real component')
plt.ylabel('imaginary component')
plt.plot(eigs_real, eigs_imag, 'o', ms=1, label='all survive')
plt.plot(np.real(eigs_died), np.imag(eigs_died), 'o', ms=1, label='species died')
#plt.plot(np.real(eigs_Mp), np.imag(eigs_Mp), 'o', ms=2, alpha=.3, label = "M'")
#plt.plot(np.real(eigs_A), np.imag(eigs_A), 'o', ms=2, alpha=.3, label = 'A')
plt.figtext(0.13, 0.12, plot_text)
plt.legend()
#plt.figtext(0.13, 0.86, plot_text_2)
#plt.xlim([-15, 3])
#plt.ylim([-9, 9])
#plt.tight_layout()

#figure: compex max eig 
plt.figure(figsize=fsize)
plt.plot(np.real(maxeig_J_complex), np.imag(maxeig_J_complex), '.', label = 'all survive')
plt.plot(np.real(maxeig_J_complex_died), np.imag(maxeig_J_complex_died), '.', label ='died')
plt.grid()
plt.legend()
plt.title('max Eigenvalue of J')
plt.xlabel('real component')
plt.ylabel('imaginary component')

# fig 2: histogram of the eigenvalues on the real axis. im(eig)=0
plt.figure(figsize=fsize)
#constrained_layrueout = T
#plt.plot(mp_bin_centers[:bend_mp], mp_counts[:bend_mp], '.', label='true Mprime')
plt.hist(np.real(eigs_real_axis), bins=nbins1, range = histrange, alpha=0.8, label='J')
#plt.hist(np.real(eigs_real_axis_M), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'M')
plt.hist(np.real(eigs_real_axis_Mp), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'M prime')
plt.hist(np.real(eigs_real_axis_A), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'A')
plt.title(hist1_title)
plt.figtext(0.13, 0.66, plot_text)
plt.figtext(0.13, 0.63, text_fitpars)
plt.figtext(0.13, 0.60, text_fitparsM)
#
# plt.figtext(0.13, 0.82, text_fitpars_M)
#plt.xlabel('real component of eigenvalue')
plt.ylabel('counts')
#plt.plot(bin_centers, gaussian(bin_centers, *parsg), '-', label='fit of J')
plt.plot(fitmx, gaussian(fitmx, *pars_mp), '-', label='fit of M prime')
#plt.plot(bin_centersm, gaussian(bin_centersm, *parsgm), '-m')
#plt.plot(bin_centers, gaussian_box(bin_centers, *pars), '-m')
#plt.tight_layout()
plt.legend()
plt.grid()


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


#figure: max eig of A vs max eig ofMp
plt.figure(figsize=fsize)
plt.plot(maxeig_A, maxeig_Mp, '.')
plt.grid()
plt.xlabel(' Max (nonzero) eigenvalue of A')
plt.ylabel('Max eigenvalue of Mprime')
plt.figtext(0.15, 0.15, text_vars, bbox=box_par)



#figure: max rowsum of A vs max nonzro eigenvalue of J'
plt.figure(figsize=fsize)
#plt.plot(yjvals, yjvals, '--', label= 'x=y')
plt.plot(np.linspace(np.min(Ars_max), np.max(Ars_max), 3), 0*np.linspace(np.min(Ars_max), np.max(Ars_max), 3), '--k')
plt.plot(0*np.linspace(np.min(maxeig_Mp), np.max(maxeig_Mp), 3), np.linspace(np.min(maxeig_Mp), np.max(maxeig_Mp), 3), '--k')
plt.plot(Ars_max, maxeig_Mp, '.', color='C1', ms = 3, label = "M'")
plt.plot(Ars_max, maxeig_J, '.', color='C0', ms = 3, label = "J")
plt.grid()
plt.title(plot_title2)
plt.legend(loc='upper left')
plt.xlabel(' Max rowsum of A')
plt.ylabel("Max real eigenvalue of J / M'")

plt.figure(figsize=fsize)
plt.plot(n_survives, maxeig_J, '.')
plt.grid()
plt.xlabel('number of subspecies with nonzero final population')
plt.ylabel('Max eigenvalue of Jacobian')

plt.figure(figsize=fsize)
plt.plot(maxeig_J, final_match, '.')
plt.grid()
plt.ylabel('final population match xstar set ')
plt.xlabel('Max eigenvalue of Jacobian')







'''
plt.figure(figsize=fsize)
plt.hist(np.real(eigs_complex), bins=nbins2)#, range=histrange)
plt.title(hist2_title)
plt.figtext(0.13, 0.12, hist2_text)
#plt.xlabel('real component of eigenvalue')
plt.ylabel('counts')
#plt.plot(bin_centers2, circ(bin_centers2, *parsc), '-r')
#plt.grid()

plt.figure(figsize=fsize)
plt.title('Real axis eigenvalues not in the gaussian')
y_rect = counts-gaussian(bin_centers, *parsg)
plt.plot(bin_centers, y_rect, '-b')
#plt.grid()'''
'''plt.figure(figsize=fsize)
c_rs, b_rs = np.histogram(np.real(A_rowsums), bins=200)
#c_rs = c_rs/np.sum(c_rs)
b_crs_centers = np.real((b_rs[:-1] + b_rs[1:]) / 2)
p0_rs = [runs, -2, 2*(n/2-1)*sigma2]
pars_rs, cov_rs = curve_fit(gaussian, b_crs_centers, c_rs, p0_rs)

plt.plot(b_crs_centers, c_rs)
plt.plot(b_crs_centers, gaussian(b_crs_centers, *pars_rs), '-m')
plt.plot(b_crs_centers, gaussian(b_crs_centers, pars_rs[0], -2, 2*(n/2-1)*sigma2), '-r')
#print('predicted: ',p0_rs, ', actual: ',pars_rs)
#plt.grid()'''

if xstar == 0:  
    plt.figure(figsize=fsize)
    plt.plot(maxeig_J, n_survives, '.')
    plt.grid()




# plot of all the real components, regardless of being on the real axis 
'''plt.figure(figsize=fsize)
plt.hist(np.real(eigs_A), bins = nbins1, range = histrange, color='b', alpha = 0.4)
plt.hist(np.real(eigs_M), bins = nbins1, range = histrange, color='r', alpha = 0.4)
plt.hist(np.real(eigs), bins = nbins1, range = histrange, color='g', alpha = 0.4)
plt.grid()'''




'''plt.figure(figsize=fsize)
plt.plot(np.real(eigs_M), np.imag(eigs_M))
plt.grid()'''



#plt.tight_layout()




#plt.figure(figsize=fsize)
#plt.hist(eigs_imag, nbins1)





plt.show()
#endregion


    


    


