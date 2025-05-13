
#%% libraries 
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

#region variables to change
K_set = 0.2
muc = -0.5
mua = -0.5
f = 1.5
g = 1

z = 0.51
xstar = 1
n = 50

runs = 1000
#endregion


#region set up other variables

x0 = x0_vec(n)
C = 1
sigma2 = K_set**2/n*2
print('sigma:', sigma2**0.5)
t_end = 30     # length of time 
Nt = 1000
M_pre = M_matrix(n, muc, mua, f, g)
xs = np.ones(n)
for i in range(0, n, 2):
    xs[i] = z
#eigs_real = np.zeros((n, runs))
#eigs_imag = np.zeros((n, runs))
#eigs_real_max = np.zeros(runs)

eigs = []
eigs_real_max = []

eigs_M = []
eigs_Mp = []
eigs_A = []

A_rowsums = []
Ars_max = []


maxeig_J = []
maxeig_A = []
maxeig_M = []
maxeig_Mp = []


#endregion 


# region loop 
for run in range(runs):
    seed = run
    np.random.seed(seed)
    if run %778 == 0:
        print(run)

    # make matrices 
    A = A_matrix(n, C, sigma2, seed, LH=1) 
    if xstar == 1:
        A_rows = np.dot(A, xs)
        M_rows = np.dot(M_pre, xs)
        scales = -np.divide(np.multiply(A_rows, xs), M_rows)
        M = np.multiply(M_pre, np.outer(scales, np.ones(n)))
        A_rowsums.extend(A_rows)
        Ars_max.append(np.max(A_rows))

        Mp = M + np.diag(A_rows)
        Mpvals, Mpvecs = np.linalg.eig(Mp)

    Mvals, Mvecs = np.linalg.eig(M)
    

    eigs_M.extend(Mvals)
    eigs_Mp.extend(Mpvals)

    Avals, Avecs = np.linalg.eig(A)
    eigs_A.extend(Avals)
 
    Jac = LH_jacobian(n, A, M, xs) 
    Jvals, Jvecs = np.linalg.eig(Jac)   

    #eigs_real[:, run] = np.real(Jvals)
    #eigs_imag[:, run] = np.imag(Jvals)
    #eigs_real_max[run] = np.max(np.real(Jvals))

    eigs.extend(Jvals)

    Avalsm = np.ma.masked_inside(Avals, -1e-10, 1e-10)
    Mpvalsm = np.ma.masked_inside(Mpvals, -1e-10, 1e-10)

    maxeig_J.append(np.max(np.real(Jvals)))
    maxeig_A.append(np.max(np.real(Avalsm)))
    maxeig_Mp.append(np.max(np.real(Mpvalsm)))

    
#endregion

# region post analysis 
eigs_real = np.real(eigs)
eigs_imag = np.imag(eigs)

eigs_real_axis = []     # eigenvalues on the real line
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

def circ(x, r, h, c):
    return h * np.sqrt(abs(r**2 - (x-c)**2))


    
#p02 = [runs/4, -6.25, .5, r_mean, runs/7, 1]
p01g = [1250, -80, 16]

p01gm = [1700, -4.25, 1]
    

counts, bin_edges = np.histogram(np.real(eigs_real_axis), bins=nbins1, range = histrange)
bin_centers = np.real((bin_edges[:-1] + bin_edges[1:]) / 2)
print('bin width = ', bin_edges[1]-bin_edges[0])

# fit the gaussian part with a gauss:
'''circ1 = -2 - 2*K_set - 0.5
circ2 = -2 + 2*K_set + 0.5
b1 = 0 
b2 = 0
for i in range(len(bin_centers)):
    if bin_centers[i] < circ1: 
        b1 = i
    elif bin_centers[i] < circ2:
        b2 = i'''

b1 = np.digitize(-2 - 2.5*K_set, bin_edges)
b2 = np.digitize(-2 + 2.5*K_set, bin_edges)

#print(b1, b2)

r1 = bin_centers[:b1]
c1 = counts[:b1]
r2 = bin_centers[b2:]
c2 = counts[b2:]
fitx = []; fity = []
fitx.extend(r1)
fitx.extend(r2)
fity.extend(c1); fity.extend(c2)

parsg, covg = curve_fit(gaussian, fitx, fity, p01g, maxfev=5000)
print('K: ', K_set, ', pars of gauss of J: ', parsg)

m1 = 1              
m2 = (mua*muc - f*g)/((mua+g)*(muc+f))      # these are the eigenvlaues of the 

print('eigs of M:', m1, m2)
print('mean diff: ', parsg[1] - m2, '; mean ratio:', parsg[1]/m2)


#countsm, bin_edgesm = np.histogram(np.real(eigs_real_axis_M), bins=nbins1, range = histrange)
#bin_centersm = np.real((bin_edgesm[:-1] + bin_edgesm[1:]) / 2)

#parsgm, covgm = curve_fit(gaussian, bin_centersm, countsm, p01gm)

counts2, bin_edges2 = np.histogram(np.real(eigs_complex), bins=nbins2)

total_circ = np.sum(counts)
#print('total circ = ', total_circ)

'''# fit the circle hist

counts2, bin_edges2 = np.histogram(np.real(eigs_complex), bins=nbins2)
bin_centers2 = np.real((bin_edges2[:-1] + bin_edges2[1:]) / 2)

p02 = [2*K_set, 0.7*runs, -2*K_set-1]

parsc, covc = curve_fit(circ, bin_centers2, counts2, p02)'''

#endregion

# region fit box(doesnt work ): 
'''y_rect = counts-gaussian(bin_centers, *parsg)
yrect_mean = np.mean(y_rect)
yrect_sigma = np.std(y_rect)

r1 = 0
r2 = 0
r1stop = 0

for i in range (len(bin_centers)):
    if y_rect[i] < 3*yrect_mean and r1stop == 0:
        r1 = bin_centers[i]
    if y_rect[i] > 3*yrect_mean:
        r2 = bin_centers[i]
        r1stop = 1

print('rs: ', r1, r2)

p01 = [*parsg, np.max(y_rect), (r1+r2)/2, r2-r1]
pars, cov = curve_fit(gaussian_box, bin_centers, counts, p01)


print('fit parameters:', pars)
x_fit = np.linspace(min(bin_edges), max(bin_edges), 1000)'''

#endregion

#region plotting setup 
plot_title = str('Eigenvalues of J: n='+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
hist1_title = str('real component of eigs ON the real axis:'+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))
hist2_title = str('real component of eigs OFF the real axis:'+str(int(n/2))+'*2, z='+str(z)+', K='+str(K_set))

plot_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+'; '+str('%.0f'%(np.max(nzeros/n/runs*100)))+'% on x=0')
plot_text_2 = str('circle centered at real component ='+str(r_mean))
text_fitpars = str('J fit mean: '+str('%0.3f'%parsg[1])+', sigma^2: '+str('%0.3f'%parsg[2])+', A= '+str('%0.3f'%parsg[0]))

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
plt.plot(eigs_real, eigs_imag, 'o', ms=2, alpha=.5)
plt.figtext(0.13, 0.12, plot_text)
#plt.figtext(0.13, 0.86, plot_text_2)
#plt.xlim([-15, 3])
#plt.ylim([-9, 9])
#plt.tight_layout()


# fig 2: histogram of the eigenvalues on the real axis. im(eig)=0
plt.figure(figsize=fsize)
#constrained_layrueout = T

plt.hist(np.real(eigs_real_axis), bins=nbins1, range = histrange, label='J')
#plt.hist(np.real(eigs_real_axis_M), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'M')
plt.hist(np.real(eigs_real_axis_Mp), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'M prime')
plt.hist(np.real(eigs_real_axis_A), bins=nbins1, range = histrange, histtype='step', alpha = 1, label = 'A')
plt.title(hist1_title)
plt.figtext(0.13, 0.69, plot_text)
plt.figtext(0.13, 0.66, text_fitpars)
#
# plt.figtext(0.13, 0.82, text_fitpars_M)
#plt.xlabel('real component of eigenvalue')
plt.ylabel('counts')
plt.plot(bin_centers, gaussian(bin_centers, *parsg), '-r')
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



#figure: max rowsum of A vs max nonzro eigenvalue of M'
plt.figure(figsize=fsize)
#plt.plot(yjvals, yjvals, '--', label= 'x=y')
plt.plot(Ars_max, maxeig_Mp, '.', label = "M'")
plt.grid()
plt.legend(loc='upper left')
plt.xlabel(' Max rowsum of A')
plt.ylabel('Max eigenvalue of M')



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

plt.figure(figsize=fsize)
c_rs, b_rs = np.histogram(np.real(A_rowsums), bins=200)
#c_rs = c_rs/np.sum(c_rs)
b_crs_centers = np.real((b_rs[:-1] + b_rs[1:]) / 2)
p0_rs = [runs, -2, 2*(n/2-1)*sigma2]
pars_rs, cov_rs = curve_fit(gaussian, b_crs_centers, c_rs, p0_rs)

plt.plot(b_crs_centers, c_rs)
plt.plot(b_crs_centers, gaussian(b_crs_centers, *pars_rs), '-m')
plt.plot(b_crs_centers, gaussian(b_crs_centers, pars_rs[0], -2, 2*(n/2-1)*sigma2), '-r')
print('predicted: ',p0_rs, ', actual: ',pars_rs)
#plt.grid()



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


    


    


