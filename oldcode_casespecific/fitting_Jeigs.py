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
import pandas as pd 

#region import data 
fdata = 'J_fit_data_f.csv'
gdata = 'J_fit_data_g.csv'
muadata = 'J_fit_data_mua.csv'
mucdata = 'J_fit_data_muc.csv'
Kdata = 'J_fit_data_K.csv'
ndata = 'J_fit_data_n.csv'

dff = pd.read_csv(fdata)
dfg = pd.read_csv(gdata)
dfmua = pd.read_csv(muadata)
dfmuc = pd.read_csv(mucdata)
dfK = pd.read_csv(Kdata)
dfn = pd.read_csv(ndata)

#region define functions for mean 
def powerlaw_f(x, a, b, c, d):
    return a * (x+d)**b + c 

def powerlaw_muc(x, a, b, c, d):
    return a * (x+d)**b + c 

def powerlaw_muc_nod(x, a, b, c):
    return a * (x)**b + c 

def powerlaw_mua(x, a, b, c, d):
    return a * (x+d)**b + c 

def powerlaw_n(x, a, b, c, d):
    return a * (x+d)**b + c 

def linear_n(x, m, b):
    return m*x + b





#region fit datasets means 
p0mf = [-8, -1, -6, -0.5]
p0mg = [-8, -1, -6, -0.5]
p0mmuc = [-1.3, -1.5, -6.5, 1.5]
p0mmua = [-1.5, -1.1, -3.4, 1]

p0sf = [2.2, -1, 5, -0.5]
p0sg = [2.2, -1, 5, -0.5]
p0smuc = [1.5, -1.7, 2.4, 1.5]
p0smua = [1.5, -1.7, 2.4, 1]

p0sn = [0.15, 0]

xf = dff['f'].values; mf = dff['mean'].values; sf = dff['sigma2'].values
parf, covf = curve_fit(powerlaw_f, xf, mf, p0mf)
parsf, covsf = curve_fit(powerlaw_f, xf, sf, p0sf)

xg = dfg['g'].values; mg = dfg['mean'].values; sg = dfg['sigma2'].values
parg, covg = curve_fit(powerlaw_f, xg, mg, p0mg)
parsg, covsg = curve_fit(powerlaw_f, xg, sg, p0sg)

xmuc = dfmuc['muc'].values; mmuc = dfmuc['mean'].values; smuc = dfmuc['sigma2'].values
parmuc, covmuc = curve_fit(powerlaw_muc, xmuc, mmuc, p0mmuc)
parsmuc, covsmuc = curve_fit(powerlaw_muc, xmuc, smuc, p0smuc)

xmua = dfmua['mua'].values; mmua = dfmua['mean'].values; smua = dfmua['sigma2'].values
parmua, covmua = curve_fit(powerlaw_mua, xmua, mmua, p0mmua)
parsmua, covsmua = curve_fit(powerlaw_mua, xmua, smua, p0smua)

xn = dfn['n'].values; mn = dfn['mean'].values; mnerr = dfn['meanerr'].values; sn = dfn['sigma2'].values; snerr = dfn['sig2err'].values
parsn, covsn = curve_fit(linear_n, xn, sn, p0sn, sigma=snerr)
mnmean = np.mean(mn)

xk = dfK['K'].values; sigsetk = dfK['sigma2_set'].values; mk = dfK['mean'].values; mkerr = dfK['meanerr'].values; sK = dfK['sigma2'].values
mkmean = np.mean(mk)



means_J_f = []
means_M_f = []
means_J_g = []
means_M_g = []
means_J_muc = []
means_M_muc = []
means_J_mua = []
means_M_mua = []

s_M_k = []
s_J_k = []

s_M_n = []
s_J_n = []

s_M_f = []
s_M_g = []
s_M_muc = []
s_M_mua = []
s_J_f = []
s_J_g = []
s_J_muc = []
s_J_mua = []



for i in range(len(xf)): 
    l = ((0.25-xf[i]) / (.5*(xf[i]-0.5)))
    means_M_f.append(2*l -2)
    s_M_f.append(4*24*.0064*(l-1)**2)
    means_J_f.append(mf[i])
    s_J_f.append(sf[i])
for i in range(len(xg)): 
    l = (0.25-1.5*xg[i]) / ((xg[i]-0.5))
    means_M_g.append(2*l - 2)
    s_M_g.append(4*24*.0064*(l-1)**2)
    means_J_g.append(mg[i])
    s_J_g.append(sg[i])
for i in range(len(xmuc)): 
    l = (-0.5*xmuc[i]-1.5) / ((1.5+xmuc[i])*0.5)
    means_M_muc.append( 2*l - 2)
    s_M_muc.append(4*24*.0064*(l-1)**2)
    means_J_muc.append(mmuc[i])
    s_J_muc.append(smuc[i])
for i in range(len(xmua)): 
    l = (-0.5*xmua[i]-1.5) / (1+xmua[i])
    means_M_mua.append( 2*l - 2)
    s_M_mua.append(4*24*.0064*(l-1)**2)
    means_J_mua.append(mmua[i])
    s_J_mua.append(smua[i])


for i in range(len(xk)):
    l = (0.25-1.5) / (0.5)
    s_M_k.append(4*24*(l-1)**2 * sigsetk[i])
    s_J_k.append(sK[i])

for i in range(len(xn)):
    l = (0.25-1.5) / (0.5)
    s_M_n.append(2*(xn[i]-2)*.0064*(l-1)**2)
    s_J_n.append(sn[i])






means_M = []
means_J = []

means_M.extend(means_M_f); means_M.extend(means_M_g); means_M.extend(means_M_muc); means_M.extend(means_M_mua)
means_J.extend(means_J_f); means_J.extend(means_J_g); means_J.extend(means_J_muc); means_J.extend(means_J_mua)

s_M = []
s_J = []
s_M.extend(s_M_f); s_M.extend(s_M_g); s_M.extend(s_M_muc); s_M.extend(s_M_mua)
s_J.extend(s_J_f); s_J.extend(s_J_g); s_J.extend(s_J_muc); s_J.extend(s_J_mua)

parsmj, covsmj = curve_fit(linear_n, means_M, means_J)
print('parsmj', parsmj)

parssj, covssj = curve_fit(linear_n, s_M, s_J)
print('parssj', parssj)

parssk, covssk = curve_fit(linear_n, s_M_k, s_J_k)
print('parssk', parssk)

parssn, covssn = curve_fit(linear_n, s_M_n, s_J_n)
print('parssn', parssn)


#print(snerr)





#region plot 

t1 = str('muc = -0.5, mua = -0.5, f = 1.5, g = 1, sigma2 = 0.0064, n = 50')

t_ns = str('y='+str('%0.3f'%parsn[0])+'n+'+str('%0.3f'%parsn[1]))
t_fm = str('f: y='+str('%0.3f'%parf[0])+' * (f+'+str('%0.3f'%parf[3])+')^'+str('%0.3f'%parf[1])+'+'+str('%0.3f'%parf[2]))
t_gm = str('g: y='+str('%0.3f'%parg[0])+' * (g+'+str('%0.3f'%parg[3])+')^'+str('%0.3f'%parg[1])+'+'+str('%0.3f'%parg[2]))
t_mucm = str('muc: y='+str('%0.3f'%parmuc[0])+' * (muc+'+str('%0.3f'%parmuc[3])+')^'+str('%0.3f'%parmuc[1])+'+'+str('%0.3f'%parmuc[2]))
t_muam = str('mua: y='+str('%0.3f'%parmua[0])+' * (mua+'+str('%0.3f'%parmua[3])+')^'+str('%0.3f'%parmua[1])+'+'+str('%0.3f'%parmua[2]))

t_fs = str('f: y='+str('%0.3f'%parsf[0])+' * (f+'+str('%0.3f'%parsf[3])+')^'+str('%0.3f'%parsf[1])+'+'+str('%0.3f'%parsf[2]))
t_gs = str('g: y='+str('%0.3f'%parsg[0])+' * (g+'+str('%0.3f'%parsg[3])+')^'+str('%0.3f'%parsg[1])+'+'+str('%0.3f'%parsg[2]))
t_mucs = str('muc: y='+str('%0.3f'%parsmuc[0])+' * (muc+'+str('%0.3f'%parsmuc[3])+')^'+str('%0.3f'%parsmuc[1])+'+'+str('%0.3f'%parsmuc[2]))
t_muas = str('mua: y='+str('%0.3f'%parsmua[0])+' * (mua+'+str('%0.3f'%parsmua[3])+')^'+str('%0.3f'%parsmua[1])+'+'+str('%0.3f'%parsmua[2]))

t_mj = str('y='+str('%0.3f'%parsmj[0])+'n+'+str('%0.3f'%parsmj[1]))
t_mjs = str('y='+str('%0.3f'%parssj[0])+'n+'+str('%0.3f'%parssj[1]))
t_mjk = str('y='+str('%0.3f'%parssk[0])+'n+'+str('%0.3f'%parssk[1]))
t_mjn = str('y='+str('%0.3f'%parssn[0])+'n+'+str('%0.3f'%parssn[1]))

plt.figure()
plt.plot(xf, mf,'.', label = 'f')
plt.plot(xf, powerlaw_f(xf, *parf), label = 'f fit')
plt.plot(xg, mg,'.', label = 'g')
plt.plot(xg, powerlaw_f(xg, *parg), label = 'g fit')
plt.legend()
plt.grid()
plt.xlabel('f, g')
plt.ylabel('mean of gaussian fit')
plt.figtext(0.13, 0.12, t1)
plt.figtext(0.4, 0.4, t_fm)
plt.figtext(0.4, 0.36, t_gm)


plt.figure()
plt.plot(xmua, mmua,'.', label = 'mua')
plt.plot(xmua, powerlaw_mua(xmua, *parmua), label = 'mua fit')
plt.plot(xmuc, mmuc, '.', label = 'muc')
plt.plot(xmuc, powerlaw_muc(xmuc, *parmuc), label = 'muc fit')
plt.legend()
plt.grid()
plt.xlabel('mua, muc')
plt.ylabel('mean of gaussian fit')
plt.figtext(0.13, 0.12, t1)
plt.figtext(0.4, 0.4, t_mucm)
plt.figtext(0.4, 0.36, t_muam)

plt.figure()
plt.errorbar(xn, mn, yerr=dfn['meanerr'], fmt='.', label = 'n')
plt.plot(xn, mnmean*np.ones(len(xn)), '--', label = 'mean of means')
plt.grid()
plt.xlabel('n')
plt.ylabel('mean of gaussian fit')
plt.legend()
plt.figtext(0.13, 0.12, t1)


plt.figure()
plt.errorbar(xk, mk, yerr=dfK['meanerr'], fmt='.', label = 'K')
plt.plot(xk, mkmean*np.ones(len(xk)), '--', label = 'mean of means')
plt.grid()
plt.xlabel('K')
plt.ylabel('mean of gaussian fit')
plt.legend()
plt.figtext(0.13, 0.12, t1)



plt.figure()
plt.plot(xf, sf,'.', label = 'f')
plt.plot(xf, powerlaw_f(xf, *parsf), label = 'f fit')
plt.plot(xg, sg,'.', label = 'g')
plt.plot(xg, powerlaw_f(xg, *parsg), label = 'g fit')
plt.legend()
plt.grid()
plt.xlabel('f, g')
plt.ylabel('sigma^2 of gaussian fit')
plt.figtext(0.13, 0.12, t1)
plt.figtext(0.4, 0.4, t_fs)
plt.figtext(0.4, 0.36, t_gs)



plt.figure()
plt.plot(xmua, smua,'.', label = 'mua')
plt.plot(xmua, powerlaw_mua(xmua, *parsmua), label = 'mua fit')
plt.plot(xmuc, smuc,'.', label = 'muc')
plt.plot(xmuc, powerlaw_muc(xmuc, *parsmuc), label = 'muc fit')
plt.legend()
plt.grid()
plt.xlabel('mua, muc')
plt.ylabel('sigma^2 of gaussian fit')
plt.figtext(0.13, 0.12, t1)
plt.figtext(0.4, 0.4, t_mucs)
plt.figtext(0.4, 0.36, t_muas)

plt.figure()
plt.errorbar(xn, sn, yerr=dfn['sig2err'], xerr=None, fmt = '.', label = 'n')
plt.plot(xn, linear_n(xn, *parsn), label = 'n fit')
plt.grid()
plt.xlabel('n')
plt.ylabel('sigma^2 of gaussian fit')
plt.figtext(0.13, 0.12, t1)
plt.figtext(0.6, 0.4, t_ns)
plt.legend()

plt.figure()
plt.plot(means_M_f, means_J_f, '.', label = 'varied f')
plt.plot(means_M_g, means_J_g, '.', label = 'varied g')
plt.plot(means_M_muc, means_J_muc, '.', label = 'varied muc')
plt.plot(means_M_mua, means_J_mua, '.', label = 'varied mua')
plt.plot(np.linspace(np.min(means_M), np.max(means_M), 100), linear_n(np.linspace(np.min(means_M), np.max(means_M), 100), *parsmj), label='fit')
plt.xlabel('eigenvalue of M')
plt.ylabel('mean of fit of J eigenvalues')
plt.figtext(0.6, 0.4, t_mj)
plt.grid()
plt.legend()

plt.figure()
plt.plot(s_M_f, s_J_f, '.', label = 'varied f')
plt.plot(s_M_g, s_J_g, '.', label = 'varied g')
plt.plot(s_M_muc, s_J_muc, '.', label = 'varied muc')
plt.plot(s_M_mua, s_J_mua, '.', label = 'varied mua')
plt.plot(np.linspace(np.min(s_M), np.max(s_M), 100), linear_n(np.linspace(np.min(s_M), np.max(s_M), 100), *parssj), label='fit')
plt.xlabel('sigma^2 of M ')
plt.ylabel('sigma^2 of J')
plt.figtext(0.6, 0.4, t_mjs)
plt.grid()
plt.legend()

plt.figure()
plt.plot(s_M_f, s_J_f, '.', label = 'varied f')
plt.plot(s_M_g, s_J_g, '.', label = 'varied g')
plt.plot(s_M_muc, s_J_muc, '.', label = 'varied muc')
plt.plot(s_M_mua, s_J_mua, '.', label = 'varied mua')
plt.plot(s_M_k, s_J_k, '.', label = 'varied sigma2 only')
plt.plot(np.linspace(np.min(s_M_k), np.max(s_M_k), 100), 
         linear_n(np.linspace(np.min(s_M_k), np.max(s_M_k), 100), *parssk), label='fit')
plt.xlabel('sigma^2 of M ')
plt.ylabel('sigma^2 of J')
plt.figtext(0.6, 0.4, t_mjk)
plt.grid()
plt.legend()

plt.figure()
plt.plot(s_M_n, s_J_n, '.', label = 'varied n, const. sigma2')
plt.plot(s_M_k, s_J_k, '.', label = 'varied sigma2 only')
plt.plot(np.linspace(np.min(s_M_n), np.max(s_M_n), 100), 
         linear_n(np.linspace(np.min(s_M_n), np.max(s_M_n), 100), *parssn), label='fit')
plt.xlabel('sigma^2 of M ')
plt.ylabel('sigma^2 of J')
plt.figtext(0.6, 0.4, t_mjn)
plt.grid()
plt.legend()



print('f mean a, b, c: ', parf)
print('g mean a, b, c: ', parg)
print('mua mean a, b, c: ', parmua)
print('muc mean a, b, c: ', parmuc)


print('f sigma a, b, c: ', parsf)
print('g sigma a, b, c: ', parsg)
print('mua sigma a, b, c: ', parsmua)
#print('muc sigma a, b, c: ', parsmuc)
print('n sigma a, b, c, d: ', parsn)




plt.show()
