
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import LH_jacobian_norowsum
from lv_functions import x0_vec
import random 
import math

seed = random.randint(0, 1000)
#seed = 680
print("seed: ", seed)


#%% initial conditions and such 
n = 20     # number of species 
s = n / 2
x0 = x0_vec(n)
#x0 = np.ones(n)
#print('x0: ', x0)
t = np.linspace(0, 50, 2000)
#t = np.linspace(0, 500000, 20000)

sigma2 = 0.0
C = 1
K_set = (s *C * sigma2)**0.5


A = A_matrix(n, C, sigma2, seed, LH=1)
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
#print(A_classic)

K = (C*sigma2*n/2)**0.5
print("complexity (classic): ", K)
#print("A:"); print(A)
Avals, Avecs = np.linalg.eig(A)
#print('real eigs:',  np.real(Avals))

A_rowsums = np.dot(A, np.ones(n))
print('max A rowsums:', np.max(A_rowsums))

# for m matrix:
muc = -0.5
mua = -0.5
f = 1.5
g = 1


xstar = 1
alpha = 0.7

M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)

#print('max real eig of M:', np.max(np.real(mvals)))
#print('M:'); print(M)""
#M_rowsums = np.dot(M, np.ones(n))
if xstar == 1:
    xs = np.ones(n)
    for i in range(0, n, 2):
        xs[i] = alpha
    #print(xs)
    A_rows = np.dot(A, xs)
    M_rows = np.dot(M, xs)
    scales = -np.divide(np.multiply(A_rows, xs), M_rows)
    M = np.multiply(M, np.outer(scales, np.ones(n)))

    Mprime = M + np.diag(A_rows)
    mpvals, mpvecs = np.linalg.eig(Mprime)

print('M after scaling :'); print(M[0:4, 0:4])
print('mpvals:', mpvals)

# run function here: 
result = lv_LH(x0, t, A, M)

#print('eigenvalues of M:', mvals)
#print('eigenvalues of A:', Avals)
#print('eigenvectors of M:', mvecs)
#print('eigenvectors of A:', Avecs)

#print('DIVIDED:', np.divide(mvecs, Avecs))



#result = integrate.odeint(derivative, x0, t, args = (M, A))



xf = result[-1, :]
#xf = np.zeros(n)



#print("final populations: ")
#print(xf)

#%% Stats: 

species_left = 0
species_stable = 0
z = np.zeros(int(n/2))
for i in range(n):
    if result[-1, i] > 1e-3:
        species_left+=1
        if abs((result[-1, i]-result[-2, i]) / result[-1, i]) < 0.001:
            species_stable +=1
    if i%2 == 0:
        z[int(i/2)]= (result[-1,i]/(result[-1,i]+result[-1,i+1]))

#print("juvinile fractions: ", z)

print("tfinal: ", t[-1], ", species remaining:", species_left, "sepcies stable: ", species_stable)

#%% Calculate the Jacobian
if xstar ==1:
    Jac = LH_jacobian(n, A, M, xs) 
elif xstar ==0:
    Jac = LH_jacobian_norowsum(result[-1, :], A, M)
#print("Jacobian: ", Jac)
Jvals, Jvecs = np.linalg.eig(Jac)
#Jvals2, Jvecs2 = np.linalg.eig(Jac2)
#print('Eigenvalues of Jacobian: \n', Jvals)
#print('Max real eigenvalue of A:', np.max(np.real(Avals)))
#print('Max real eigenvalue of Jac:', np.max(np.real(Jvals)))
#print('Max real eigenvalue of Jac2:', np.max(np.real(Jvals2)))




#%% Plotting: 

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

plot_text = str('$\mu_c =$'+str(muc)+', $\mu_a =$'+str(mua)+', $f =$'+str(f)+', $g =$'+str(g)+', A seed ='+str(seed)+ ', K='+str('%.3f'%K))
if xstar == 1:
    plot_text2 = str('Max real eigenvalue of J: '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eigenvalue of Mprime: '+ str('%.3f'%(np.max(np.real(mpvals))))
                    +'\n Max real eigenvalue of A: '+ str('%.3f'%(np.max(np.real(Avals)))))
elif xstar == 0:
    plot_text2 = str('Max real eigenvalue of J: '+ str('%.3f'%(np.max(np.real(Jvals)))) + 
                    '\n Max real eigenvalue of M (unscaled): '+ str('%.3f'%(np.max(np.real(mvals))))
                    +'\n Max real eigenvalue of A: '+ str('%.3f'%(np.max(np.real(Avals)))))

box_par = dict(boxstyle='square', facecolor='white', alpha = 0.5)



plt.figure()

plt.grid()
if xstar == 1:
    title = str('Species Population over time, N=2S='+str(n)+', x*=1, z = '+str(alpha))
elif xstar ==0: 
    title = str('Species Population over time, f='+str(f)+', x*/=1')
plt.title(title)
#plt.title("Species Population over time, f=0.49, x*=1")
for i in range(n):
    if i%2 == 0:
        plt.plot(0, result[0, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', mfc = 'none', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))       # child (empty)
    else:
        plt.plot(0, result[0, i], 'o', color=colors[math.floor(i/2)], ms = 3)
        plt.plot(t, result[:, i], 'o', color=colors[math.floor(i/2)], ms = 3, markevery = (i, 20))     # adult (full)
plt.xlabel('Time t')
plt.ylabel('Population density')
plt.figtext(0.13, 0.12, plot_text)
plt.figtext(0.4, 0.6, plot_text2, bbox=box_par)
#plt.figtext(0.5, 0.80, plot_text3)
#plt.figtext(0.5, 0.76, plot_text4)
plt.semilogx

legend_elements = [Line2D([0], [0], marker = 'o', color='C0', mfc = 'none', label='child'),
                   Line2D([0], [0], marker = 'o', color='C0', label='adult')]
plt.legend(handles=legend_elements)

#plt.ylim(-0.1, 6)

#plt.ylim(min(0, np.min(result)-0.1), 1.1*np.max(result))


plt.show()