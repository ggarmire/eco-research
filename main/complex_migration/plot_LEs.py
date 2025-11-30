# goal here is to replicate https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007827. 
# see functions as well! 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys 
sys.path.insert(0, '/Users/gracegarmire/Documents/UIUC/eco_research/LV_chotic')

LEs = np.load('LE_s200M8A86ds1t450.npy')
print(LEs.shape)
nLE = LEs.shape[0]
nruns = LEs.shape[1]
index = np.arange(nLE) + 1
print(index)

LEmean = np.mean(LEs, axis = 1)
LEmean_cumsum = np.cumsum(LEmean)

for i in range(nLE):
    if LEmean_cumsum[i] < 0: 
        KYi = i-1
        break
KY_dim = KYi + LEmean_cumsum[KYi] / abs(LEmean_cumsum[KYi+1])
print('KY dim:', KY_dim)

# region plot setup things 
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
box_par = dict(boxstyle='square', facecolor = 'blue', alpha = 0.2)
plot_text = str('200 species interacting in 8 patches')
plot_text2 = str('KY Dimension = '+str('%.3f'%KY_dim))
# region plotting 
fig, ax = plt.subplots()
ax.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.5)
ax.fill_between(index[:7], np.zeros(7), LEmean[:7], color = 'red', alpha = 0.2)
ax.fill_between(index[7:30], np.zeros(23), LEmean[7:30], color = 'blue', alpha = 0.2)
for i in range(nruns):
    ax.plot(index, LEs[:,i], 'o', alpha = 0.2, ms = 3, color = 'green')
ax.plot(index, LEmean, 'o--', ms = 3, label = 'average LE', color = 'black')
#ax.grid()
legend_elements = [Line2D([0], [0], linestyle = 'none', marker = 'o', color='green', alpha = 0.2, ms = 3, label = 'by run'),
                   Line2D([0], [0], linestyle = '--', marker = 'o', color='black', ms = 3,  label = 'mean over runs')]
ax.legend(handles=legend_elements)
plt.xlabel('LE index')
plt.ylabel('Lyapunov exponent')
plt.title('Lyapunov Spectrum for fluctuating gLV system')
plt.figtext(0.2, 0.8, plot_text)
plt.figtext(0.4, 0.3, plot_text2, bbox=box_par)
#ax.set_yscale('log')


plt.show()

plt.savefig('LEspec_a86.s)