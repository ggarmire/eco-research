
#%% libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from scipy import integrate
from lv_functions import A_matrix
from lv_functions import A_matrix_juvscale
from lv_functions import M_matrix
from lv_functions import lv_LH
from lv_functions import LH_jacobian
from lv_functions import x0_vec
import random 
import math

seed = random.randint(0, 1000)
#seed = 944
print('\n')
print("seed: ", seed)

random.seed(1)

#region initial conditions 

# values to set 
n = 20     # number of species 
s = int(n / 2)
x0 = x0_vec(n, 1)

t = np.linspace(0, 200, 2000)
K_set = 1
C = 1

muc = -1.5
mua = -0.5
f = 1.5
g = 1.2

# values that dont get set 
sigma2 = 2 * K_set**2 / n / C
print('sigma2:', sigma2)
K = (sigma2*C*s)**0.5

z = (muc-mua+((muc-mua)**2 +4*g*f)**0.5)/(2*g)
R_c = (z*muc+f)/z; R_a = z*g+mua
print('z =','%.3f'%z, 'R child =', '%.3f'%R_c, ', R adult =', '%.3f'%R_a)

zest = 1e-5         # anything less is zero

Rvec = R_a * np.ones(s)   # M part of equilibrium equation

# for m matrix:
M = M_matrix(n, muc, mua, f, g)
mvals, mvecs = np.linalg.eig(M)

# checks:
if n % 2 != 0:
    raise Exception("n is not a multiple of 2.")
if K!=K_set:
  raise Exception("K set does not match K.")
if abs(R_c-R_a) > 1e-10:
    raise Exception("error calculating R values.")
# endregion set variables 

# region classical case
A_classic = A_matrix(int(n/2), C, sigma2, seed, LH=0)
Avals_cl, Avecs_cl = np.linalg.eig(A_classic)
print('max classic eig:', np.max(np.real(Avals_cl)))
# endregion



#region loop
nj = 201
js = np.linspace(0, 1, nj)

Jvals_real = np.empty((0,n))
Jvals_imag = np.empty((0,n))


for j in js: 

    A = A_matrix_juvscale(n, C, sigma2, seed, j)

    # region Analytical Final Abundances'
    D = (j*z+1)*np.ones((s,s)) + (z-j*z)*np.identity(s)
    #print('D matrix:\n', D )
    Aprime = np.multiply(D, A_classic)
    Ap_inv = np.linalg.inv(Aprime)
    xf_an_adult = -np.dot(Ap_inv, Rvec)
    xf_an = np.repeat(xf_an_adult, 2)   # make unscaled
    xf_an[::2] *= z     # scale child 

    Jac = LH_jacobian(A, M, xf_an)
    Jvals, Jvecs = np.linalg.eig(Jac)

    Jvals_real = np.vstack((Jvals_real, np.real(Jvals)))
    Jvals_imag = np.vstack((Jvals_imag, np.imag(Jvals)))

    # endregion
#endregion loop

#region plot setup 

mpar_text = str('$\u03bc_c =$'+str(muc)+', $\u03bc_a =$'+str(mua)+', $f=$'+str(f)+', $g =$'+str(g)+' (z='+str('%.2f'%z)+')')
apar_text = str('n='+ str(n)+', A seed ='+ str(seed)+', K='+str(K_set))


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='$Re(\lambda_J)>0$', markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='$Re(\lambda_J)\leq 0$', markerfacecolor='blue', markersize=8)
]
xlimits = (np.min(Jvals_real)-0.1, np.max(Jvals_real)+0.1)
ylimits = (np.min(Jvals_imag)-0.02, np.max(Jvals_imag)+0.02)
# endregion


#region animate! 
j0 = 0

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom = 0.25)

# Set axis limits (customize as needed)
ax.set_xlim(xlimits)
ax.set_ylim(ylimits)
ax.set_title(f"j = {js[j0]:.3f}")
ax.grid(True)

# Initial scatter plot
x0 = Jvals_real[0]
y0 = Jvals_imag[0]
colors = np.where(x0 > 0, 'red', 'blue')
sc = ax.scatter(x0, y0, c=colors, s = 30)

# Add a slider for time step
ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
j_slider = Slider(ax_slider, 'j value', js[0], js[-1], valinit=j0, valstep=js[1] - js[0])

# Play button
ax_button = plt.axes([0.45, 0.1, 0.1, 0.1])
play_button = Button(ax_button, '▶ Play')
is_playing = [False]  # Use mutable object so it can be changed in closure

# Direction flag
direction = [1]  # 1 = forward, -1 = backward

# Update function
def update(val):
    j_val = j_slider.val
    # Find index closest to current j value
    t = np.searchsorted(js, j_val)
    if t >= nj:
        t = nj - 1
    x = Jvals_real[t]
    y = Jvals_imag[t]
    colors = np.where(x > 0, 'red', 'blue')
    sc.set_offsets(np.c_[Jvals_real[t], Jvals_imag[t]])
    sc.set_color(colors)
    ax.set_title(f"j = {js[t]:.3f}")
    fig.canvas.draw_idle()

# Connect slider to update function
j_slider.on_changed(update)

frame_index = [0]
def update_frame(_):
    if not is_playing[0]:
        return

    frame_index[0] += direction[0]
    
    if frame_index[0] >= nj:
        frame_index[0] = nj - 2
        direction[0] = -1
    elif frame_index[0] < 0:
        frame_index[0] = 1
        direction[0] = 1

    j_slider.set_val(js[frame_index[0]])

# Button toggle
def toggle_play(event):
    is_playing[0] = not is_playing[0]
    play_button.label.set_text('⏸ Pause' if is_playing[0] else '▶ Play')

play_button.on_clicked(toggle_play)

ani = FuncAnimation(fig, update_frame, interval=100)




#endregion

plt.show()












print('\n')