calc_LEs.py can be run to generate lyapunov spectrum for many sets of initial conditions, then these are plotted with plot_LEs.py, where the KY dimension is also found. 

Functions used for the Lotka Volterra system are in funcs/lv_patches_functions.py. Functions for calculating the Lyapunov exponents are in funcs/gglyapunov.py. make_init_conditions is used by calc_LEs.py to generate initial conditions after extinctions. 

Data used to generate plot is in LE_s200M8A86ds1t450.npy. 