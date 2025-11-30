import numpy as np
import scipy 



LEs = np.load('LE_s200M8A86ds1t450.npy')
LE30 = LEs[:30, :]
print(LE30.shape)
