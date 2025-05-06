import numpy as np
import random
'''
runs = 10
the1 = 0
the0 = 0
bigger1 = 0
for i in range(runs):
    s = np.random.normal(0, 1, 10)
    one = np.var(s, ddof=1)
    zero = np.var(s, ddof=0)
    #print(one, zero)

    if abs(1-one) < abs(1-zero):
        the1 += 1
    else: 
        the0 += 1

    if one > 1:
        bigger1 += 1
    
    var_meanact = np.var(s, ddof = 1)
    var_mean0 = abs(np.sum(s)/(len(s)-1))

    print('mean 0: ', var_mean0, ', mean act: ', var_meanact, 'acctual mean: ', np.mean(s))

print('the1: ', the1)
print('the0: ', the0)
print(' was bigger than 1:', bigger1)
'''
print(list(range(10, 100, 4)) + list(range(102, 200, 30)))
