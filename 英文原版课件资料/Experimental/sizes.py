import numpy as np

a = 12345
b = range(1000000)
c = np.random.random((1000,1000))

def grow(x):
    return np.concatenate([x, x])

c2 = grow(c)