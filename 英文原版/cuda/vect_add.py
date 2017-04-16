from numbapro import cuda, vectorize, guvectorize
from numbapro import void, int64, float32, float64
import numpy as np

@vectorize(['int64(int64, int64)', 
            'float64(float32, float32)',
            'float64(float64, float64)'], 
           target='gpu')
def cu_add(a, b):
    return a + b

if __name__ == '__main__':
    n = 100
    A = np.arange(n)
    B = np.arange(n)
    C = cu_add(A, B)
    print C
    
