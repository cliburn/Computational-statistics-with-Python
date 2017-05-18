from numbapro import cuda, void, float32
import numpy as np

@cuda.jit('void(float32[:], float32[:], float32[:])')
def cu_add(a, b, c):
    # i = cuda.grid(1)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw

    if i  > c.size:
        return
    c[i] = a[i] + b[i]

if __name__ == '__main__':
    gpu = cuda.get_current_device()

    n = 100
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.empty_like(a)

    nthreads = gpu.WARP_SIZE
    nblocks = int(np.ceil(float(n)/nthreads))
    print 'Blocks per grid:', nblocks
    print 'Threads per block', nthreads

    cu_add[nblocks, nthreads](a, b, c)
    print c
