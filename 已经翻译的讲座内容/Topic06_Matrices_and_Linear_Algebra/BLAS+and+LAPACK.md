

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
%precision 4
import os, sys, glob
import scipy.linalg as la
import scipy.linalg.blas as blas
import scipy.linalg.lapack as lapack
```

## 特别提醒

BLAS 和 LAPACK 的各种方法都各自在`scipy.linalg.blas` 和 `scipy.linalg.lapack` 这两个模块内。通常情况下，一般很少在写 Python 代码的时候直接用这两个，除非是对性能有特殊要求，此外在`scipy.linalg`中有对这些低层次方法的封装，用起来也更安全更简便。但如果你要用 C、Fortran 或者 CUDA 来进行开发，那就可能经常要直接使用这些线性代数方法了，Python 的模块提供了一些很友好的接口，可以借此来熟悉一下这些方法。


## 基础线性代数子方法（基础方法）（Basic Linear Algebra Subroutines (low level routines)）

BLAS 里的方法命名简短（terse），但有一个标准格式：

第一个字母表示的是精度（precision） - 比如 D 表示的就是双精度浮点数， C 表示的就是单精度浮点数复数。剩下的字符就是助记符，用来提示子方法的用途：例如，axpy 就是 a\*x + y，而 gemm 就是广义矩阵乘法（generalized matrix multiplication）。具体内容可以参考<http://web.stanford.edu/class/me200c/tutorial_77/18.1_blas.html>来查看一份子方法列表。

所有可用方法列表都在 `scipy.linalg.blas` 里面列出，更多内容可以参考文档 <http://docs.scipy.org/doc/scipy/reference/linalg.blas.html>

### Level 1 (向量运算，vector operations)


```python
x = np.random.randn(10)
y = np.arange(10)
a = 5
```


```python
# 计算 a*x + y
blas.daxpy(x, y, a=a)
```




    array([  2.1661,   2.0229,  10.6779,   3.9591,  11.5855,   9.8504,
             5.7268,   7.2025,  -4.6427,   6.5273])




```python
a*x + y
```




    array([  2.1661,   2.0229,  10.6779,   3.9591,  11.5855,   9.8504,
             5.7268,   7.2025,  -4.6427,   6.5273])




```python
%timeit blas.daxpy(x, y, a=a)
%timeit a*x + y
```

    100000 loops, best of 3: 2.86 µs per loop
    100000 loops, best of 3: 9.95 µs per loop



```python
# 计算一个向量的 L2 范数（L2 norm）
blas.dnrm2(x)
```




    3.6282




```python
la.norm(x)
```




    3.6282




```python
np.sqrt(np.sum(x**2.0))
```




    3.6282




```python
%timeit blas.dnrm2(x)
%timeit la.norm(x)
%timeit np.sqrt(np.sum(x**2.0))
```

    1000000 loops, best of 3: 426 ns per loop
    10000 loops, best of 3: 19.2 µs per loop
    100000 loops, best of 3: 19 µs per loop


### Level 2 (矩阵-向量运算，matrix-vector operations)


```python
alpha = 4.5
A = np.array(np.random.random((10,10)), order='F')
x = np.arange(10)

blas.dgemv(alpha, A, x)
```




    array([  78.8999,  114.8763,   72.1202,   92.6146,   90.4172,  107.0824,
            114.8884,   86.1693,  116.6441,   96.7868])




```python
alpha*np.dot(A, x)
```




    array([  78.8999,  114.8763,   72.1202,   92.6146,   90.4172,  107.0824,
            114.8884,   86.1693,  116.6441,   96.7868])




```python
%timeit blas.dgemv(alpha, A, x)
%timeit alpha*np.dot(A, x)
```

    100000 loops, best of 3: 3.48 µs per loop
    100000 loops, best of 3: 9.33 µs per loop


### Level 3 (矩阵-矩阵运算，matrix-matrix operations)


```python
alpha = 4.5
A = np.array(np.random.random((10,10)), order='F')
B = np.array(np.random.random((10,10)), order='F')
```


```python
# 使用 BLAS 的广义矩阵乘法（Generalized matrix multiplication using BLAS）
blas.dgemm(alpha, A, B)
```




    array([[ 10.0071,   5.5994,  10.6722,  10.1818,   6.8569,   4.2423,
              7.222 ,   7.1324,   9.1195,  11.0741],
           [ 18.9872,  12.9696,  18.7636,  13.0136,  13.493 ,   6.1952,
             10.8192,  11.5351,  13.6816,  18.3832],
           [ 18.009 ,  12.6172,  17.8935,  12.5518,  12.2465,   7.8041,
             11.2491,   8.6173,  11.609 ,  17.6362],
           [ 13.5082,   7.0443,  11.3788,   9.974 ,   8.0157,   5.4126,
              8.7252,   8.8902,  10.138 ,  12.5954],
           [ 13.8247,   9.0688,  14.543 ,  11.6694,   9.4144,   5.6807,
              9.7722,   9.443 ,  10.9853,  15.2538],
           [ 13.8562,  11.9601,  13.3123,  11.7678,   9.1058,   8.0995,
              8.8613,  11.9655,  13.8767,  15.3657],
           [ 15.7258,  11.6832,  14.5085,  11.1698,  10.5742,   7.6311,
              9.7384,   6.2711,  10.2244,  14.349 ],
           [  9.0453,   7.515 ,   7.6851,   9.2072,   6.036 ,   4.8551,
              5.6475,   7.4739,  10.1664,   8.6669],
           [ 19.035 ,  11.754 ,  17.0319,  14.4103,  10.8171,   8.4039,
             12.3496,  11.069 ,  13.188 ,  15.6755],
           [ 14.6138,  11.5766,  12.7563,  13.5197,   9.8956,   6.7646,
              8.8172,  10.4491,  14.5853,  13.942 ]])




```python
# 使用 numpy 的等效操作（Equivalent operation）
alpha * np.dot(A, B)
```




    array([[ 10.0071,   5.5994,  10.6722,  10.1818,   6.8569,   4.2423,
              7.222 ,   7.1324,   9.1195,  11.0741],
           [ 18.9872,  12.9696,  18.7636,  13.0136,  13.493 ,   6.1952,
             10.8192,  11.5351,  13.6816,  18.3832],
           [ 18.009 ,  12.6172,  17.8935,  12.5518,  12.2465,   7.8041,
             11.2491,   8.6173,  11.609 ,  17.6362],
           [ 13.5082,   7.0443,  11.3788,   9.974 ,   8.0157,   5.4126,
              8.7252,   8.8902,  10.138 ,  12.5954],
           [ 13.8247,   9.0688,  14.543 ,  11.6694,   9.4144,   5.6807,
              9.7722,   9.443 ,  10.9853,  15.2538],
           [ 13.8562,  11.9601,  13.3123,  11.7678,   9.1058,   8.0995,
              8.8613,  11.9655,  13.8767,  15.3657],
           [ 15.7258,  11.6832,  14.5085,  11.1698,  10.5742,   7.6311,
              9.7384,   6.2711,  10.2244,  14.349 ],
           [  9.0453,   7.515 ,   7.6851,   9.2072,   6.036 ,   4.8551,
              5.6475,   7.4739,  10.1664,   8.6669],
           [ 19.035 ,  11.754 ,  17.0319,  14.4103,  10.8171,   8.4039,
             12.3496,  11.069 ,  13.188 ,  15.6755],
           [ 14.6138,  11.5766,  12.7563,  13.5197,   9.8956,   6.7646,
              8.8172,  10.4491,  14.5853,  13.942 ]])




```python
%timeit blas.dgemm(alpha, A, B)
%timeit alpha * np.dot(A, B)
```

    100000 loops, best of 3: 4.5 µs per loop
    100000 loops, best of 3: 10.5 µs per loop


## LAPACK

LAPACK 提供的方法要比 BLAS 更高级，可以用于用于求解线性方程组（simultaneous linear equations），求线性方程组的最小二乘解（least-squares solutions of linear systems of equations），特征值问题（eigenvalue problems），奇异值问题（singular value problems）以及各种矩阵因式分解（matrix factorizations）。 LAPACK 中的大部分方法都利用了 BLAS 中提供的低级方法 。所以 LAPACK 里面的命名规则（naming convention）也都跟 BLAS 相似。LAPACK 的用户指南就是它的官方文档了，可以在下面这个链接里面找到：<http://www.netlib.org/lapack/lug/>.

The list of routines avaiable in `scipy.linalg.lapack` are listed at <http://docs.scipy.org/doc/scipy/reference/linalg.lapack.html>

### 解决最小二乘估计问题（least squares estimation problem）


```python
A = np.array([[-0.09,  0.14, -0.46,  0.68,  1.29], 
              [-1.56,  0.20,  0.29,  1.09,  0.51], 
              [-1.48, -0.43,  0.89, -0.71, -0.96], 
              [-1.09,  0.84,  0.77,  2.11, -1.27],
              [0.08,   0.55, -1.13,  0.14,  1.74],
              [-1.59, -0.72,  1.06,  1.24,  0.34]])
b = np.array([ 7.4, 4.2, -8.3, 1.8,  8.6,   2.1]) #.reshape(-1,1)
```


```python
# 设置 cond 来反映输入数据的相对精度（relative accuracy of the input data ）
v, x, s, rank, work, info = lapack.dgelss(A, b, cond=0.01) 
```


```python
x # 最小二乘法解（least squeares solution）
```




    array([ 0.6344,  0.9699, -1.4403,  3.3678,  3.3992, -0.0035])




```python
s # 矩阵 A 的奇异值（Singular values of A）
```




    array([  3.9997e+00,   2.9962e+00,   2.0001e+00,   9.9883e-01,   2.4992e-03])




```python
rank # 矩阵 A 的估计秩（Estimated rank of A）
```




    4



### 使用更方便的 `lstsq` 封装（Using convenient `lstsq` wrapper）


```python
x, res, rank, s = la.lstsq(A, b, cond=0.01)
```


```python
x
```




    array([ 0.6344,  0.9699, -1.4403,  3.3678,  3.3992])




```python
s
```




    array([  3.9997e+00,   2.9962e+00,   2.0001e+00,   9.9883e-01,   2.4992e-03])




```python
rank
```




    4




```python
%timeit lapack.dgelss(A, b, cond=0.01) 
%timeit la.lstsq(A, b, cond=0.01)
```

    10000 loops, best of 3: 24.7 µs per loop
    10000 loops, best of 3: 122 µs per loop


### Cholesky 矩阵分解（decomposition）


```python
x = np.random.multivariate_normal(np.zeros(5), np.eye(5), 100).T
s = np.cov(x)
```


```python
c, info = lapack.dpotrf(s)
```


```python
c
```




    array([[ 1.0173, -0.1342,  0.3306, -0.0822, -0.0075],
           [ 0.    ,  0.9128, -0.0327,  0.0869,  0.1051],
           [ 0.    ,  0.    ,  1.0893, -0.0078,  0.0871],
           [ 0.    ,  0.    ,  0.    ,  1.039 , -0.0178],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  1.0475]])



### 使用更方便的 `cholesky`封装（Using convenient `cholesky` wrapper）


```python
la.cholesky(s)
```




    array([[ 1.0173, -0.1342,  0.3306, -0.0822, -0.0075],
           [ 0.    ,  0.9128, -0.0327,  0.0869,  0.1051],
           [ 0.    ,  0.    ,  1.0893, -0.0078,  0.0871],
           [ 0.    ,  0.    ,  0.    ,  1.039 , -0.0178],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  1.0475]])




```python
%timeit lapack.dpotrf(s)
%timeit la.cholesky(s)
```

    100000 loops, best of 3: 3.03 µs per loop
    10000 loops, best of 3: 43.3 µs per loop



```python

```
