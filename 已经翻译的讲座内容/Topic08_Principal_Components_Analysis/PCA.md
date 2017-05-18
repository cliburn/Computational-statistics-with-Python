

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
%precision 4
import os, sys, glob
plt.style.use('ggplot')
```


```python
from scipy.stats import ttest_ind as t
import matplotlib.colors as mcolors
```

方差和协方差
----

回忆协方差的公式：

$$
\text{Cov}(X, Y) = \frac{\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{n-1}
$$

其中 $\text{Cov}(X, X)$ 是 $X$ 的样本方差。


```python
def cov(x, y):
    """返回向量 x 和 y 的协方差。"""
    xbar = x.mean()
    ybar = y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)
```


```python
X = np.random.random(10)
Y = np.random.random(10)
```


```python
np.array([[cov(X, X), cov(X, Y)], [cov(Y, X), cov(Y,Y)]])
```




    array([[ 0.0781, -0.007 ],
           [-0.007 ,  0.0707]])




```python
# 这当然可以使用 NumPy 的 cov() 自带函数计算
np.cov(X, Y)
```




    array([[ 0.0781, -0.007 ],
           [-0.007 ,  0.0707]])




```python
# 一次性计算多个变量的扩展
Z = np.random.random(10)
np.cov([X, Y, Z])
```




    array([[ 0.0781, -0.007 , -0.0108],
           [-0.007 ,  0.0707,  0.0133],
           [-0.0108,  0.0133,  0.078 ]])



协方差矩阵的特征值分解
----


```python
mu = [0,0]
sigma = [[0.6,0.2],[0.2,0.2]]
n = 1000
x = np.random.multivariate_normal(mu, sigma, n).T
```


```python
A = np.cov(x)
```


```python
m = np.array([[1,2,3],[6,5,4]])
ms = m - m.mean(1).reshape(2,1)
np.dot(ms, ms.T)/2
```




    array([[ 1., -1.],
           [-1.,  1.]])




```python
e, v = np.linalg.eig(A)
```


```python
plt.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e, v.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3])
plt.title('Eigenvectors of covariance matrix scaled by eigenvalue.');
```


![png](output_13_0.png)


主成分分析（PCA）
----

主成分分析（PCA）的主要含义是，寻找并排列协方差矩阵的所有特征值和特征向量。这非常实用，因为高维数据（带有 $p$ 个属性）的所有变化都在较小的维度 $k$ 中，也就是位于由协方差矩阵的特征向量所划分的子空间，协方差矩阵拥有 $k$ 个最大的特征值。如果我们将原始数据投影到这个子空间，我们就可以对其降维（从 $p$ 到 $k$），而不会损失信息。

数值上看，PCA 通常使用数据矩阵的奇异值分解，而不是协方差矩阵的特征值分解。下一节会解释为什么这么做。

### 所有属性向量均值为零的数据矩阵

\begin{align}
\text{Cov}(X, Y) &= \frac{\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{n-1} \\
  &= \frac{\sum_{i=1}^nX_iY_i}{n-1} \\
  &= \frac{XY^T}{n-1}
\end{align}

所以如果数据集 $X$ 的所有属性向量的均值都为零，它的协方差矩阵就是 $XX^T/(n-1)$。

换句话说，我们也可以从半正定阵 $XX^T$ 获得协方差矩阵的特征值分解。


```python
e1, v1 = np.linalg.eig(np.dot(x, x.T)/(n-1))
```


```python
plt.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e1, v1.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3]);
```


![png](output_17_0.png)


使用 PCA 的基变换
----

### 我们可以变换原始数据集，使特征向量是基向量，并找到数据点在新的基底下的坐标。

这就是线性代数模块中的基变换。首先，要注意协方差矩阵是个实对称矩阵，所以特征向量矩阵是正交矩阵。


```python
e, v = np.linalg.eig(np.cov(x))
v.dot(v.T)
```




    array([[ 1.,  0.],
           [ 0.,  1.]])



### 基变换的线性代数回顾

让我们考虑 $\mathbb{R}^2$ 中两个不同的基向量组 $B$ 和 $B'$。假设 $B$ 的基向量是 ${u, v}$，$B'$ 的基向量是 ${u', v'}$。同样假设 $B'$ 的基向量 ${u', v'}$ 在 $B$ 下的坐标为 $u' = (a, b)$ 以及 $v' = (c, d)$，也就是 $u' = au + bv$ 以及 $v' = cu + dv$，因为这就是坐标的含义。

假设我们想找出基底 $B'$ 中的向量 $w = (x', y')$ 在 $B$ 中是什么。我们做一些代数：

\begin{align}
w' &= x'u' + y'v' \\
&= x'(au + bv) + y'(cu + dv) \\
&= (ax' + cy')u + (bx' + dy')v
\end{align}

所以

\\[
[w]_B =
\left( \begin{array}{cc}
ax' + cy' \\
bx' + dy' 
\end{array} \right)
\\]

表达为矩阵形式
\\[
[w]_B =
\left( \begin{array}{cc}
a & c \\
b & d
\end{array} \right)
%
\left( \begin{array}{c}
x' \\
y'
\end{array} \right)
\\]

因为 $[w]_{B'} = (x', y')$，我们看到，我们所需的线性变换将 $B'$ 中的向量转换为 $B$ 中的向量（译者注，也能将 $B$ 本身转换为 $B'$ 本身，因为基底乘坐标是个定值）。我们简单左乘*坐标变换*矩阵 $P$，将基向量用作列向量，就能得到它。

\\[
P =
\left( \begin{array}{cc}
a & c \\
b & d 
\end{array} \right)
\\]

为了从 $B$ 变换为 $B'$，我们需要左乘 $P^{-1}$。

为了从标准基（$B$）转换为由特征向量提供的基底（$B'$），我们左乘特征向量的逆 $V^{-1}$。由于特征向量矩阵 $V$ 是正交的，$V^T = V^{-1}$。 假设存在矩阵 $M$，它的列是新的基向量，向量 $x$ 的新坐标就是 $M^{-1}x$。所以为了将基底变换为特征值向量矩阵（也就是找到向量 $x$ 在特征值向量所划分的空间中的坐标），我们只需要将 $x$ 左乘 $V^{-1} = V^T$。

基变换的图解
----

![$A = Q^{-1}\Lambda Q$](https://raw.githubusercontent.com/cliburn/Computational-statistics-with-Python/master/Lectures/Topic08_Principal_Components_Analysis/spectral.png)

假设向量 $u$ 在标准基 $B$ 中，矩阵 $A$ 可以将 $u$ 映射为 $v$，也位于 $B$ 中。我们可以使用 $A$ 的特征值来组成新的基底 $B$。像上面所说，为了将向量 $u$ 从空间 $B$ 中转换为空间 $B'$ 中的向量 $u'$，我们需要左乘 $Q^{-1}$，它是以特征向量作为列向量的矩阵的逆。现在，在特征向量的基底中，$A$ 的等价操作是对角矩阵 $\Lambda$ -- 它将 $u'$ 变为 $v'$。最后，我们通过左乘 $Q$，将 $v'$ 转换回标准基中的向量 $v$。


```python
ys = np.dot(v1.T, x)
```

#### 主成分

主成分仅仅是用作基向量的协方差的特征向量。每个原始数据点都表达为主成分的线性组合，从而产生新的坐标系。


```python
plt.scatter(ys[0,:], ys[1,:], alpha=0.2)
for e_, v_ in zip(e1, np.eye(2)):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3]);
```


![png](output_25_0.png)


例如，如果我们仅仅使用`ys`的第一列，我们就将数据投影到第一个主成分上，使用单一属性捕获数据中最主要的变动，它是原始数据的线性组合。

#### 转换回原始坐标

为了解释，我们可能需要将数据集转换（降维）为原始属性的坐标系。这只是另一个线性变换（矩阵乘法）。


```python
zs = np.dot(v1, ys)
```


```python
plt.scatter(zs[0,:], zs[1,:], alpha=0.2)
for e_, v_ in zip(e1, v1.T):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3]);
```


![png](output_29_0.png)



```python
u, s, v = np.linalg.svd(x)
u.dot(u.T)
```




    array([[  1.0000e+00,  -5.5511e-17],
           [ -5.5511e-17,   1.0000e+00]])



使用 PCA 来降维
----

我们看到了协方差矩阵的迷之分解

$$
A = Q^{-1}\Lambda Q
$$

假设 $\Lambda$ 是秩为 $p$ 的矩阵。为了将维度降至 $k \le p$，我们简单将所有 $\Lambda$ 对角线上的值设为零，除了前 $k$个。这相当于忽略所有主成分，除了前 $k$ 个。

这会实现什么呢？回忆 $A$ 是个协方差矩阵，矩阵的迹是 total variability，因为它是方差的和。


```python
A
```




    array([[ 0.5997,  0.2015],
           [ 0.2015,  0.2011]])




```python
A.trace()
```




    0.8009




```python
e, v = np.linalg.eig(A)
D = np.diag(e)
D
```




    array([[ 0.6838,  0.    ],
           [ 0.    ,  0.117 ]])




```python
D.trace()
```




    0.8009




```python
D[0,0]/D.trace()
```




    0.8539



由于迹在基变换下保持不变，整体可变性也没有被 PCA 改变。通过只保留前 $k$ 个主成分，我们仍旧可以“解释”整体可变性的 $\sum_{i=1}^k e[i]/\sum{e}$。有时，降维特指保留足够的主成分，使 $90\%$ 的整体可变性能够解释。

### 将奇异值分解用于 PCA

SVD 是数据矩阵 $X = U S V^T$ 的分解，其中 $U$ 和 $V$ 都是正交矩阵，$S$ 是对角矩阵。

回忆一下，正交矩阵的转置也是它的你，所以如果我们右乘 $X^T$，我们就可以这样化简：

\begin{align}
X &= U S V^T \\
X X^T &= U S V^T (U S V^T)^T \\
 &= U S V^T V S U^T \\
 &= U S^2 U^T
\end{align}

与矩阵 $A = W \Lambda W^{-1}$ 的特征值分解相比，我们看到，SVD 向我们提供了 $XX^T$ 的特征值分解，我们之前看到，本质上是均值为零的数据矩阵的协方差的放大版本，它的特征向量由 $U$ 提供，特征值由 $S^2$ 提供（放大了 $n-1$ 倍）。



```python
u, s, v = np.linalg.svd(x)
```


```python
e2 = s**2/(n-1)
v2 = u
plt.scatter(x[0,:], x[1,:], alpha=0.2)
for e_, v_ in zip(e2, v2):
    plt.plot([0, 3*e_*v_[0]], [0, 3*e_*v_[1]], 'r-', lw=2)
plt.axis([-3,3,-3,3]);
```


![png](output_40_0.png)



```python
v1 # from eigenvectors of covariance matrix
```




    array([[-0.923 , -0.3849],
           [-0.3849,  0.923 ]])




```python
v2 # from SVD
```




    array([[-0.923 , -0.3849],
           [-0.3849,  0.923 ]])




```python
e1 # from eigenvalues of covariance matrix
```




    array([ 0.6843,  0.1171])




```python
e2 # from SVD
```




    array([ 0.6843,  0.1171])




```python

```


```python

```


```python

```


```python
a0 = np.random.normal(0,1,100)
a1 = a0 + np.random.normal(0,3,100)
a2 = 2*a0 + a1 + np.random.normal(5,0.01,100)
xs = np.vstack([a0, a1, a2])
xs.shape
```




    (3, 100)




```python
C = np.cov(xs)
```


```python
C
```




    array([[  0.9401,   0.7277,   2.6087],
           [  0.7277,  10.4607,  11.9147],
           [  2.6087,  11.9147,  17.1322]])




```python
e, v = np.linalg.eig(C)
```


```python
v
```




    array([[-0.0985, -0.8167,  0.5686],
           [-0.5982, -0.408 , -0.6896],
           [-0.7952,  0.4081,  0.4484]])




```python
U, s, V = np.linalg.svd(xs)
```


```python
(s**2)/(99)
```




    array([ 41.7498,   7.1581,   0.7866])




```python
U
```




    array([[-0.0615,  0.0082, -0.9981],
           [-0.3147, -0.9491,  0.0116],
           [-0.9472,  0.3148,  0.061 ]])




```python


np.round(e/e.sum(), 4)
```




    array([ 0.9496,  0.    ,  0.0504])




```python
ys = np.linalg.inv(v).dot(xs)
```


```python
plt.scatter(ys[0,:], ys[2,:])
```




    <matplotlib.collections.PathCollection at 0x11783d7d0>




![png](output_58_1.png)



```python
zs = v[:, [0,2]].dot(ys[[0,2],:])
zs.shape
```




    (3, 100)




```python
v[:, [0,2]]
```




    array([[-0.1033,  0.5677],
           [-0.5924, -0.6947],
           [-0.799 ,  0.4417]])




```python
plt.figure(figsize=(12.,4))
plt.subplot(1,3,1)
plt.scatter(zs[0,:], zs[1,:])
plt.subplot(1,3,2)
plt.scatter(zs[0,:], zs[2,:])
plt.subplot(1,3,3)
plt.scatter(zs[1,:], zs[2,:])
```




    <matplotlib.collections.PathCollection at 0x117bcc110>




![png](output_61_1.png)



```python
plt.figure(figsize=(12.,4))
plt.subplot(1,3,1)
plt.scatter(xs[0,:], xs[1,:])
plt.subplot(1,3,2)
plt.scatter(xs[0,:], xs[2,:])
plt.subplot(1,3,3)
plt.scatter(xs[1,:], xs[2,:])
```




    <matplotlib.collections.PathCollection at 0x117f81550>




![png](output_62_1.png)



```python
plt.figure(figsize=(12.,4))
plt.subplot(1,3,1)
plt.scatter(ys[0,:], ys[1,:])
plt.subplot(1,3,2)
plt.scatter(ys[0,:], ys[2,:])
plt.subplot(1,3,3)
plt.scatter(ys[1,:], ys[2,:])
```




    <matplotlib.collections.PathCollection at 0x118338990>




![png](output_63_1.png)



```python

```
