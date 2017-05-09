

```python
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
%precision 4
```




    u'%.4f'



分治
----

这个分类包含许多著名的递归算法，例如快速排序、选择排序、KD 树、二分搜索，以及其它。

### 归并排序


```python
from collections import deque

def merge(left, right):
    """将两个有序双向队列合并为一个有序列表。"""
    result = []
    left = deque(left)
    right = deque(right)
    while left or right:
        if left and right:
            if left[0] < right[0]:
                result.append(left.popleft())
            else:
                result.append(right.popleft())
        else:
            if not left:
                result.extend(right)
                right.clear()
            elif not right:
                result.extend(left)
                left.clear()
    return result

def mergesort(xs):
    """返回有序列表 xs。"""
    # base case
    n = len(xs)
    if n <= 1:
        return xs
    # recursive case
    else:
        left = xs[:(n/2)]
        right = xs[(n/2):]
        left = mergesort(left)
        right = mergesort(right)
        return merge(left, right)
```


```python
print mergesort([3,1,5,4,6,2,5])
```

    [1, 2, 3, 4, 5, 5, 6]
    

减治
----

这一类算法通常每次减少一个或多个元素，之后递归处理剩余的较小集合。

### 用于计算 GCD 的辗转相除法（欧几里得算法）

gcd(m, n) = gcd(n, m mod n)


```python
def gcd(m, n):
    """辗转相除法。"""

    # 基本情况
    if n == 0:
        return m
    # 递归情况
    else:
        result = gcd(n, m % n)
    return result
```


```python
print gcd(60, 240)
```

    60
    

贪心法
----

贪心法在每个阶段选取局部最优的解。取决于应用，它可能或者不可能找到全局最优的解。一个统计学的例子是前向或后向逐步回归。

### 使用 Prim 算法的最小生成树

我们会使用接邻列表结构来表示树，其中每条边是 (weight, vertex1, vertex2)。


```python
def prim_mst(edges):
    import heapq
    seen = set([])
    heapq.heapify(edges)
    
    tree = []
    if len(edges) == 0:
        return tree
    else:
        w, v1, v2 = heapq.heappop(edges)
        tree.append((w, v1, v2))
        seen.update((v1, v2))
    
    while True:
        if len(edges) == 0:
            break
        while len(edges) > 0:
            tmp = []
            w, v1, v2 = heapq.heappop(edges)
            # edge does not share vertex with existing tree
            if v1 not in seen and v2 not in seen:
                tmp.append((w, v1, v2))
            # edge is redundant
            elif v1 in seen and v2 in seen:
                pass
            # next best edge to add
            else:
                tree.append((w, v1, v2))
                seen.update((v1, v2))
                
        # put edges not used back into heap
        for item in tmp:
            heapq.heappush(edges, item)
    return tree
```


```python
edges = [(2.5,'a','b'), (1,'a','d'), (2,'b','d'), (3,'c','d')]
prim_mst(edges)
```




    [(1, 'a', 'd'), (2, 'b', 'd'), (3, 'c', 'd')]



动态规划
----

动态规划的本质特征，是问题拥有重复的子问题，它们需要重复计算多次。此外，组合这些子问题必须能够求解整个问题 -- 这被称为“最优结构”。如果是这样，动态规划是一种思想，如果我们缓存子问题的解（“记忆”），我们就不需要解决它们两次。


```python
# 一个普通的案例就是递归的斐波那契函数
from collections import defaultdict

def fib(n):
    """返回斐波那契数，1,1,2,3,5,8,...."""
    if n==0 or n==1:
        return 1
    else:
        print "fib(%d) + fib(%d)" % ((n-2), (n-1))
        return fib(n-2) + fib(n-1)
```


```python
# 注意重复调用
fib(6)
```

    fib(4) + fib(5)
    fib(2) + fib(3)
    fib(0) + fib(1)
    fib(1) + fib(2)
    fib(0) + fib(1)
    fib(3) + fib(4)
    fib(1) + fib(2)
    fib(0) + fib(1)
    fib(2) + fib(3)
    fib(0) + fib(1)
    fib(1) + fib(2)
    fib(0) + fib(1)
    




    13




```python
# 整理之后
def fib1(n):
    """返回斐波那契数，1,1,2,3,5,8,...."""
    return 1 if n < 2 else fib1(n-2) + fib1(n-1)
```


```python
# 我们可以通过缓存之前的计算结果来避免重复调用
cache = {}
def fib2(n):
    if n < 2:
        return 1
    if n not in cache:
        cache[n] = fib2(n-2) + fib2(n-1)
    return cache[n]
```


```python
# 对于更大的 n，缓存版本更快
%time fib1(33)
%time fib2(33)
```

    CPU times: user 2.61 s, sys: 13.3 ms, total: 2.62 s
    Wall time: 2.62 s
    CPU times: user 36 µs, sys: 0 ns, total: 36 µs
    Wall time: 41 µs
    




    5702887




```python
# 但是使用全局缓存并不是很好
# 我们需要为每个函数重复编写缓存代码
# 为了推广，我们编写一个记忆装饰器

def memoize(f):
    """用于缓存结果来重复使用的装饰器"""
    cache = {}
    def func(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return func
```


```python
@memoize
def fib3(n):
    """返回第 n 和斐波那契数，1,1,2,3,5,8,...."""
    return 1 if n < 2 else fib3(n-2) + fib3(n-1)
```


```python
# 性能
%time fib1(33)
%time fib2(33)
%time fib3(33)
```

    CPU times: user 2.62 s, sys: 8.5 ms, total: 2.63 s
    Wall time: 2.62 s
    CPU times: user 5 µs, sys: 1 µs, total: 6 µs
    Wall time: 8.82 µs
    CPU times: user 66 µs, sys: 29 µs, total: 95 µs
    Wall time: 98.9 µs
    




    5702887




```python
# 一个使用线性组合的例子
# 最少需要多少硬币才能组成 $0.63？ 
# 使用一美分、五美分、十美分和 25 美分

def factors(n, weights):
    """通过枚举硬币数量来爆破"""
    s = ((i,j,k,l) 
            for i in range(1+n/weights[0]) 
            for j in range(1+n/weights[1])
            for k in range(1+n/weights[2]) 
            for l in range(1+n/weights[3]))
    for _ in s:
        yield _

def combinations(n, weights):
    """返回硬币问题的解"""
    for f in factors(n, weights):
         if np.dot(f, weights) == n:
            yield f
```


```python
# 爆破解法非常慢
weights = (25, 10, 5, 1)
n = 47
%time sorted(list(combinations(n, weights)), key=sum)[0]
```

    CPU times: user 34.1 ms, sys: 10.3 ms, total: 44.4 ms
    Wall time: 37.5 ms
    




    (1, 2, 0, 2)




```python
# 让我们编写递归版本

def combinaions1(n, weights, solution):
    """返回组成 n 的最小硬币数量
    weights 是硬币的面值，降序排列
    """
    current = n
    if n in weights:
        return 1
    else:
        for i in (w for w in weights if w < n):
            count = combinaions1(n-i, weights) + 1
            if count < current:
                current = count
    return current
```


```python
# 递归解法更慢
weights = (25, 10, 5, 1)
n = 47
%time combinaions1(n, weights)
```


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-19-ae9ae642e9f4> in <module>()
          2 weights = (25, 10, 5, 1)
          3 n = 47
    ----> 4 get_ipython().magic(u'time combinaions1(n, weights)')
    

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/core/interactiveshell.pyc in magic(self, arg_s)
       2203         magic_name, _, magic_arg_s = arg_s.partition(' ')
       2204         magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
    -> 2205         return self.run_line_magic(magic_name, magic_arg_s)
       2206 
       2207     #-------------------------------------------------------------------------
    

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/core/interactiveshell.pyc in run_line_magic(self, magic_name, line)
       2124                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2125             with self.builtin_trap:
    -> 2126                 result = fn(*args,**kwargs)
       2127             return result
       2128 
    

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/core/magics/execution.pyc in time(self, line, cell, local_ns)
    

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/core/magic.pyc in <lambda>(f, *a, **k)
        191     # but it's overkill for just that one bit of state.
        192     def magic_deco(arg):
    --> 193         call = lambda f, *a, **k: f(*a, **k)
        194 
        195         if callable(arg):
    

    /Users/cliburn/anaconda/lib/python2.7/site-packages/IPython/core/magics/execution.pyc in time(self, line, cell, local_ns)
       1123         if mode=='eval':
       1124             st = clock2()
    -> 1125             out = eval(code, glob, local_ns)
       1126             end = clock2()
       1127         else:
    

    <timed eval> in <module>()
    

    TypeError: combinaions1() takes exactly 3 arguments (2 given)



```python
# 但是可以记忆
@memoize
def combinaions2(n, weights):
    """返回组成 n 的最小硬币数量
    weights 是硬币的面值，降序排列
    """
    current = n
    if n in weights:
        return 1
    else:
        for i in (w for w in weights if w < n):
            count = 1 + combinaions2(n-i, weights)
            if count < current:
                current = count
    return current
```


```python
weights = (25, 10, 5, 1)
n = 47
%time combinaions2(n, weights)
```

    CPU times: user 257 µs, sys: 218 µs, total: 475 µs
    Wall time: 338 µs
    




    5




```python
n = 213
%time sorted(list(combinations(n, weights)), key=sum)[0]
%time combinaions2(n, weights)
```

    CPU times: user 10.9 s, sys: 38 ms, total: 11 s
    Wall time: 10.9 s
    CPU times: user 820 µs, sys: 1 µs, total: 821 µs
    Wall time: 826 µs
    




    12




```python
# 缓存（记忆）是一个简单的方式，它比动态规划要好得多
# 而动态规划通常指代局部解的显式跟踪
```


```python
def combinations_dp1(weights, n, counts, show=False):
    """动态规划解法。
    counts 是一个向量，包含达到下标值所需的硬币数量。"""
    for i in range(n+1):
        s = i
        for j in (w for w in weights if w <= i):
            value = counts[i-j] + 1
            if value < s:
                s = value
        counts[i] = s
        if show:
            print counts
    return counts[n]
```


```python
weights = (25, 10, 5, 1)
n = 5
counts = [0]*(n+1)
combinations_dp1(weights, n, counts, show=True)
```

    [0, 0, 0, 0, 0, 0]
    [0, 1, 0, 0, 0, 0]
    [0, 1, 2, 0, 0, 0]
    [0, 1, 2, 3, 0, 0]
    [0, 1, 2, 3, 4, 0]
    [0, 1, 2, 3, 4, 1]
    




    1




```python
weights = (25, 10, 5, 1)
n = 11
counts = [0]*(n+1)
combinations_dp1(weights, n, counts, show=True)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0]
    [0, 1, 2, 3, 4, 1, 0, 0, 0, 0, 0, 0]
    [0, 1, 2, 3, 4, 1, 2, 0, 0, 0, 0, 0]
    [0, 1, 2, 3, 4, 1, 2, 3, 0, 0, 0, 0]
    [0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0]
    [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 0, 0]
    [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 0]
    [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2]
    




    2




```python
def combinations_dp2(weights, n, counts, used, show=False):
    """动态规划解法。
    counts 是一个向量，包含达到下标值所需的硬币数量。
    used 返回了实际的解法。"""
    for i in range(n+1):
        s = i
        t = 1
        for j in (w for w in weights if w <= i):
            value = counts[i-j] + 1
            if value < s:
                s = value
                t = j
        counts[i] = s
        used[i] = t
        if show:
            print counts
    return counts[n], used
```


```python
def change(n, used):
    """寻找用于组成总数的硬币"""
    idx = len(used) - 1
    result = []
    while n > 0:
        n -= used[idx]
        result.append(used[idx])
        idx -= used[idx]
    return result
```


```python
weights = (25, 10, 5, 1)
for n in range(4, 15):
    counts = [0]*(n+1)
    used = [0]*(n+1)
    ncoins, used = combinations_dp2(weights, n, counts, used, show=False)
    print n, change(n, used)
```

    4 [1, 1, 1, 1]
    5 [5]
    6 [5, 1]
    7 [5, 1, 1]
    8 [5, 1, 1, 1]
    9 [5, 1, 1, 1, 1]
    10 [10]
    11 [10, 1]
    12 [10, 1, 1]
    13 [10, 1, 1, 1]
    14 [10, 1, 1, 1, 1]
    

爬山法
----

爬山法类似于贪心法，并且在连续优化问题中更加普遍。我们在这篇文章中会看到几个例子。 

### 均值移动（Mean Shift）算法

给出描述和示例代码


```python
def gaussian_kernel(xs, m, h=1.0):
    # 定义了以 m 为中心的移动窗口的权重
    return (2*np.pi)**-0.5*np.exp(-((xs - m)/h)**2)
```


```python
xs = np.linspace(0, 10, 100)
m = 3
plt.plot(xs, gaussian_kernel(xs, m));
```


![png](output_39_0.png)



```python
def mean_shift(xs, kernel, x=None, niter=1000, tol=1e-6):
    """使用爬山法找出局部模式。"""
    if x is None:
        x = xs.mean()
    for i in range(niter):
        m = (kernel(xs, x)*xs).sum()/kernel(xs, x).sum() - x
        if np.sum(m**2) < tol: 
            break
        x += m
    return i, x
```


```python
xs = np.concatenate([np.random.normal(-5, 1, 1000), 
                     np.random.normal(0,2,1000), 
                     np.random.normal(4, 1, 1000)])
```


```python
from functools import partial

kernel = gaussian_kernel
i1, m1 = mean_shift(xs, kernel, x=1, niter=1000, tol=1e-12)
i2, m2 = mean_shift(xs, kernel, x=-7, niter=1000, tol=1e-12)
i3, m3 = mean_shift(xs, kernel, x=7, niter=1000, tol=1e-12)
print i1, m1
print i2, m2
print i3, m3
```

    161 0.307380887446
    32 -4.89665274673
    53 3.84321153311
    


```python
import scipy.stats as stats
xp = np.linspace(0, 1.0, 100)
plt.hist(xs, 50, histtype='step', normed=True);
plt.axvline(m1, c='red')
plt.axvline(m2, c='red')
plt.axvline(m3, c='red')
```




    <matplotlib.lines.Line2D at 0x114dafed0>




![png](output_43_1.png)



```python
np.save('x1d.npy', xs)
```

归约和变换
----

并不需要特别多的算法，因为特定问题，在变换的帮助下，可能能够映射到一个已解决的问题。例如，广义线性模型可以归约到简单线性模型的框架，使用一种变换（“链接函数”）。图算法是另一个例子 -- 数量惊人的图问题可以归约为网络流问题，其中存在高效的算法。


```python

```


```python

```

蒙特卡罗方法
----

任何使用随机数生成器来寻找解的方法都是蒙特卡罗方法 -- 基本上所有统计学模拟都位于这个分类，包含重要性采样和 MCMC。同时，它常用于高维空间的数值积分，因为它们的复杂度只随样本数 $n$ 增大而增大，而不是维度 $d$。

### 寻找 $\pi$

用于展示蒙特卡罗积分的常用示例是估计 $\pi$。我们在单位正方形中生成随机点，并计算位于四分之一单位圆内的比值。这个比值的期望是 $\pi r^2$，由于 $r=1$，$\pi$ 可以估计为 4 $\times$ 这个比值。


```python
n = 1000000
pts = np.random.uniform(0,1,(n,2))
frac = np.sum(pts[:,0]**2 + pts[:,1]**2 < 1)/float(n)
print 4*frac
```

    3.139952
    


```python

```

参考
----

找零问题中关于记忆和动态规划用法的一篇[教程](http://interactivepython.org/courselib/static/pythonds/Recursion/DynamicProgramming.html)


```python

```
