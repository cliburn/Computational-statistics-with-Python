

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



函数猜测
----

我们经常需要猜测，提供的数据看上去像什么函数。有两个通用方法

- 使用泛化的函数族（函数近似）
- 估计特定模型族的参数（函数估计）

函数近似
----

警告：大多数这些方法都不适用于高维数据

### 使用正交基

#### 多项式函数

基于泰勒级数的多项式近似，可能是这个方法的最熟悉的例子。这种情况下，基底是 ${1, x, x^2, x^3, \ldots}$，误差由我们选择截断级数的位置决定。


```python
np.random.seed(1)
xs = np.linspace(0, 2*np.pi, 27)
ys = np.sin(xs) + np.random.normal(0, 0.3, xs.shape)

fit = np.polyfit(xs, ys, 3)
p = np.poly1d(fit)

xp = np.linspace(0, 2*np.pi, 100)
plt.plot(xs, ys, 'o');
plt.plot(xp, p(xp), 'r');
plt.xlim([-0.1, 2*np.pi+0.1])
```




    (-0.1000, 6.3832)




![png](output_5_1.png)



```python
# 对于非参数拟合，每个数据点自然要使用一个变量

fit = np.polyfit(xs, ys, len(xs)-1)
p = np.poly1d(fit)

xp = np.linspace(0, 2*np.pi, 100)
plt.plot(xs, ys, 'o');
plt.plot(xp, p(xp), 'r');
plt.xlim([-0.1, 2*np.pi+0.1])
```

    /Users/cliburn/anaconda/lib/python2.7/site-packages/numpy/lib/polynomial.py:588: RankWarning: Polyfit may be poorly conditioned
      warnings.warn(msg, RankWarning)
    




    (-0.1000, 6.3832)




![png](output_6_2.png)



```python
# 但是多项式拥有一个讨厌的属性，它是无界的

xp = np.linspace(-1, 1+2*np.pi, 100)
plt.plot(xs, ys, 'o');
plt.plot(xp, p(xp), 'r');
plt.xlim([-1.1, 2*np.pi+1.1])
```




    (-1.1000, 7.3832)




![png](output_7_1.png)



```python
# 这由于过拟合产生的过大系数导致
# 并且可以通过否定过大系数来改善
# 通过“岭回归”或者在系数分布中使用先验
# 这被称为正则化 - 之后可能会详细涵盖
```

#### 切比雪夫多项式

它是一个多项式正交基，拥有比常规多项式拟合更好的收敛性质。但是，它们拥有相同的限制，易受过拟合影响。


```python
import numpy.polynomial.chebyshev as chb

np.random.seed(1)
xs = np.linspace(0, 2*np.pi, 27)
ys = np.sin(xs) + np.random.normal(0, 0.3, xs.shape)

fit = chb.chebfit(xs, ys, 3)

xp = np.linspace(0, 2*np.pi, 100)
plt.plot(xs, ys, 'o');
plt.plot(xp, chb.chebval(xp, fit), 'r');
plt.xlim([-0.1, 2*np.pi+0.1])
```




    (-0.1000, 6.3832)




![png](output_10_1.png)


#### 傅里叶基

对于周期函数（特别是时间序列），傅里叶级数是用于函数近似的良好基底。


```python
from scipy import fftpack

np.random.seed(1)
xs = np.linspace(0, 2*np.pi, 27)
ys = np.sin(xs) + np.random.normal(0, 0.3, xs.shape)

ys_fft = fftpack.fft(ys, axis=0)
ys_freq = fftpack.fftfreq(ys.shape[0], xs[1] - xs[0])
periods = 1.0 / ys_freq

plt.figure()
plt.plot(xs, ys, 'o')
plt.plot(xs, fftpack.ifft(ys_fft), 'red')

plt.figure()
plt.polar(periods, abs(ys_fft), 'o')
```




    [<matplotlib.lines.Line2D at 0x1161f8190>]




![png](output_12_1.png)



![png](output_12_2.png)



```python
# FFT 适用于从噪声中分离信号

pidxs = np.where(ys_freq > 0)
freqs = ys_freq[pidxs]
power = np.abs(ys_fft)[pidxs]
freq = freqs[power.argmax()]
ys_fft[np.abs(ys_freq) > freq] = 0

plt.plot(xs, ys, 'o')
plt.plot(xs, fftpack.ifft(ys_fft), 'red')
```




    [<matplotlib.lines.Line2D at 0x116189610>]




![png](output_13_1.png)


### 使用样条

分段多项式近似。


```python
from scipy.interpolate import UnivariateSpline

np.random.seed(1)
xs = np.linspace(0, 2*np.pi, 27)
ys = np.sin(xs) + np.random.normal(0, 0.3, xs.shape)

s = UnivariateSpline(xs, ys, s=3)

xp = np.linspace(0, 2*np.pi, 100)
plt.plot(xs, ys, 'o');
plt.plot(xp, s(xp), 'r');
plt.xlim([-0.1, 2*np.pi+0.1])
```




    (-0.1000, 6.3832)




![png](output_15_1.png)


Approximating PDFs
----


```python

```
