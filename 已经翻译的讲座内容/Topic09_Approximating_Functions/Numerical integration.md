

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
%precision 4
import os, sys, glob
```

为了演示，我们尝试对一元高斯分布函数求积分。

### 符号积分


```python
from sympy import init_session
init_session() 
```

    IPython console for SymPy 0.7.5 (Python 2.7.5-64-bit) (ground types: python)
    
    These commands were executed:
    >>> from __future__ import division
    >>> from sympy import *
    >>> x, y, z, t = symbols('x y z t')
    >>> k, m, n = symbols('k m n', integer=True)
    >>> f, g, h = symbols('f g h', cls=Function)
    
    Documentation can be found at http://www.sympy.org
    

    WARNING: Hook shutdown_hook is deprecated. Use the atexit module instead.
    


```python
expr = integrate(1/sqrt(2*pi)*exp(-(x**2)/2), (x, -1, 1))
expr
```




$$\operatorname{erf}{\left (\frac{\sqrt{2}}{2} \right )}$$




```python
N(expr)
```




$$0.682689492137086$$



### 数值积分


```python
from scipy.integrate import quad, simps
from functools import partial
```


```python
f = lambda x: 1/np.sqrt(2*np.pi)*np.exp(-(x**2)/2)
s, err = quad(f, -1, 1)
s
```




$$0.682689492137$$



### 对等间隔的样本使用辛普森法则


```python
x = np.linspace(-1, 1, 101)
simps(f(x), x)
```




$$0.682689492998$$



### 蒙特卡罗积分


```python
def mc_uniform(f, low, high, n):
    volume = high - low
    samples = np.random.uniform(low, high, size=n)
    return volume * f(samples).sum()/n
```


```python
n =10000
low = -1
high = 1
mc_uniform(f, low, high, n)
```




$$0.683268145006$$




```python

```
