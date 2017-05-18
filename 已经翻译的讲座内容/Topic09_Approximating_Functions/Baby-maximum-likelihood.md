

```python
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
%precision 4
plt.style.use('ggplot')
```


```python
import statsmodels.api as sm 
import scipy.stats as st
```


```python
sunspots = sm.datasets.sunspots.load_pandas()
df = sunspots.data
```


```python
df.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>SUNACTIVITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 1700</td>
      <td>  5</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 1701</td>
      <td> 11</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 1702</td>
      <td> 16</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 1703</td>
      <td> 23</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 1704</td>
      <td> 36</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.plot(x='YEAR', y='SUNACTIVITY', legend=False);
```


![png](output_4_1.png)


在 1980 年 Barnes 所写的文章中，文章表明，观测到的太阳黑子数据的年平均值服从 $\chi^2$ 分布。我们使用最大可能性来实现它。


```python
shape, loc, scale = st.chi2.fit(df.SUNACTIVITY, floc=0)
rv = st.chi2(shape, loc, scale)

xs = np.linspace(0, 200, 100)
df.SUNACTIVITY.plot(kind='hist', bins=20, normed=True, alpha=0.4);
plt.plot(xs, rv.pdf(xs), 'red', linewidth=1)
plt.title('$\chi^2$ MLE fit to annual sunspot activity');
```


![png](output_6_0.png)


### 文章中建议的模拟模型


```python
phi1 = 1.90693
phi2 = -0.98751
theta1 = 0.78512
theta2 = -0.40662
alpha = 0.03
sigma = 0.4

np.random.seed(121)
n = 1000 + df.shape[0]
a = np.random.normal(0, sigma, n)
x = np.zeros(n)
y = np.zeros(n)
z = np.zeros(n)

for i in range(2, n):
    z[i:] = phi1*z[i-1] + phi2*z[i-2] + a[i] - theta1*a[i-1] - theta2*a[i-2]
    x[i] = z[i]**2
    y[i] = x[i] + alpha*(x[i-1] - x[i-2])**2
```


```python
start = 1000
stop = start + df.shape[0]
df['Simulated'] = y[start:stop]
df[['YEAR', 'SUNACTIVITY', 'Simulated']].plot(x='YEAR');
```


![png](output_9_0.png)



```python

```
