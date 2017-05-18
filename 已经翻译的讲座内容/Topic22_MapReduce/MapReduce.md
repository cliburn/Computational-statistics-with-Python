
### 概述

MapReduce 本质上是一个设计模式，用于将多个难以并行化的问题分配给多个处理器，基于`map`和`reduce`的函数式编程。如果原始问题规模太大，难以放进单个电脑的内存，map-reduce 就会非常实用。不像一些并行计算算法，map-reduce 的概念十分简单。


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```


```python
%precision 4
import operator
from collections import Counter
from multiprocessing import Pool
```

### 回顾`map`、`reduce`和基本的并行处理

**字数统计示例**


```python
# 这个示例统计文档集中的每个单词
# 假设每个文档是 DNA 序列，我们想要统计每个核苷酸 'A', 'C', 'T', G' 的出现频率

# make up some data repreensting DNA sequences
nseqs= 10
seq_length = 10000
seqs = [''.join(s) for s in np.random.choice(list('ACTG'), size=(nseqs, seq_length), replace=True, p=[0.1,0.2,0.3,0.4])]

# map-reduce 的最基本形式需要一个映射器（mapper）函数
# 它处理每个独立的块，并返回汇总结果
# 之后是一个归约器（reducer），它组合映射操作的结果

# 返回数量字典的 mapper 函数
def mapper(seq):
    c = Counter(seq)
    return c

#  合并每个计数对象的相似项的 reducer 函数
def reducer(counter1, counter2):
    return counter1 + counter2

# 将函数映射到序列上
counters = map(mapper, seqs)

# 归约返回的字典列表
counts = reduce(reducer, counters)

counts
```




    Counter({'G': 40044, 'T': 29914, 'C': 20070, 'A': 9972})




```python
# 这可以写成一行
counts = reduce(reducer, map(mapper, seqs))
```


```python
# 我们可以使用多处理器来并行计算它
pool =  Pool()
counts = reduce(reducer, pool.map(mapper, seqs))
pool.close()
counts
```




    Counter({'G': 40065, 'T': 30029, 'C': 19972, 'A': 9934})



**用于计算均值和方差的 map 和 reduce**


```python
# 以一个更加普通的例子继续
# 让我们计算巨大数据样本的均值和方差

nrows = 10
ncols = 1000
a = 2
b = 3

xs = np.random.beta(a, b, (nrows, ncols))
xs.size, xs.mean(), xs.var()
```




    (10000, 0.4023, 0.0395)




```python
# 假设我们决定按行分割
# 我们需要 mapper 函数来返回每行的大小、均值和方差
# 以及 reducer 函数来组合行的汇总

def mapper(x):
    return len(x), x.mean(), x.var()

def reducer(s1, s2):
    (n1, m1, v1), (n2, m2, v2) = s1, s2
    n = n1 + n2
    m = (n1*m1 + n2*m2)/(n1 + n2)
    v = (n1*v1 + n2*v2)/ n + ((n1*n2) * ((m2 - m1) / n)**2)
    return n, m, v
    
reduce(reducer, map(mapper, xs))
```




    (10000, 0.4023, 0.0395)




```python
# 并行版本
pool =  Pool()
n, m, v = reduce(reducer, pool.map(mapper, xs))
pool.close()
n, m, v
```




    (10000, 0.4023, 0.0395)



### 用于多元线性回归、朴素贝叶斯分类和 k-means 聚类的 Map 和 reduce


```python
# 未完待续
```

### 使用 AWS EMR


```python
# 未完待续
```


```python

```
