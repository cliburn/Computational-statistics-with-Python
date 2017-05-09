

```python
import collections
import Queue
import networkx as nx
import igraph
```

### 双向队列（提供最大长度时，也可以作为环）

* FIFO 和 LIFO 队列
* 管理资源池


```python
dq = collections.deque(range(5), maxlen=5)
```


```python
dq
```




    deque([0, 1, 2, 3, 4], maxlen=5)




```python
dq.appendleft(10)
```


```python
dq
```




    deque([10, 0, 1, 2, 3], maxlen=5)




```python
dq.extendleft([20,30])
```


```python
dq
```




    deque([30, 20, 10, 0, 1], maxlen=5)




```python
dq.extend([2,3,4])
```


```python
dq
```




    deque([0, 1, 2, 3, 4], maxlen=5)



### 堆

* 实现优先级队列和调度算法
* 对非常大的集合排序，它可能不能放进内存


```python
import heapq
```


```python
items = [(1, 'job 1'), (2, 'job 2'), (10, 'job 3'), (4, 'job 4')]
pq = []
for item in items:
    heapq.heappush(pq, item)
```


```python
heapq.heappop(pq)
```




    (1, 'job 1')




```python
heapq.heappop(pq)
```




    (2, 'job 2')




```python
heapq.heappop(pq)
```




    (4, 'job 4')




```python
heapq.heappush(pq, (11, 'job 5'))
heapq.heappush(pq, (9, 'job 6'))
```


```python
heapq.heappop(pq)
```




    (9, 'job 6')




```python
heapq.heappop(pq)
```




    (10, 'job 3')




```python
heapq.heappop(pq)
```




    (11, 'job 5')



### 图


```python
G = nx.watts_strogatz_graph(n=100, k=5, p=0.3)
mst = nx.minimum_spanning_tree(G)
pos=nx.graphviz_layout(mst, prog='neato', args='')
plt.figure(figsize=(8,8))
nx.draw(mst,pos, node_size=50,alpha=0.5,node_color="blue", with_labels=False)
plt.axis('equal');
```


![png](output_21_0.png)



```python
from igraph import *


```

### 排序


```python
xs = list(random_integers(0,100, 20))
xs
```




    [88, 84, 39, 5, 18, 41, 18, 91, 18, 44, 97, 95, 4, 86, 33, 63, 84, 47, 6, 27]




```python
sorted(xs)
```




    [4, 5, 6, 18, 18, 18, 27, 33, 39, 41, 44, 47, 63, 84, 84, 86, 88, 91, 95, 97]




```python
sorted(xs, key=str)
```




    [18, 18, 18, 27, 33, 39, 4, 41, 44, 47, 5, 6, 63, 84, 84, 86, 88, 91, 95, 97]




```python
sorted(xs, key=lambda x: x % 7, reverse=True)
```




    [41, 97, 6, 27, 5, 33, 47, 88, 39, 18, 18, 18, 95, 4, 44, 86, 84, 91, 63, 84]




```python
# 原地排序
xs.sort()
xs
```




    [4, 5, 6, 18, 18, 18, 27, 33, 39, 41, 44, 47, 63, 84, 84, 86, 88, 91, 95, 97]




```python
# 插入到有序列表中
import bisect

bisect.insort(xs, 50)
print xs
```

    [4, 5, 6, 18, 18, 18, 27, 33, 39, 41, 44, 47, 50, 63, 84, 84, 86, 88, 91, 95, 97]
    

### 搜索


```python
# 在无序列表中搜索
xs = list(random_integers(0,1e7, 1e6))
print xs[:20]
```

    [6015315, 620243, 1574916, 9029215, 5750922, 4691453, 2045193, 4657679, 524884, 6736598, 9234008, 9150991, 2299380, 3025996, 1637112, 4700335, 9248654, 1502960, 6325975, 7078459]
    


```python
# 如果搜索是重复的，首先对列表排序，之后二分搜索更加高效

import time

num_keys = 100
keys = list(set(xs))[:num_keys]

start = time.clock()
for i in keys:
    xs.index(i)
print '%.2f s' % (time.clock() - start)

start = time.clock()
ys = sorted(xs)
for i in keys:
    bisect.bisect(ys, i)
print '%.2f s' % (time.clock() - start)
```

    1.68 s
    0.99 s
    


```python

```

References
----

[使用数据结构和算法解决问题](http://www.interactivepython.org/courselib/static/pythonds/index.html)


```python

```
