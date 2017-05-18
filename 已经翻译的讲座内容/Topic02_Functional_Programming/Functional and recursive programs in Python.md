
Python 中的功能和递归程序
====
先用如下命令使用 pip 来安装模块 functional

```
pip install functional
```

迭代器， 列表推导，生成器表达式，以及其他高级函数（Iterators, list comprehension, generator expressions and higher order functions）
----

### 交互方式（Imperative style）


```python
xs = []
for i in range(10):
    if i % 2 == 0:
        xs.append(i*i)
xs
```




    [0, 4, 16, 36, 64]



### 使用列表推导


```python
### 一个列表推导式会在内存中创建一个列表
xs = [i*i for i in range(10) if i % 2 == 0] # 在列表推导中这一步已经创建了列表了，内存已经分配了
xs
```




    [0, 4, 16, 36, 64]



### 使用生成器表达式


```python
### 生成器表达式看着和列表推导很相似，不同的是要手动调用才能返回值
xs = (i*i for i in range(10) if i % 2 == 0) # 使用生成器表达式，这一步还没创建列表，内存没有分配
list(xs)
```




    [0, 4, 16, 36, 64]



### 使用其他高级函数 - map 和 filter


```python
def square(x):
    return x*x

def even(x):
    return x % 2 == 0

xs = map(square, filter(even, range(10)))
list(xs)
```




    [0, 4, 16, 36, 64]




```python
### 使用匿名函数

xs = map(lambda x: x*x, filter(lambda x: x % 2 == 0, range(10)))
list(xs)
```




    [0, 4, 16, 36, 64]



Operators 模块和 Reductions（减少下降？规约？）
----

### 交互方式


```python
s = 10
for i in range(10):
    s += i
s
```




    55



### 使用 operator.add 和 reduce 函数


```python
import operator
from functools import reduce

s = reduce(operator.add, range(10), 10)
s
```




    55




```python
# 然而实际上平时咱就用sum函数加一下就得了，才没那么复杂的呢
sum(range(10)) + 10
```




    55




```python
### 映射-规约 map-reduce 原文这里用了idioom？估计是写错了吧，应该是idiom？
```


```python
# 但是理解这些高级函数对于我们日后写并行程序很有用处
import numpy as np

# 加入我们有了一个由数值列表组成的列表
xs = np.random.randint(0,10,(5,12)).tolist()
xs
```




    [[1, 3, 2, 9, 7, 9, 2, 2, 7, 1, 6, 6],
     [3, 9, 0, 1, 0, 5, 9, 7, 6, 8, 2, 6],
     [9, 1, 3, 6, 5, 7, 5, 6, 4, 0, 8, 3],
     [6, 7, 1, 8, 6, 4, 4, 3, 3, 1, 5, 1],
     [3, 8, 7, 5, 2, 8, 1, 1, 1, 5, 0, 3]]




```python
# 然后我们要把这些列表中的值的平方都加到一起来(就是平方和)成为一个个的值
# 然后找到最大值
# 这就可以用到 映射-规约模式 map-reduce pattern

# 所以这里就在一个并行程序中使用列表推导式
# 其中每一个 map 操作都将由单独的一个处理器核心来运行

row_sums = []
for x in xs:
    # 首先对每个列表中的所有元素都求平方，要用到的就是一个map映射运算，当然这里还用到了lambda函数
    row = map(lambda x: x*x, x)
    # 然后用reduce规约运算，把所有平方过的运算累加起来得到row_sum，累加得到的结果再存储到row_sums里面去
    row_sum = reduce(operator.add, row)
    row_sums.append(row_sum)
print(row_sums)
```

    [355, 386, 351, 263, 252]



```python
# 接下来就可以用一个单个的处理器来收集这个结果，很容易就找到最大值了
max(row_sums)
```




    386




```python
# 当然刚刚这一切也可以用一行代码来实现，但是如果这样的代码让你去读，不觉得太残忍么？

max(reduce(operator.add, map(lambda x: x*x, x)) for x in xs)
```




    386




```python
# 当然了，比较正规的应用场景，我们还是用向量运算来实现的（译者注：估计这里的正规是指对性能比较关注？向量运算可以使用某些已经优化过的例如numpy之类的库，貌似速度是最优的？）
xs = np.array(xs)
max(np.sum(xs*xs, axis=1)) # axis=1 sums by rows
```




    386



### 函数创作（Function composition）

只要安装了`functional`模块，咱们就能获得好多好多的函数式编程的工具了

```bash
pip install functional
```


```python
from functional import compose, foldl, partial
```


```python
# partial 可以减少一个函数需要的变量（argumnents）的个数
# 在有部分参数（parameters）给定（"filled in"）的情况下， partial 就可以返回一个新的函数

def f(x, y, z):
    return x, y, z

g = partial(f, 1)
h = partial(f, 1, 2)
k = partial(g, 2)
j = partial(f, z=3)

print(f(1,2,3))
print(g(2,3))
print(h(3))
print(k(3))
print(j(1, 2))
```

    (1, 2, 3)
    (1, 2, 3)
    (1, 2, 3)
    (1, 2, 3)
    (1, 2, 3)



```python
# 比如下面这个简单的例子

def add(a, b):
    return a + b

add10 = partial(add, b=10)
print(add10(5))
```

    15



```python
# compose 可以用来创作新的函数

def f(x):
    return x**2

def g(x):
    return x + 2

fg = compose(f, g)
x =3
fg(x)
```




    25




```python
# 我们可以结合 reduce 和 parital 来创作多个函数

# reduce 接受两个变量（arguments） - 函数以及一个可迭代对象（iterable）
# 首先用 parital 来 预加载（"pre-load"） reduce，使用 compose 函数作为改造对象
composeN = partial(reduce, compose)

# 然后就可以创作多个函数了

fggf = composeN([f, g, g, f])
fggf(x)
```




    169




```python
# 分开每步来检查一下
f(g(g(f(x))))
```




    169




```python
从 <https://docs.python.org/release/2.6/howto/functional.html> 这份文档中我们了解到：

foldl() 接收一个二值化函数（binary function），一个起始值(通常可能就是0)，然后还要有一个可迭代对象（iterable）。
运行内容是先针对起始值以及列表中第一个元素进行该二值化函数运算，然后用结果跟列表中第二个元素进行该函数运算，然后用结果跟第三个元素进行该函数运算，以此类推

这就意味着，假如有下面这样的调用：
```python
foldl(f, 0, [1, 2, 3])
```
实际上就等价于：
```python
f(f(f(0, 1), 2), 3)
```
```

### 用高级函数实现的 bootstrap 样例（自助抽样？）


```python
# 我们希望 bootstrap 函数能够用于任何数据分布以及任何我们感兴趣的统计模型

def bootstrap(xs, nsamples, low, high, statistic):
    """A simple bootstrap function.
    xs = data for bootstrapping
    nsamples = number of bootstrap samples
    low = lower percentile
    high = upper percentile
    dist = random number generator that will give n samples
    statistic = boootstrap summary of interest"""
    
    bs = np.random.choice(xs, (nsamples, len(xs)), replace=True)
    bss = np.apply_along_axis(statistic, 1, bs)
    bss.sort()
    return np.percentile(bss, 100*low),  np.percentile(bss, 100*high)
```


```python
nsamples = 500
low = 0.025
high = 0.975
dist = partial(np.random.normal, 10, 3)
statistic = np.mean
n =1000
xs = dist(n)
print(bootstrap(xs, nsamples, low, high, statistic))

# 对于一个泊松分布的方差（variance of a poisson disttribution）找到自助抽样法的 CI 是很容易的（译者注：我不知道这个CI是什么鬼哈）
dist = partial(np.random.poisson, 5)
xs = dist(n)
print(bootstrap(xs, nsamples, low, high, np.var))

# 函数创作也能实现目的，比如找到精确度（precision）等等
print(bootstrap(xs, nsamples, low, high, compose(np.reciprocal, np.var)))
```

    (9.7001821758518894, 10.085722964842198)
    (4.3044644000000094, 5.1235598749999891)
    (0.19309519672067721, 0.23122277631375732)


### 装饰器（Decorators）

装饰器是一种特殊的函数，接收一个函数作为变量（argument），然后对这个函数进行"装饰（decorates）"，来增加新的功能，然后返回这个函数。


```python
# 一个简单的装饰器
def func_timer(f):
    import time
    def func(x):
        start = time.time()        
        result = f(x)
        print('Elapsed time:', time.time() - start)
        return result
    return func

# 手动调用
def snooze1(n):
    import time
    time.sleep(n)
    
func_timer(snooze1)(1)

# 使用装饰器的语法糖~
@func_timer
def snooze2(n):
    import time
    time.sleep(n)
    
snooze2(2)
```

    ('Elapsed time:', 1.0001499652862549)
    ('Elapsed time:', 2.001127004623413)


### 使用递归（recursion）


```python
# 经典的递归样例要么是斐波那契数列（Fibonacci series），要么就是阶乘函数（factorial function）
# 递归（Recursion）的意思其实就是一个函数调用自身而已

def fact(n):
    if n <= 1:
        return 1
    else:
        return n*fact(n-1)

def fib(n):
    if n < 2:
        return 1
    else:
        return fib(n-1) + fib(n-2)
```


```python
fact(6)
```




    720




```python
[fib(i) for i in range(10)]
```




    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]



### 考虑移到优化那部分（Consider moving to optimization session）


```python
# 在 Python 中，递归速度很慢，而且还很容易崩溃，因为函数调用次数太多了。
# 然而，如果咱们把之前函数调用计算的结果保存下来用于下次计算，而不是每次都完全重复计算过程
# 这样就显著地能降低内存使用以及计算要消耗的时间了
```


```python
# 下面的代码源自 http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
# 只是为了好玩哈（Just for fun）
# 展示了一种经典版本的装饰器（decorator）

def memoize1(f):
    store = {}
    def func(n):
        if n not in store:
            store[n] = f(n)
        return store[n]
    return func

@memoize1
def cfib(n):
    return fib(n)

def memoize2(f):
  class memodict(dict):
      __slots__ = ()
      def __missing__(self, key):
          self[key] = ret = f(key)
          return ret
  return memodict().__getitem__
    
@memoize2
def mfib(n):
    return fib(n)
```


```python
%timeit -n 3 fib(30)
%timeit -n 10 cfib(30)
%timeit -n 10 mfib(30)
```

    3 loops, best of 3: 655 ms per loop
    10 loops, best of 3: 620 ns per loop
    10 loops, best of 3: 95.4 ns per loop



```python

```
