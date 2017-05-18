

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




    '%.4f'



函数 首先是类的对象/第一等的对象（Functions are first class objects）
----
在 Python 里面， 函数就和其他的对象一样，比如int整数或者list列表之类的。这就意味着你可以把函数用作变量（argument）传递给其他函数，还可以把函数存成字典的键值，或者把函数作为返回值从另一个函数来返回。总之这样用函数就能有很多非常神奇的使用方法了。


(译者注：这篇文章和整个系列，其实都是基于 Python2的，但是我从最开始学 Python，就用的是3，然后为了及时翻译出来，我并没有把这些代码完全都改写成能用在 Python3 里面的，不过估计用 2to3 之类的工具大概是能转换的，等全部翻译完了之后，我才能有精力来进行转换，当然也欢迎广大朋友们来参与进来了。)


```python
def square(x):
    """求 x 的平方"""
    return x*x

def cube(x):
    """求 x 的立方"""
    return x*x*x
```


```python
# 创建一个以函数为键值的字典（dictionary）

funcs = {
    'square': square,
    'cube': cube,
}
```


```python
x = 2

print square(x)
print cube(x)

for func in sorted(funcs):
    print func, funcs[func](x)
```


      File "<ipython-input-12-3d24de83b66f>", line 3
        print square(x)
                   ^
    SyntaxError: invalid syntax



高级函数（Higher-order functions）
----

如果一个函数能使用其他函数作为输入的变量（argument）或者返回一个函数作为返回值，那么这个函数就是一个高级函数（higher-order function，简写 HOF）。一般用得最多的就是 `map` 和 `filter`。


```python
# map 函数会对一个集合(collection) 里面的全部成员使用某个函数来进行操作
#（译者注：在深度学习中，map和reduce好像是译作映射与规约，这里的map似乎也能从字面上这么理解，将每个元素都用一个函数映射到了一个新的集合中）

map(square, range(5))
```


```python
# filter 函数会对一个集合中的每个成员进行一个判断（predciate） 
# 然后只保存判断的值为真（True）的成员（译者注：所以顾名思义，就是类似过滤器，基本也就是英文单词filter的原意）

def is_even(x):
    return x%2 == 0

filter(is_even, range(5))
```


```python
# 通常 map 和 reduce 都是联合起来使用的 - 也就是很有名的"map-reduce" 结构（construct）
#（译者注：不知道这里是不是搞错了，原文说的就是 map-reduce，但是这里分明用的是 map-filter）

map(square, filter(is_even, range(5)))
```


```python
# The reduce function reduces a collection using a binary operator to combine items two at a time
# reduce 函数每次对两个元素进行一种 二值化/二进制？运算 ，然后依次应用到整个集合。
#（译者注：上面这句我把握不好怎么表达，所以带了英文原文）

def my_add(x, y):
    return x + y

# 下面这就是用 reduce 来实现的另外一种累加函数
reduce(my_add, [1,2,3,4,5])
```


```python
# 自定义函数当然也可以以函数作为返回值，这样就也是高级函数（HOF）了

def custom_sum(xs, transform):
    """Returns the sum of xs after a user specified transform."""
    return sum(map(transform, xs))

xs = range(5)
print custom_sum(xs, square)
print custom_sum(xs, cube)
```


```python
# 返回一个函数是很有用的

# 一个闭包？（closure）
#（译者注：我水平很初级，所以不太理解闭包的概念，回头我多学习一下）

def make_logger(target):
    def logger(data):
        with open(target, 'a') as f:
            f.write(data + '\n')
    return logger

foo_logger = make_logger('foo.txt')
foo_logger('Hello')
foo_logger('World')
```


```python
!cat 'foo.txt'
```

纯函数
----
什么样的函数是纯函数呢？纯函数不能有任何*附带效果*，也不能依赖全局变量。纯函数就和数学意义上的函数差不多，每次给定相同的输入，也得返回相同的输出。这对于降低bug很有帮助，在并行开发中也是如此，因为每个函数的调用都是彼此独立的，也就是可并行的（parallelizable）



```python
def pure(xs):
    """创建一个新的列表，然后用之作为返回值"""
    xs = [x*2 for x in xs]
    return xs
```


```python
xs = range(5)
print "xs =", xs
print pure(xs)
print "xs =", xs
```


```python
def impure(xs):
    for i, x in enumerate(xs):
        xs[i] = x*2
    return xs
```


```python
xs = range(5)
print "xs =", xs
print impure(xs)
print "xs =", xs
```


```python
# 要记住，可变函数（mutatble functions）是在函数声明的时候创建的，而不是使用的时候
# 下面这个就是初学者常犯的一种错误：

def f1(x, y=[]):
    """记住，一定不要用空白列表或者类似的其他可变数据结构来作为默认值。"""
    y.append(x)
    return sum(y)
```


```python
print f1(10)
print f1(10)
print f1(10, y =[1,2])
```


```python
# 下面是正确的符合 Python 风格的示例：

def f2(x, y=None):
    """检查一下y是不否空的，如果是，就创建一个列表来赋值给y"""
    if y is None:
        y = []
    y.append(x)
    return sum(y)
```


```python
print f1(10)
print f1(10)
print f1(10, y =[1,2])
```

递归
----

递归函数就是那种调用自身的函数。
递归函数在分治法算法的样例中非常有用，另外也是一种有限差分方程（finite difference equations）的表现形式。
不过递归函数在计算上效率很低很差，所以在 Python 里面通常很少使用递归函数。

Recursvie functions generally have a set of  where the answer is obvious and can be returned immediately, and a set of recursive cases which are split into smaller pieces, each of which is given to the same function called recursively. A few examples will make this clearer.

通常递归函数都有一系列的*基准条件（base cases）*，而其答案（answer）通常都是很简单的那种能立即返回的，一系列的递归条件就被分成了一个个小块，然后其中的某一个传递给相同的函数来进行递归调用。下面来几个例子说明一下就更清楚了。


```python
# 阶乘函数函数（factorial function）可能是经典递归样例中最简单的一个了。

def fact(n):
    """返回 n 的阶乘"""
    # base case
    if n==0:
        return 1
    # recursive case
    else:
        return n * fact(n-1)

print [fact(n) for n in range(10)]
```


```python
# 斐波那契数列（Fibonacci sequence）是另外一个经典的递归样例。

def fib1(n):
    """使用递归给出斐波那契数列"""

    # 基准条件 base case
    if n==0 or n==1:
        return 1
    # 递归条件 recurssive case
    else:
        return fib1(n-1) + fib1(n-2)

print [fib1(i) for i in range(10)]
```


```python
# 在 Python 语言中个，还可以使用另外一种方法实现 斐波那契数列，不用递归，而且更有效率：

def fib2(n):
    """不用递归实现的斐波那契数列"""
    a, b = 0, 1
    for i in range(1, n+1):
        a, b = b, a+b
    return b

print [fib2(i) for i in range(10)]
```


```python
# 一定要注意，使用了递归的版本速度要比不用递归的慢很多

%timeit fib1(20)
%timeit fib2(20)

# 这是因为递归产生了多次函数调用
# 注意下面的运行顺序中对 fib(2)和 fib(1)的多次调用
# fib(4) -> fib(3), fib(2)
# fib(3) -> fib(2), fib(1)
# fib(2) -> fib(1), fib(0)
# fib(1) -> 1
# fib(0) -> 1
```


```python
# 当然可以用递归来演示分治法（dividde-and-conquer paradigm）

def almost_quick_sort(xs):
    """差不多能算是一个快速排列"""

    # 基准条件 base case
    if xs == []:
        return xs
    # 递归条件 recursive case
    else:
        pivot = xs[0]
        less_than = [x for x in xs[1:] if x <= pivot]
        more_than = [x for x in xs[1:] if x > pivot]
        return almost_quick_sort(less_than) + [pivot] + almost_quick_sort(more_than)

xs = [3,1,4,1,5,9,2,6,5,3,5,9]
print almost_quick_sort(xs)
```

迭代器（Iterators）
----


迭代器表示了值的流（streams of values）。因为每次用到的只有一个值，所以都一下子存入内存会耗费很多空间。用迭代器就可以帮助解决这类问题，比如数据集太大不能一下子放入RAM里面，就可以用迭代器来帮忙。


```python
# 使用 iter() 这个 Python 的内置函数（built-in function） 就可以从一个有序的数据结构创建迭代器

xs = [1,2,3]
x_iter = iter(xs)

print x_iter.next()
print x_iter.next()
print x_iter.next()
print x_iter.next()
```


```python
# 大多数情况里，迭代器都在一个 for 循环里面被（自发地）使用了。
# 当遇到了停止迭代警告（StopIteration exception）的时候就会停止迭代

x_iter = iter(xs)
for x in x_iter:
    print x
```

生成器（Generators）
----

生成器创建迭代器的流（iterator streams）。


```python
# 包含有'yield'这个关键词（keyword）的函数会返回迭代器（iterators）
# 在完成了 yielding 之后，这个函数还会恢复到之前的状态

def count_down(n):
    for i in range(n, 0, -1):
        yield i
```


```python
counter = count_down(10)
print counter.next()
print counter.next()
for count in counter:
    print count,
```


```python
# 还可以使用'生成器表达式（generator expressions）'来生成迭代器
# 其形式有点像“列表生成器list generators”
# 不过不同的是生成器表达式用的是圆括号，而列表生成器使用的是方括号。
# （译者注：这个很好记，列表就有方括号，所以列表生成器语句中要用的是方括号，这就好记忆一点不容易弄混了。

xs1 = [x*x for x in range(5)]
print xs1

xs2 = (x*x for x in range(5))
print xs2

for x in xs2:
    print x,
print
```


```python
# 迭代器可以用在无穷函数（infinite functions）上

def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b
```


```python
for i in fib():
    # 生成器返回了一个无穷的流，所以必须停止条件判断
        if i > 1000:
        break
    print i,
```


```python
# 很多 Python 内置函数都返回迭代器
# 例如 处理文件的 file handlers
# 所以通过下面的示范（idiom）你就可以逐行地处理文件，哪怕有 1 TB（terabyte）那么大
# 这样即便用性能比较一般的笔记本都可以，不会遇到什么问题
# 在 Pyhton 3 里面 ，map 和 filter 返回的都是迭代器，而不再是列表了

for line in open('foo.txt'):
    print(line)
```

匿名函数（Anonymous functions）
----

进行函数式编程的是，时候，经常会有这种需求，就是创建一些非常小的函数，来完成一些很细小的任务，比如作为高级函数（map 或者 filter 之类的 HOF）的输入变量。在这类情况下，通常就可以用匿名函数或者 lambda 函数。如果你觉得很难理解一个 lambda 函数是用来干嘛的，就可以把它改写的成普通的常规函数的形式。


```python
# 使用常规函数

def square(x):
    return x*x

print map(square, range(5))
```


```python
# 用匿名函数

print map(lambda x: x*x, range(5))
```

装饰器
----

装饰器是一种高级函数，接受一个函数为参数，然后返回一个添加了额外性质的打包的新函数

例子：

- logging
- profiling
- 即时编译，Just-In-Time (JIT) compilation


```python
# 下面是一个简单的装饰器，用来对一个任意函数进行计时

def func_timer(func):
    """程序消耗的时间"""
    
    def f(*args, **kwargs):
        import time
        start = time.time()
        results = func(*args, **kwargs)
        print "Elapsed: %.2fs" % (time.time() - start)
        return results
    
    return f
```


```python
# 装饰器函数有一个特别的简短记号：（ a special shorthand notation for decorating functions）


@func_timer
def sleepy(msg, sleep=1.0):
    """响应之前沉睡给定的时间"""
    import time
    time.sleep(sleep)
    print msg

sleepy("Hello", 1.5)
```

运算器模块 `operator`
----

`operator`模块提供了各种常规 Python 运算符（+, *, []等等）的”函数版本”，可以用到使用函数作为变量（argument）的情景


```python
import operator as op

# 先买就是求和函数的另一种表述方式
print reduce(op.add, range(10))
print reduce(op.mul, range(10))
```


```python
my_list = [('a', 1), ('b', 4), ('c', 2), ('d', 3)]

# 标准排序
print(sorted(my_list))

# 根据每个元素的位置1 来进行排列，然后返回该列表 (注意，Python 是从0开始计数，1就是第二个位置了。)
print(sorted(my_list, key=op.itemgetter(1)))
```

函数工具模块 `functools` 
----

函数工具模块 `functools`中个最常用的函数可能就是 `partial`，这个函数可以用来基于旧函数和一些额外给定的新的参数（arguments "filled-in"），来生成一个新的函数。


```python
from functools import partial

sum_ = partial(reduce, op.add)
prod_ = partial(reduce, op.mul)
print(sum_([1,2,3,4]))
print(prod_([1,2,3,4]))
```

迭代器模块 `itertools`
----

这个模块提供了很多用于处理迭代器的核心函数，对于函数式编程来说非常重要。


```python
from itertools import cycle, groupby, islice, permutations, combinations

print list(islice(cycle('abcd'), 0, 10))
print 

animals = sorted(['pig', 'cow', 'giraffe', 'elephant', 'dog', 'cat', 'hippo', 'lion', 'tiger'], key=len)
for k, g in groupby(animals, key=len):
    print k, list(g)
print

print [''.join(p) for p in permutations('abc')]
print 

print [list(c) for c in combinations([1,2,3,4], r=2)]
```

其他的几个模块 `toolz`, `fn` 和 `funcy`
----

如果你希望使用函数式编程风格，还需要了解一下下面这几个模块：

- [toolz](https://github.com/pytoolz/toolz)
- [fn](https://github.com/kachayev/fn.py)
- [funcy](https://github.com/Suor/funcy)


```python
# 下面是一个小例子，利用细菌酶（译者注：原文这里就是 bacterial enzyme）的DNA转换成对应的蛋白质序列
# 译者注：这里感觉明显是错的，酶通常就是催化作用，除非是特殊的，常规的酶就是蛋白或者肽链哪来的DNA，我记得染色体和线粒体才有吧
# 另外DNA转换的应该是氨基酸序列吧，怎么能是蛋白质的序列呢？应该是氨基酸序列对应的肽链再组合才能成蛋白质吧？如果我记错了请更正。
# 我的一位青年科学家朋友说：觉得原文的意思应该是，这个细菌酶所对应的DNA，然后将其转换成氨基酸序列。当然连成肽链和折叠成蛋白质的工作这里就不考虑了。
# 上面的内容我觉得很有参考意义，值得借鉴，所以分享给大家！


codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

gene = """
>ENA|BAE76126|BAE76126.1 Escherichia coli str. K-12 substr. W3110 beta-D-galactosidase 
ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCT
GGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGC
GAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGC
TTTGCCTGGTTTCCGGCACCAGAAGCGGTGCCGGAAAGCTGGCTGGAGTGCGATCTTCCT
GAGGCCGATACTGTCGTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATC
TACACCAACGTGACCTATCCCATTACGGTCAATCCGCCGTTTGTTCCCACGGAGAATCCG
ACGGGTTGTTACTCGCTCACATTTAATGTTGATGAAAGCTGGCTACAGGAAGGCCAGACG
CGAATTATTTTTGATGGCGTTAACTCGGCGTTTCATCTGTGGTGCAACGGGCGCTGGGTC
GGTTACGGCCAGGACAGTCGTTTGCCGTCTGAATTTGACCTGAGCGCATTTTTACGCGCC
GGAGAAAACCGCCTCGCGGTGATGGTGCTGCGCTGGAGTGACGGCAGTTATCTGGAAGAT
CAGGATATGTGGCGGATGAGCGGCATTTTCCGTGACGTCTCGTTGCTGCATAAACCGACT
ACACAAATCAGCGATTTCCATGTTGCCACTCGCTTTAATGATGATTTCAGCCGCGCTGTA
CTGGAGGCTGAAGTTCAGATGTGCGGCGAGTTGCGTGACTACCTACGGGTAACAGTTTCT
TTATGGCAGGGTGAAACGCAGGTCGCCAGCGGCACCGCGCCTTTCGGCGGTGAAATTATC
GATGAGCGTGGTGGTTATGCCGATCGCGTCACACTACGTCTGAACGTCGAAAACCCGAAA
CTGTGGAGCGCCGAAATCCCGAATCTCTATCGTGCGGTGGTTGAACTGCACACCGCCGAC
GGCACGCTGATTGAAGCAGAAGCCTGCGATGTCGGTTTCCGCGAGGTGCGGATTGAAAAT
GGTCTGCTGCTGCTGAACGGCAAGCCGTTGCTGATTCGAGGCGTTAACCGTCACGAGCAT
CATCCTCTGCATGGTCAGGTCATGGATGAGCAGACGATGGTGCAGGATATCCTGCTGATG
AAGCAGAACAACTTTAACGCCGTGCGCTGTTCGCATTATCCGAACCATCCGCTGTGGTAC
ACGCTGTGCGACCGCTACGGCCTGTATGTGGTGGATGAAGCCAATATTGAAACCCACGGC
ATGGTGCCAATGAATCGTCTGACCGATGATCCGCGCTGGCTACCGGCGATGAGCGAACGC
GTAACGCGAATGGTGCAGCGCGATCGTAATCACCCGAGTGTGATCATCTGGTCGCTGGGG
AATGAATCAGGCCACGGCGCTAATCACGACGCGCTGTATCGCTGGATCAAATCTGTCGAT
CCTTCCCGCCCGGTGCAGTATGAAGGCGGCGGAGCCGACACCACGGCCACCGATATTATT
TGCCCGATGTACGCGCGCGTGGATGAAGACCAGCCCTTCCCGGCTGTGCCGAAATGGTCC
ATCAAAAAATGGCTTTCGCTACCTGGAGAGACGCGCCCGCTGATCCTTTGCGAATACGCC
CACGCGATGGGTAACAGTCTTGGCGGTTTCGCTAAATACTGGCAGGCGTTTCGTCAGTAT
CCCCGTTTACAGGGCGGCTTCGTCTGGGACTGGGTGGATCAGTCGCTGATTAAATATGAT
GAAAACGGCAACCCGTGGTCGGCTTACGGCGGTGATTTTGGCGATACGCCGAACGATCGC
CAGTTCTGTATGAACGGTCTGGTCTTTGCCGACCGCACGCCGCATCCAGCGCTGACGGAA
GCAAAACACCAGCAGCAGTTTTTCCAGTTCCGTTTATCCGGGCAAACCATCGAAGTGACC
AGCGAATACCTGTTCCGTCATAGCGATAACGAGCTCCTGCACTGGATGGTGGCGCTGGAT
GGTAAGCCGCTGGCAAGCGGTGAAGTGCCTCTGGATGTCGCTCCACAAGGTAAACAGTTG
ATTGAACTGCCTGAACTACCGCAGCCGGAGAGCGCCGGGCAACTCTGGCTCACAGTACGC
GTAGTGCAACCGAACGCGACCGCATGGTCAGAAGCCGGGCACATCAGCGCCTGGCAGCAG
TGGCGTCTGGCGGAAAACCTCAGTGTGACGCTCCCCGCCGCGTCCCACGCCATCCCGCAT
CTGACCACCAGCGAAATGGATTTTTGCATCGAGCTGGGTAATAAGCGTTGGCAATTTAAC
CGCCAGTCAGGCTTTCTTTCACAGATGTGGATTGGCGATAAAAAACAACTGCTGACGCCG
CTGCGCGATCAGTTCACCCGTGCACCGCTGGATAACGACATTGGCGTAAGTGAAGCGACC
CGCATTGACCCTAACGCCTGGGTCGAACGCTGGAAGGCGGCGGGCCATTACCAGGCCGAA
GCAGCGTTGTTGCAGTGCACGGCAGATACACTTGCTGATGCGGTGCTGATTACGACCGCT
CACGCGTGGCAGCATCAGGGGAAAACCTTATTTATCAGCCGGAAAACCTACCGGATTGAT
GGTAGTGGTCAAATGGCGATTACCGTTGATGTTGAAGTGGCGAGCGATACACCGCATCCG
GCGCGGATTGGCCTGAACTGCCAGCTGGCGCAGGTAGCAGAGCGGGTAAACTGGCTCGGA
TTAGGGCCGCAAGAAAACTATCCCGACCGCCTTACTGCCGCCTGTTTTGACCGCTGGGAT
CTGCCATTGTCAGACATGTATACCCCGTACGTCTTCCCGAGCGAAAACGGTCTGCGCTGC
GGGACGCGCGAATTGAATTATGGCCCACACCAGTGGCGCGGCGACTTCCAGTTCAACATC
AGCCGCTACAGTCAACAGCAACTGATGGAAACCAGCCATCGCCATCTGCTGCACGCGGAA
GAAGGCACATGGCTGAATATCGACGGTTTCCATATGGGGATTGGTGGCGACGACTCCTGG
AGCCCGTCAGTATCGGCGGAATTCCAGCTGAGCGCCGGTCGCTACCATTACCAGTTGGTC
TGGTGTCAAAAATAA
"""
from toolz import partition

# 把上面的 FASTA 转成一个单个的 DNA 序列
dna = ''.join(line for line in gene.strip().split('\n') 
              if not line.startswith('>'))

# 把 DNA 切割成密码子（codons）长度为 3， 然后转换密码子成为氨基酸（amino acid）
codons = (''.join(c) for c in partition(3, dna))
''.join(codon_table[codon] for codon in codons)
```

这个 `partition` 函数还可以用于序列窗口的统计（statistics on sequence windows）
例如,计算一个移动的平均线（calculating a moving average）

<font color=red>练习 Exercises</font>
----

**1**. 把下面的网状循环改写成列表推导式（list comprehension）

```python
ans = []
for i in range(3):
    for j in range(4):
        ans.append((i, j))
print ans
```


```python
# YOUR CODE HERE



```

**2**. 把下面的列表改写成列表推导式

```python
ans = map(lambda x: x*x, filter(lambda x: x%2 == 0, range(5)))
print ans
```


```python
# YOUR CODE HERE



```

**3**. 把下面的函数改写成纯函数，不要依赖全局变量，也不要有任何附带效果


```python
x = 5
def f(alist):
    for i in range(x):
        alist.append(i)
    return alist

alist = [1,2,3]
ans = f(alist)
print ans
print alist # alist has been changed!
```


```python
# YOUR CODE HERE



```

**4.** 写一个装饰器`hello` ，使所有装饰过的函数都用 print 输出 "Hello!" 

For example

```python
@hello
def square(x):
    return x*x
```

调用的时候给出类似下面的结果：
```python
[In]
square(2)
[Out]
Hello!
4
```


```python
# YOUR CODE HERE



```

**5**. 重写一个阶乘函数，不要用递归

```python
def fact(n):
    """Returns the factorial of n."""
    # base case
    if n==0:
        return 1
    # recursive case
    else:
        return n * fact(n-1)
```


```python
# YOUR CODE HERE



```

**6**. 那下面的匿名函数重写成一个常规的命名的函数

```python
lambda x, y: x**2 + y**2
```


```python
# YOUR CODE HERE



```
