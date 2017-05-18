
# Python 简介

本章节内容基本上是参考了下面这个编程指南: http://www.afterhoursprogramming.com/tutorial/Python/Introduction/

本课程中相当一大部分内容都需要使用 Python。Python 是一门高级的脚本语言，提供了一个交互式的开发环境。 我们假设大家已经有了一定的编程经验，所以本章节主要讲的也就是 Python 的一些特性。

编程语言一般都要有下面这些常见的组成部分：变量、运算、迭代、条件判断、函数（内置函数或者用户自定义函数），以及一些更高级的数据结构。本章我们关注的就是 Python 中的这些内容，然后尤其会着重介绍 Python 这门语言独有的一些特性。

## 变量

Pythong 中的变量在你设定一个值给他的时候就被定义和给定类型了。


```python
my_variable = 2 
print(my_variable)
type(my_variable)

```

    2





    int



这就让变量定义对开发者来说变得很简单。不过当然了，所谓能力越大责任越大，还是要多小心仔细。比如下面这个情况：


```python
my_varible = my_variable+1
print (my_variable)
```

    2


如果你缺了单词，拼写检查是不能帮你补上的。"If you leave out word, spell-check will not put the word in you" -- Taylor Mali, <em>The the impotence of proofreading</em>

如果你偶然地弄错了一个变量的类型，Python 也不会帮你记着之前的正确类型。这种情况容易引起一些特别难发现问题原因的 bug，所以一定要谨慎小心处理变量类型。

### 类型和转换 Types and Typecasting

Python 中可以进行各种常见的类型转换，所以可以把字符串转换成整形或者浮点数，还可以吧浮点数转换成整数等等。语法上面和 C 语言稍微不太一样：


```python
a = "1"
b = 5 
print(a+b)

```


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-bf1e73fdfc72> in <module>()
          1 a="1"
          2 b=5
    ----> 3 print(a+b)
    

    TypeError: cannot concatenate 'str' and 'int' objects



```python
a = "1"
b = 5
print(int(a)+b)
```

    6


一定要注意类型是 <em>动态变化的</em>。也就是说，只要重新赋值一下，就可以把一个变量立即从整形编程其他类型，比如浮点数或者字符串等等都可以。


```python
a = "1"
type(a)
print(type(a))

a = 1.0
print(type(a))
```

    <type 'str'>
    <type 'float'>


Python 还有一些其他的特别的数据类型，比如列表、元组和字典等等，后面会提到的。

## 运算 Operators

Python 支持常见的各种运算，比如 +,-,/,\*,=,>,<,==,!=,&,|, 
(求和, 做差, 相除, 乘积, 赋值, 大于, 小于, 双等号表示判断相等关系, 等号前面加叹号表示不等于, 并且（表示两个条件都要满足） , 或者（表示两个条件满足其一即可).  
此外还有%,// 以及 ** (相除取余数, 地板除法，以及乘方运算')。 注意一些细节:


```python
print(3/4)
print(3.0 / 4.0)
print(3%4)
print(3//4)
print(3**4)
```

    0
    0.75
    3
    0
    81


注意一下这里对整形值的除法！这个运算和其他强类型语言比如 C/C++ 很相似。整数除法的运算结果和地板除法 // 是一样的。所以如果你要想得到一个浮点数的结果，那用于进行除法运算的参数也都必须是浮点数（或者你要转换他们成浮点数）。


```python
a = 3
b = 4
print(a/b)
print(float(a)/float(b))
```

    0
    0.75


## 循环

Python 支持各种常见的循环，while , for, 以及其他的一些结构等等，后面再细说。下面是一些样例：


```python
for i in range(1,10):
     print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9


这里一定要注意，在上面的这个循环里面，range 函数给出的值不包含右端的设定终点的值。也就是 range(1,10) 其实是从 1 到 9 ，以此类推，range(m,m+n)其实也就是从 m 到 m+n-1，而不是到 m+n。


```python
i = 1
while i < 10:
    print(i)
    i+=1
```

    1
    2
    3
    4
    5
    6
    7
    8
    9


这个 while 循环 就简单的不得了了，没啥可额外讲解的。

## 条件语句


```python
a = 20
if a >= 22:
   print("if")
elif a >= 21:
    print("elif")
else:
    print("else")
```

    else


这里也很简单，没啥可说的，就注意一下语法格式就行了。另外这里我们要说下空格。Python对缩进有严格的要求：在每个条件分支语句之后，必须新起一行（上面的循环其实也是这样），然后每一个条件语句都要和同层次的条件语句有一样的缩进，这些缩进的空格数必须都是一样的。


```python
a = 23
if a >= 22:
   print("if")
    print("greater than or equal 22")
elif a >= 21:
    print("elif")
else:
    print("else")
```


    IndentationError: unexpected indent




```python
a = 23
if a >= 22:
   print("if")
   print("greater than or equal 22")
elif a >= 21:
    print("elif")
else:
    print("else")
```

    if
    greater than or equal 22


一般情况大家都是用四个空格作为一层缩进，当然你也可以自己随便设置自己喜欢的。但一定要保持某一种风格，不要乱改，要不然你前后缩进千奇百怪就会各种报错各种狗带，你说悲催不？

#### 异常处理 Exceptions

Python 还有另一个功能，也是关于条件处理的，这个功能超级有用。加入你的程序正在处理用户输入的数据或者从文件读取数据。你不一定能保证每次程序接收到的输入是什么，这有时候就会导致一些问题。所以这时候就可以用 'try/except' 语句来搞定这个问题：



```python
a = "1"

try:
  b = a + 2 
except:
  print(a, " is not a number") 

```

    ('1', ' is not a number')


上面这个代码里面，就是尝试给字符串 a 加上一个数值。这就会产生一个异常，不过我们已经捕获了这个异常了，希望通知用户这个问题产生的原因。这个就比下面这种程序莫名其妙反正就是崩溃要好得多：


```python
a = "1"
b = a + 2 

```


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-49-1b1ef016a6b2> in <module>()
          1 a = "1"
    ----> 2 b = a + 2
    

    TypeError: cannot concatenate 'str' and 'int' objects


## 函数


```python
def Division(a, b):
    print(a/b)
Division(3,4)
Division(3.0,4.0)
Division(3,4.0)
Division(3.0,4)
```

    0
    0.75
    0.75
    0.75


要注意刚刚上面那个函数并没有指定变量类型，所以上面输入的几种类型都可以。不过这个是好用却也挺危险的。比如下面这个情况：



```python
def Division(a, b):
    print(a/b)
Division(2,"2")
```


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-56-663c15047357> in <module>()
          1 def Division(a, b):
          2     print(a/b)
    ----> 3 Division(2,"2")
    

    <ipython-input-56-663c15047357> in Division(a, b)
          1 def Division(a, b):
    ----> 2     print(a/b)
          3 Division(2,"2")


    TypeError: unsupported operand type(s) for /: 'int' and 'str'


在固定类型的编程语言中，程序员就必须要指定每个变量的类型，比如上面例子中的 a 和 b，要指定他们是浮点数还是整数等等，然后编译器在编译的时候就会提示说传入函数的变量类型不正确。在 Python 里，就不会这样，所以我们需要使用刚刚提到的 'try/except' 语句来处理异常。


```python
def Division(a, b):
    try:
        print(a/b)
    except:
        if b == 0:
           print("cannot divide by zero")
        else:
           print(float(a)/float(b))
Division(2,"2")
Division(2,0)
```

    1.0
    cannot divide by zero


# 字符串和字符串的处理

Python 最重要的一个特征就是对字符串的处理，既功能强大，有非常简单。在 Python 里面，定义字符串就和其他大多属于盐一样简单。但 Python 更厉害的是对字符串的查找、替换、大小写转换、缩减以及读取字符串元素。本小节只会涉及其中的一小部分。更多完整的与字符串相关的 Python 功能，可以参考 http://www.tutorialspoint.com/python/python_strings.htm


```python
a = "A string of characters, with newline \n CAPITALS, etc."
print(a)
b=5.0
newstring = a + "\n We can format strings for printing %.2f"
print(newstring %b)

```

    A string of characters, with newline 
     CAPITALS, etc.
    A string of characters, with newline 
     CAPITALS, etc.
     We can format strings for printing 5.00


接下来试试其他的字符串操作吧：


```python
a = "ABC DEFG"
print(a[1:3])
print(a[0:5])
```

    BC
    ABC D


上面这写代码中有好几处需要学习。首先， Python 给字符串设定了一个索引。其次，就是这个索引是从 0 开始数的。最后，右侧的值依然是不包含终点。（a[0:5] 包含了元素从 0 到4，不包含 5，也就是 0,1,2,3,4，可以理解成一个左闭右开区间。


```python
a = "ABC defg"
print(a.lower())
print(a.upper())
print(a.find('d'))
print(a.replace('de','a'))
print(a)
b = a.replace('def','aaa')
print(b)
b = b.replace('a','c')
print(b)
b.count('c')

```

    abc defg
    ABC DEFG
    4
    ABC afg
    ABC defg
    ABC aaag
    ABC cccg





    3



是不是挺有意思的？用 Python 还能对字符串进行哪些操作呢？你可以发挥想象力然后动手试试，挺有意思的！

# 列表 Lists, 元组 Tuples, 字典 Dictionaries

## 列表 Lists

列表 list 这个就跟他的名字的意思一样，就是一个列元素组成的表。他是由对象组成的列表。作为列表的元素，这些对象可以是任何的数据类型（也包括列表本身），而且在列表中是可以使用各种不同的元素类型来作为列表元素的。所以这就比数组要灵活多了。对列表还可以进行很多操作，比如增加元素、删除元素、插入元素，以及查元素个数和排序反转等等。


```python
a_list = [1,2,3,"this is a string",5.3]
b_list = ["A","B","F","G","d","x","c",Alist,3]
print(b_list)
print(b_list[7:9])
```

    ['A', 'B', 'F', 'G', 'd', 'x', 'c', [1, 2, 3, 'this is a string', 5.3], 3]
    [[1, 2, 3, 'this is a string', 5.3], 3]



```python
a = [1,2,3,4,5,6,7]
a.insert(0,0)
print(a)
a.append(8)
print(a)
a.reverse()
print(a)
a.sort()
print(a)
a.pop()
print(a)
a.remove(3)
print(a)
a.remove(a[4])
print(a)
```

    [0, 1, 2, 3, 4, 5, 6, 7]
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    [8, 7, 6, 5, 4, 3, 2, 1, 0]
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    [0, 1, 2, 3, 4, 5, 6, 7]
    [0, 1, 2, 4, 5, 6, 7]
    [0, 1, 2, 4, 6, 7]


跟字符串一样，列表元素的索引也是从 0 开始的。（译者注：实际上这个在大多数编程语言里都是这样，数组、字符串、列表的索引都是从0开始，所以有人说程序员查数都是从 0 开始数。

列表可以用在 for 语句或者其他的一些条件判断语句中。这些也叫做列表表达式。例如下面这个：


```python
even_numbers = [x for x in range(100) if x % 2 == 0]
print(even_numbers)
```

    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]


实际上列表表达式也可以用于字符串：


```python
first_sentence = "It was a dark and stormy night."
characters = [x for x in first_sentence]
print(characters)
```

    ['I', 't', ' ', 'w', 'a', 's', ' ', 'a', ' ', 'd', 'a', 'r', 'k', ' ', 'a', 'n', 'd', ' ', 's', 't', 'o', 'r', 'm', 'y', ' ', 'n', 'i', 'g', 'h', 't', '.']


关于列表表达式的更多内容可以参考官方文档:  https://docs.python.org/3/tutorial/datastructures.html?highlight=comprehensions

另外一个类似的功能是 'map' 映射。Map 是对一个列表进行一个函数操作。语法是map(aFunction, aSequence)。例如下面的例子：



```python
def sqr(x): return x ** 2
a = [2,3,4]
b = [10,5,3]
c = map(sqr,a)
print(c)
d = map(pow,a,b)
print(d)
```

    [4, 9, 16]
    [1024, 243, 64]


要注意，用 map 通常要比使用列表表达式或者循环语句要更高效。

### 元组 Tuples

Python 的元组与列表类似，不同之处在于元组的元素不能修改。而且，元组使用小括号，列表使用方括号。


```python
a = (1,2,3,4)
print(a)
a[1] = 2
```


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-94-3f7ee3924a64> in <module>()
          1 a=(1,2,3,4)
          2 print(a)
    ----> 3 a[1]=2
    

    TypeError: 'tuple' object does not support item assignment


    (1, 2, 3, 4)



```python
a = (1,"string in a tuple",5.3)
b = (a,1,2,3)
print(a)
print(b)

```

    (1, 'string in a tuple', 5.3)
    ((1, 'string in a tuple', 5.3), 1, 2, 3)


看，上面的代码就很明显了把，所有列表的灵活性都还保留了，又不会被改变。所以当你不希望一个列表被修改的时候，就可以使用元组了。

元组有另一个吊炸天的功能，就是 元组拆包（'tuple unpacking'）。这个可以用来把一个列表中的一系列值作为值来赋值给另一序列的变量名，例如下面的例子：


```python
my_pets = ("Chestnut", "Tibbs", "Dash", "Bast")
(aussie,b_collie,indoor_cat,outdoor_cat) = my_pets
print(aussie)
cats=(indoor_cat,outdoor_cat)
print(cats)

```

    Chestnut
    ('Dash', 'Bast')


## 字典 Dictionaries

字典本来是无序的，但是自从 Python 3.6 开始，字典也是有序的了。本来字典可以当做无序的由键和值组成的。在字典中，可以用键来进行索引。


```python
a = ["A","B","C","D"] #list example
print(a[1])

```

    B



```python
a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"} # dictionary example
print(a[1])
```


    ---------------------------------------------------------------------------
    KeyError                                  Traceback (most recent call last)

    <ipython-input-3-d314b6c47c6e> in <module>()
          1 a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"} # dictionary example
    ----> 2 print(a[1])
    

    KeyError: 1



```python
a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"} # dictionary example
print(a['anItem'])
print(a)
```

    A
    {'athirdItem': 'C', 'afourthItem': 'D', 'anItem': 'A', 'anotherItem': 'B'}

在 3.6 之前的 Python 中，字典不具有排序性，所以不对字典元素进行排列，所以你不能按照某个顺序来读取（也就是不能顺序来索引），所以早先的版本中你只能靠 key（键）来读取元素。
### 集合 Sets

集合是无序的，对一系列'不重复'的元素的集合。集合支持的运算有交集、并集以及集合对比（Intersections, unions and set differences ）。所以可以用集合来移除一个数据集中的重复元素，或者测试某个元素是否属于某个集合。例如下面的代码：


```python
from sets import Set
fruits = Set(["apples","oranges","grapes","bananas"])
citrus = Set(["lemons","oranges","limes","grapefruits","clementines"])
citrus_in_fruits = fruits & citrus   #intersection
print(citrus_in_fruits)
diff_fruits = fruits - citrus        # set difference
print(diff_fruits)
diff_fruits_reverse = citrus - fruits  # set difference
print(diff_fruits_reverse)
citrus_or_fruits = citrus | fruits     # set union
print(citrus_or_fruits)
```

    Set(['oranges'])
    Set(['apples', 'grapes', 'bananas'])
    Set(['grapefruits', 'clementines', 'lemons', 'limes'])
    Set(['clementines', 'grapes', 'limes', 'oranges', 'grapefruits', 'apples', 'lemons', 'bananas'])



```python
a_list = ["a", "a","a", "b",1,2,3,"d",1]
print(a_list)
a_set = Set(a_list)  # Convert list to set
print(a_set)         # Creates a set with unique elements
new_list = list(a_set) # Convert set to list
print(new_list)        # Obtain a list with unique elements 
```

    ['a', 'a', 'a', 'b', 1, 2, 3, 'd', 1]
    Set(['a', 1, 2, 'b', 'd', 3])
    ['a', 1, 2, 'b', 'd', 3]


更多样例以及详细的关于集合的内容可以参考官方文档：https://docs.python.org/3/library/sets.html

类 Classes
----

类（或者类的对象）是对一系列数据（作为类的属性 attribute ）和函数（作为类的方法 method ）的打包。可以用'.'这个点号来读取类中的属性或者使用类的方法。例如，当我们调用`'hello'.upper()`的时候，就是用到了`'hello'`这个属于字符串 `string`类的实例的 `upper` 方法。

本章节不会涉及关于自定义累的创建等更多内容。

模块 Modules
----

随着代码规模越来越大，就应该把代码拆分成多个 *module* 模块或者包，这样组织起来也更方便。最简单的情况下，模块可以就是普通的 python 文件。我们常常要先用 `import` 导入模块，然后在后面的代码中就能使用这些模块中的内容：

```python
import numpy
import numpy as np # 这是设置用 np 作为别名
import numpy.linalg as la # 模块下面还可以有子模块
from numpy import sin, cos, tan # 这里是把三角函数的名字引入到了全局的命名空间，这样就不用加模块名前缀了
from numpy import * # 尽量别这么做，因为这样可能就污染了命名空间
```

标准库 The standard library
----

Python 已经自带了很赞的一些标准库，其中有各种各样的函数都包含在了标准库中。

**参考**

- [Python 官方标准库文档](https://docs.python.org/3/library/)
- [Python 模块示例](http://pymotw.com/3/contents.html) 给出了各种对标准库进行使用的样例。

### 安装额外的模块

多数情况下，咱们都要用`pip`这个包管理器来安装或者卸载模块。实际上，只要输入命令就可以了：

```bash
pip install <要安装的包的名字>
```
在命令行中或者在 IPython notebook 中输入：
```python
! pip install <要安装的包的名字>
```

有哪些包能用 `pip`安装呢？可以参考[Python Package Index (PyPI)](https://pypi.python.org/pypi)。

Pip 的官方文档在 <https://pip.pypa.io/en/latest/>.

更新 Anaconda 发行版到最新的版本
----

在命令行中输入下面的命令就行了：
```bash
conda update conda
conda update anaconda
```

这里要注意， `conda` 能做的事情还有 [很多很多哦](http://conda.pydata.org/docs/index.html).

<font color=red>练习 Exercises</font>
----

**1**. 解决 FizzBuzz 问题

写一个程序，输出从 1 到 100 的数字。但当数字是 3 的倍数的时候，就不输出数字，而是输出'Fizz';当数字是 5 的倍数的时候，就不输出数字，而是输出'Buzz'；当数字同时是 3 和 5 的倍数的时候，就不输出数字，而是输出'FizzBuzz';



```python
# YOUR CODE HERE



```

**2**. 给定 x=3 y=4，交换两个变量的值


```python
x = 3
y = 4
# YOUR CODE HERE



```

**3**. 写一个程序，在平面直角坐标系中计算并返回两个点的距离，假设两个点为 $u$ 和 $v$, 这两个都是一个含有两个元素的元组，各自都有自己的坐标值 $(x, y)$。例如, 假如 $u = (3,0)$ 然后 $v = (0,4)$， 函数就会计算并返回这两个点的距离 $5$.


```python
# YOUR CODE HERE



```

使用字典，来写一个程序，计算每个字母在给定字符串 s 中出现的次数。忽略大小写。例如 'a' 和 'A' 就都当作同样的一个键。下面的这个 s 中， a 就似乎是出现了 7 次。


```python
s = """
Write a program that prints the numbers from 1 to 100. 
But for multiples of three print 'Fizz' instead of the number and f
or the multiples of five print 'Buzz'. For numbers which are 
multiples of both three and five print 'FizzBuzz'
"""

# YOUR CODE HERE



```

**5**. Write a program that finds the percentage of sliding windows of length 5 for the sentence s that contain at least one 'a'. Ignore case, spaces and punctuation. For example, the first sliding window is 'write' which contains 0 'a's, and the second is 'ritea' which contains 1 'a'.

写一个程序，在一段句子中，找出长度为 5 的子串且至少含有一个字母 'a' 的比例。忽略大小写、空格以及标点符号。例如下面这个字符串 s 中， 'write' 就包含了 0 个 'a'，第二个就是'ritea' 包含了一个 'a'。（这个我翻译得很差，所以提供了原文作为对比。）


```python
s = """
Write a program that prints the numbers from 1 to 100. 
But for multiples of three print 'Fizz' instead of the number and f
or the multiples of five print 'Buzz'. For numbers which are 
multiples of both three and five print 'FizzBuzz'
"""

# YOUR CODE HERE



```

**6**. 从下面的列表中查找只出现过一次的数字：


```python
x = [36, 45, 58, 3, 74, 96, 64, 45, 31, 10, 24, 19, 33, 86, 99, 18, 63, 70, 85,
 85, 63, 47, 56, 42, 70, 84, 88, 55, 20, 54, 8, 56, 51, 79, 81, 57, 37, 91,
 1, 84, 84, 36, 66, 9, 89, 50, 42, 91, 50, 95, 90, 98, 39, 16, 82, 31, 92, 41,
 45, 30, 66, 70, 34, 85, 94, 5, 3, 36, 72, 91, 84, 34, 87, 75, 53, 51, 20, 89, 51, 20]

# YOUR CODE HERE



```

**7**. 写两个函数，一个返回一个数值的平方，另一返回立方。然后利用这两个函数再来写第三个函数，来求一个数值的 $6^{th}$ ，即六次方。


```python
# YOUR CODE HERE

def square(x):
    pass

def cube(x):
    pass

def f(x):
    pass
```

**8**. 创建一个新列表，列表内容是 x for x in [0, 10] 的立方，分别使用以下方法：

- for 循环
- 列表表达式推导
- map 函数


```python
# YOUR CODE HERE



```

**9**. 毕达哥拉斯三角（Pythagorean triple），也就是中国的勾股定理啦，$a^2 + b^2 = c^2$，比如第一组就是(3,4,5)，写个程序找出来 100 以内的所有符合勾股定理的三角形组合吧。


```python
# YOUR CODE HERE



```

**10**. 解决下面这个函数的 bug，这个函数式接收一个数值列表，然后返回归一化的列表（a list of normalized numbers）。

```python
def f(xs):
    """Retrun normalized list summing to 1."""
    s = 0
    for x in xs:
        s += x
    return [x/s for x in xs]
```


```python
# YOUR CODE HERE



```


```python

```
