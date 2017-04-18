
上手使用 Python 和 IPython notebook
====



IPython Notebook 是一个交互式的网络访问的 Python 环境，用户可以在这里将自己的代码、文本说明、图片等等整合成一篇文档。、本课程的所有讲座内容都用这个工具来制作。在本节，咱们就来简单介绍一下 IPython Notebook 的交互界面，并且展示一些常用的功能。

# 单元组成 Cells

IPython notebook 有两种组成单元: 
    
    * Markdown 文本
    * Code 代码
    
Markdown 主要就是用来输入文本的，不过也可以输入各种数学公式，而 Code 代码部分就是用来输入 Python 代码，并且能够调用很多其他的包和编译器等等。

## Markdown

选择一个单元的 tab ，设置类型为 'Markdown' 就可以了。


```python
from IPython.display import Image
```


```python
Image(filename='screenshot.png')
```




![png](output_7_0.png)



按照上面的图所示，当前的单元就是 Markdown 模式了，然后输入的所有内容都会被当做 Markdown 文本来处理。例如可以设置文本为 *斜体italics* 或者 **加粗**.  还可以像下面这样设置带小星号的列表：

Bulleted List
* Item 1
* Item 2

Markdown 有很多功能，详细的内容可以参考下面这个链接：
http://daringfireball.net/projects/markdown/syntax

## 代码单元 Code Cells

代码单元就是使用符合 Python 语法的文本来作为输入内容。稍后在我们对 Python 的简介中，就会看到很多这种例子了。眼下咱们先主要看看关于代码单元的一些额外用处。

### 各种神奇命令（Magic Commands）

Magic commands 内含很多很多功能，比如调用操作系统的命令行工具等等，实际上一般也就是这些功能。可以用下面这个命令来获取全部可用的 Magic Command：


```python
%lsmagic
```




    Available line magics:
    %alias  %alias_magic  %autocall  %automagic  %autosave  %bookmark  %cat  %cd  %clear  %colors  %config  %connect_info  %cp  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %lf  %lk  %ll  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %lx  %macro  %magic  %man  %matplotlib  %mkdir  %more  %mv  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd  %pprint  %precision  %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %rep  %rerun  %reset  %reset_selective  %rm  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



这里要注意，magic 命令有两类，一类是 line magics，一类是 cell magics。前者将一整行作为参数，而后者是将一个单元作为参数。'automagic'设置为 on，就可以随时加上百分号 % 来调用 line magics。


```python
%ls

```

    IPythonNotebookIntroduction.ipynb  Untitled0.ipynb  [0m[01;35mscreenshot.png[0m
    IntroductionToPython.ipynb         [01;34mbinstar[0m/



```python
%cp IntroductionToPython.ipynb IP2.ipynb

```


```python
%ls
```

    IP2.ipynb                          IntroductionToPython.ipynb  [0m[01;34mbinstar[0m/
    IPythonNotebookIntroduction.ipynb  Untitled0.ipynb             [01;35mscreenshot.png[0m



```python
%rm IP2.ipynb
```


```python
%ls

```

    IPythonNotebookIntroduction.ipynb  Untitled0.ipynb  [0m[01;35mscreenshot.png[0m
    IntroductionToPython.ipynb         [01;34mbinstar[0m/


上面的系统命令调用分了好几行，这挺麻烦的。其实也可以在一个单元内来解决，这时候就要使用 cell magic，就是在开头加上 %%system


```python
%%system
cp IntroductionToPython.ipynb  IP2.ipynb
ls
rm IP2.ipynb
ls
```




    ['IP2.ipynb',
     'IPythonNotebookIntroduction.ipynb',
     'IntroductionToPython.ipynb',
     'Untitled0.ipynb',
     'binstar',
     'screenshot.png',
     'IPythonNotebookIntroduction.ipynb',
     'IntroductionToPython.ipynb',
     'Untitled0.ipynb',
     'binstar',
     'screenshot.png']



不过还要注意，这些 magics 实际上并不仅仅能进行系统命令调用。
在 IPython notebook 中可以通过安装 rpy2 包来使用 R 语言：
（译者注：这个我没能成功安装，不知道怎么回事，还没来得及解决，不过还好不用 R语言。如果有朋友知道这个的解决方案，欢迎留言来指点，提前表示感谢！）
```bash
pip install rpy2
```


```python
!pip install rpy2
```

    Searching for rpy2
    Reading https://pypi.python.org/simple/rpy2/
    Downloading https://pypi.python.org/packages/d9/ca/c53301591b5d1e2fdeb49f2e4256ff24b185a1fc1b9711c15a73f0f61741/rpy2-2.8.5.tar.gz#md5=b1c3ef432b3a5c83cec06658eeb85581
    Best match: rpy2 2.8.5
    Processing rpy2-2.8.5.tar.gz
    Writing /tmp/easy_install-62a37j1p/rpy2-2.8.5/setup.cfg
    Running rpy2-2.8.5/setup.py -q bdist_egg --dist-dir /tmp/easy_install-62a37j1p/rpy2-2.8.5/egg-dist-tmp-ebz7gvig
    Error: Tried to guess R's HOME but no command 'R' in the PATH.
    error: Setup script exited with 1



```python
%load_ext rpy2.ipython 
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-23-a69f80d0128e> in <module>()
    ----> 1 get_ipython().magic('load_ext rpy2.ipython')
    

    /root/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py in magic(self, arg_s)
       2156         magic_name, _, magic_arg_s = arg_s.partition(' ')
       2157         magic_name = magic_name.lstrip(prefilter.ESC_MAGIC)
    -> 2158         return self.run_line_magic(magic_name, magic_arg_s)
       2159 
       2160     #-------------------------------------------------------------------------


    /root/anaconda3/lib/python3.5/site-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line)
       2077                 kwargs['local_ns'] = sys._getframe(stack_depth).f_locals
       2078             with self.builtin_trap:
    -> 2079                 result = fn(*args,**kwargs)
       2080             return result
       2081 


    <decorator-gen-62> in load_ext(self, module_str)


    /root/anaconda3/lib/python3.5/site-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
        186     # but it's overkill for just that one bit of state.
        187     def magic_deco(arg):
    --> 188         call = lambda f, *a, **k: f(*a, **k)
        189 
        190         if callable(arg):


    /root/anaconda3/lib/python3.5/site-packages/IPython/core/magics/extension.py in load_ext(self, module_str)
         35         if not module_str:
         36             raise UsageError('Missing module name.')
    ---> 37         res = self.shell.extension_manager.load_extension(module_str)
         38 
         39         if res == 'already loaded':


    /root/anaconda3/lib/python3.5/site-packages/IPython/core/extensions.py in load_extension(self, module_str)
         81             if module_str not in sys.modules:
         82                 with prepended_to_syspath(self.ipython_extension_dir):
    ---> 83                     __import__(module_str)
         84             mod = sys.modules[module_str]
         85             if self._call_load_ipython_extension(mod):


    ImportError: No module named 'rpy2'



```python
%matplotlib inline
```


```python
%%R
library(lattice) 
attach(mtcars)

# scatterplot matrix 
splom(mtcars[c(1,3,4,5,6)], main="MTCARS Data")
```

Matlab 也能调用，只要安装 pymatbridge：
```bash
pip install pymatbridge
```


```python
import pymatbridge as pymat
ip = get_ipython()
pymat.load_ipython_extension(ip)
```


```python
%%matlab

xgv = -1.5:0.1:1.5;
ygv = -3:0.1:3;
[X,Y] = ndgrid(xgv,ygv);
V = exp(-(X.^2 + Y.^2));
surf(X,Y,V)
title('Gridded Data Set', 'fontweight','b');
```

当然了，如果你更喜欢用 Octave，还可以安装 oct2py:
```bash
pip install oct2py
```


```python
%load_ext octavemagic
```


```python
%%octave

A = reshape(1:4,2,2)'; 
b = [36; 88];
A\b
[L,U,P] = lu(A)
[Q,R] = qr(A)
[V,D] = eig(A)
```

### 然后咱们重新尝试一下 Python 当中的例子


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm 
from pandas.tools.plotting import scatter_matrix
```


```python
# 首先是加载 mtcars 数据集，然后生成一个散点矩阵 scatterplot matrix

mtcars = sm.datasets.get_rdataset('mtcars')
df = pd.DataFrame(mtcars.data)
scatter_matrix(df[[0,2,3,4,5]], alpha=0.3, figsize=(8, 8), diagonal='kde', marker='o');
```


```python
# 接下来创建 3D 网格 3D mesh

xgv = np.arange(-1.5, 1.5, 0.1)
ygv = np.arange(-3, 3, 0.1)
[X,Y] = np.meshgrid(xgv, ygv)
V = np.exp(-(X**2 + Y**2))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0.25)
plt.title('Gridded Data Set');
```


```python
# 最后就是对矩阵进行各种操作 matrix manipulations

import scipy

A = np.reshape(np.arange(1, 5), (2,2))
b = np.array([36, 88])
ans = scipy.linalg.solve(A, b)
P, L, U = scipy.linalg.lu(A)
Q, R = scipy.linalg.qr(A)
D, V = scipy.linalg.eig(A)
print 'ans =\n', ans, '\n'
print 'L =\n', L, '\n'
print "U =\n", U, '\n'
print "P = \nPermutation Matrix\n", P, '\n'
print 'Q =\n', Q, '\n'
print "R =\n", R, '\n'
print 'V =\n', V, '\n'
print "D =\nDiagonal matrix\n", np.diag(abs(D)), '\n'
```

### 使用 Julia


```python
%load_ext julia.magic
```


```python
%%julia
1 + sin(3)
```


```python
%%julia
s = 0.0
for n = 1:2:10000
    s += 1/n - 1/(n+1)
end
s # an expression on the last line (if it doesn't end with ";") is printed as "Out"
```


```python
%%julia
f(x) = x + 1
f([1,1,2,3,5,8])
```

### 使用 Perl


```python
%%perl

use strict;
use warnings;
 
print "Hello World!\n";
```

我们希望上面的这些样例能让你对使用 IPython Notebook 环境所提供的强大功能和灵活特性有所帮助！
