Title: Outline of STA663
Date: 2017-04-16
Category: Duke
Tags: Translation,Lesson,Programming,Python

Duke 大学 STA 663 课程教学大纲
========================================
----------------------------------------
# 课程信息
----------------------------------------
授课教师: [Cliburn Chan](https://medschool.duke.edu/about-us/our-faculty/chi-wei-cliburn-chan)
邮箱：<cliburn.chan@duke.edu>
全名：Chi Wei Cliburn Chan
职位：Duke 大学 生物统计学和生物信息学 副教授
（译者注：看上去好像是一位华裔，至少能确定应该是亚裔，看来亚裔在计算机相关的学术领域有很多高人了，李开复、张亚勤、吴恩达，以及这位我还不知道中文名的老兄。）
![](https://medschool.duke.edu/sites/medschool.duke.edu/files/styles/profile/public/i5063472.jpg?itok=cE0Z7UWQ)


授课教师: [Janice McCarthy](https://scholars.duke.edu/display/per0747252)
邮箱：<janice.mccarthy@duke.edu>
职位：Duke 大学 生物统计学和生物信息学 讲师
![](https://scholars.duke.edu/file/t0747252/thumb_image_0747252.jpg)

助教: Matt Johnson
邮箱：<mcj15@stat.duke.edu>
职位：Duke 大学 统计科学 博士生
![](https://media.licdn.com/mpr/mpr/shrinknp_400_400/AAEAAQAAAAAAAANTAAAAJGM2MDE3Y2ZkLTljMjAtNGQ5Mi05YTZlLTg0NjRhMTk1MTBkNg.jpg)


Github 上本课程相关的 Repo：
<https://github.com/cliburn/Computational-statistics-with-Python>
<https://github.com/cliburn/sta-663-2017>

[2015 版课程中的 Python 和 IPython 资源](http://people.duke.edu/~ccc14/sta-663-2015/general.html)

[2017 版课程中的 Python 和 IPython 资源](http://people.duke.edu/~ccc14/sta-663-2017)

----------------------------------------
# 课程简介
----------------------------------------
STA 663 这门课的目标是教会学生以统计学的方法来进行编程，也就是如何写代码来解决统计学的问题。一般来说，统计学上面的问题，都会接触到一些从数据而产生的各种特征估计，可以是点估计，也可以是对一个区间的估计，或者是对一个整个函数范围内的。通常情况下，解决这样的各种统计学问题，都需要写代码来对数据进行收集、整理、探究、分析以及呈现。很显然，咱们都希望能够写出优质代码，也就是可读性好，又功能正确而运行高效，而且最好充分利用已有的工具而不是“重新发明轮子”。

这门课程的覆盖内容比较广泛，包括一些高性能开发相关的思路（数据结构、算法，以及包括并行编程在内的代码优化）；还有一些用于数据分析的比较重要的数值计算方法（计算数值、矩阵分解、线性/非线性回归，数值优化、函数估计，以及蒙特卡罗方法）。我们假设学习者已经了解了一些基础的编程概念（函数、类、循环等等），而且有良好的开发习惯（literate programming：文学编程，强调程序的解释性和可读性；懂得测试以及版本控制），然后还需要有基础的数学和统计学背景知识（线性代数、微积分、概率论）。

要解决统计学问题，你一般需要满足下面这些条件：
1. 掌握对数据进行采集、组织、探索、呈现的基本技巧
2. 应用某种特定的数值方法来分析数据
3. 优化代码来提高运行速度（这个在应对“大数据”的时候尤其重要）

与之对应，STA663 这门课也分成了三个部分，基础知识部分占 20%，数值方法部分占 60%，高性能计算部分占 20%。

----------------------------------------
# 学习目标
----------------------------------------



这个课程主要介绍的内容是在**优化**和**模拟**这两方面的各种算法的开发，这些内容也是计算统计学的核心内容。针对统计学的**计算**是本课程的重点，也就是如何构建原型，继而优化，然后使用 Python 和 C/C++ 编程语言开发高性能计算的算法。各种各样的算法和数据集的复杂度都会逐渐增加，（一维 -> 多维，固定 -> 自适应，线性 -> 并行 -> 大规模并行，少量数据 -> 巨量数据），这是为了让学这门课的学生能够掌握并联系下面这些内容：

* 练习可重现的分析（reproducible analysis）
* 掌握基本的数据管理和处理技能
* 使用 Python 语言来进行统计计算
* 使用数学和统计学的链接库来提高效率
* 能够理解和优化线性代码（serial code）
* 能够掌握不同的并行开发范式并高效利用

----------------------------------------
# 要求事先掌握的知识和技能
----------------------------------------


下面这些内容你需要掌握，作为本课程的先决条件，如果不熟悉，赶紧补一下：

- Unix 系统的命令（译者注：实际上也就是 Linux + Bash 熟悉了就差不多）
- 使用 `git` 进行版本控制
- 能够写 Markdown
- 会使用 $\LaTeX$ 来编辑数学公式等等（译者注：这个我就不会，不过不用担心，用上的时候再学呗）
- 能通过 `make` 来使用源代码构建和安装软件



本课程会简单介绍一些 Python 相关的基础知识，但速度会特别快特别概略。所以除非你已经是一个很有经验的开发者，否则还是建议你好好学习一下 Python 的基本知识，推荐使用 ThinkPython 这本教材，[中文版在线阅读地址](http://www.gitbook.com/book/cycleuser/think-python)，[英文原版地址](http://www.greenteapress.com/thinkpython/html/index.html)，中英双语以及更多与该书相关的信息[可以参考这里](https://zhuanlan.zhihu.com/p/24644499)。这本书很重要，本课程的作业中的内容也用这本书作为参考了。

Python 官方也提供了一个 [Python 指南](https://docs.python.org/3/tutorial/)，也可以参考一下。

----------------------------------------
# 成绩和打分
----------------------------------------

- 机房上机作业(50%)
- 期中考试  (25%)
- 最终大作业  (25%)

----------------------------------------
# 计算平台
----------------------------------------


所有学生都可以使用本课程教学团队提供的一个虚拟机镜像，里面运行的是一个 Ubuntu 操作系统，下载链接会单独给选修该课程的学术。GPU 计算和 "Map（映射）"和"Reduce（归约)" 这些大数据相关的内容，推荐使用亚马逊云服务（AWS）平台。再次强调一下，具体如何获取和使用这些内容，会在适当的时候提供给选课的同学。（译者注：这只适用于 Duke 大学选修该课程的同学，我翻译过来之后，大家如果也需要这样的虚拟机镜像，回头我可以给大家来做一个放到百度云上面分享一下，如果大家有这个需求，请尽管提出来。）



本课程的所有相关代码，都要求提交到个人的 Github 库中，名字命名为 sta-663-名字-姓氏。此外还要把授课的老师和助教都添加成该项目的合作者，这样他们能够有完整权限读取你的代码。关于如何配置 Github 这部分内容，我们相信同学们有能力自己探索明白的。（译者注：如果探索不明白，那真的不太适合上这门课了。）


----------------------------------------
# 课程目录
----------------------------------------

Lecture 1
----------------------------------------

- 关于 IPython notebook 的基础内容
    - Markdown 基础知识
    - Code 基础知识
    - 显示系统
    - IPython 的一些魔法技巧
    - 与其他编程语言的交互

- Python 编程语言相关的基础内容
    - 基本类型
    - 基本的数据集类型 (元组 tuple , 列表 list , 集合 set , 字典 dict , 字符串string)
    - 控制流（分支循环选择等）
    - 函数
    - 类
    - 模块
    - 标准库
    - PyPI 以及包管理器 `pip`
    - 导入更多模块

上机实验课1
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab01/Exercises01.ipynb 这个文件。

Lecture 2
----------------------------------------

- 函数式编程
	- 函数首先是类的对象
	- 纯函数
    - 迭代器
    - 生成器
    - 使用 lambda 表达式的匿名函数
    - 递归
	- 装饰器
	- 运算符模块`operator`
	- 循环器模块 `itertools`
	- 函数工具模块`functools`
	- 工具模块 `toolz`
	- 构建一个懒惰的数据管道（lazy data pipeline）
- 处理文本
    - 字符串方法
    - 字符串模块`string`
    - 正则表达式模块`re`

Lecture 3
----------------------------------------

- 获取数据
    - CSV 文件`csv`
    - JSON 文件`json`
    - 通过 `scrapy` 进行网页抓取
    - HDF5 文件 `pyhdf`
    - 关系型数据库以及 SQL语句 `sqlite3`
    - 数据集模块 `datasets`
- 整理数据
	- 删除备注
	- 按行过滤
	- 按列过滤
	- 解决不连续
	- 处理缺失数据
	- 移除冗余信息
	- 导出信息
	- 完整性检查和可视化

上机实验课2
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab02/Exercises02.ipynb 这个文件。



Lecture 4
----------------------------------------

- 使用 `numpy`
	- 数据类型
    - 创建数组
    - 检索
	- 广播（Broadcasting）
		- 产出效果（Outer product）
    - 通用函数，Ufuncs（universal functions）
    - 生成通用函数
    - 使用 numpy 解决线性代数问题
        - 计算协方差矩阵
        - 最小二乘法线性回归
    - numpy 中的 I/O
- 使用`pandas`
	- 读写数据
	- 切分、应用、合并（Split-apply-combine）
	- 合并和拼接（Merging and joining）
	- 处理时间序列
- 使用`blaze`


上机实验课3
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab03/Exercises03.ipynb 这个文件。

Lecture 5
----------------------------------------

- 从数学到计算机
    - 各种数字在计算机上的表示法
    - Overflow 溢出, underflow 下溢出, catastrophic cancellation 灾难性取消
    - 稳定性
    - 条件判断
	- 符号到代码的直接转换是危险的
- 计算的目的是洞察和理解，而不是简单的数字结果
- 统计学中的计算样例
    - 参数估计 (点和区间估计)
    - 函数估计
    - 特征提取、类别发现和数据降维
    - 分类和回归
    - 模拟和计算推理
- 算法效率以及 大$\mathcal{O}$ 标识符
    - 经典数据结构和算法的范例

Lecture 6
----------------------------------------

- 数值线性代数
    - 线性方程组
    - 列空间、行空间和阶（Column space, row space and rank）
    - 阶、基、张成（Rank, basis, span）
    - 范数和距离（Norms and distance）
    - 迹和行列式（Trace and determinant）
    - 特征值和特征向量（Eigenvalues and eigenvectors）
    - 内积（Inner product）
    - 外积（Outer product）
    - 爱因斯坦求和符号（Einstein summation notation）
- 矩阵与线性变换
    - 矩阵类型
        - 方阵与非方阵（Square and non-square）
        - 奇异矩阵（Singular）
        - 正定矩阵（Positive definite）
        - 幂等矩阵和投影（Idempotent and projections）
        - 正交和标准化（Orthogonal and orthonormal）
        - 对称矩阵（Symmetric）
        - 矩阵变换（Transition）
    - 矩阵几何说明
 
上机实验课4
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab04/Exercises04.ipynb 这个文件。


Lecture 7
----------------------------------------

- 矩阵分解
    - LU 分解法(高斯消去法Gaussian elimination)
    - QR 分解法
    - Spectral 谱分析
    - SVD 矩阵奇异值分解法
    - Cholesky 分解法
- 使用`scipy.linalg`
- BLAS（基础线性代数子程序库 Basic Linear Algebra Subprograms）和 LAPACK（线性代数工具包，Linear Algebra PACKage，以 Fortran 开发的用于数值计算的函式集）

Lecture 8
----------------------------------------

- 投影、排序、坐标变换（Projections, ordination, change of coordinates）
    - 详解 PCA（主成分分析，principal component analysis)
    - 用特征分解（eigendecomposition）来实现 PCA
    - 用 SVD 矩阵奇异值分解法实现 PCA
   	- 相关的方法
 		- LSA（Latent semantic analysis，隐含语义分析）

（译者注：特征分解（Eigendecomposition），又称谱分解（Spectral decomposition）是将矩阵分解为由其特征值和特征向量表示的矩阵之积的方法。需要注意只适用于可对角化矩阵。）


上机实验课5
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab05/Exercises05.ipynb 这个文件。


Lecture 9
----------------------------------------

- 回归和最大似然法作为优化问题（Regression and maximum likelihood as optimization problems）
- 局部和全局极值（Local and global extrema）
- 单根查找和优化（Univariate root finding and optimization）
    - 黄金分割搜索（Golden section search）
    - 二分搜索（Bisection）
	- 牛顿-拉普森算法（Newton-Raphson）以及割线法（secant methods）
- 常规优化途径
	- 理解问题
	- 多次随机启动
	- 结合算法
	- 过程可视化

Lecture 10
----------------------------------------

- 优化实践（Practical optimization）
- 凸性、局部最优与约束（Convexity, local optima and constraints）
- 条件数（Condition number）
- 求根问题（Root finding）
- 单变量标量函数优化（Optimization for univariate scalar functions）
- 多变量标量函数优化（Optimization for multivariate scalar functions）
- 过程可视化
- 应用实例
    - 曲线拟合
    - 拟合方程
	- 实用弹簧模型进行图形显示（Graph dispaly using a spring model）
	- 多元逻辑回归分析法（Multivariate logistic regression）

上机实验课6
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab06/Exercises06.ipynb 这个文件。

Lecture 11
----------------------------------------

- 多变量和约束优化（Multivariate and Constrained optimization）
    - 向量积分复习（Review of vector calculus）
    - 向量内积（Innner products）
	- 共轭梯度（Conjugate gradients）
	- 牛顿法 (2nd order)
	- 拟牛顿算法 (1st order)
	- Powells and Nelder Mead (0th order)
    - 拉格朗日乘数法（Lagrange multipliers）

Lecture 12
----------------------------------------

- EM 算法
    - 凸凹函数
    - 詹森不等式（Jensen's inequality）
    - 缺失数据设置
    - 玩具案例-翻硬币与两枚偏见硬币（2 biased coins）
    - 高斯混合模型
	- 向量化代码示例

上机实验课7
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab07/Exercises07.ipynb 这个文件。

Lecture 13 - 期中考试
----------------------------------------

- 周一 1月 23号 4:40 - 6:25 (105 分钟)

Lecture 14
----------------------------------------

- 背景知识
    - 简介蒙特卡罗方法（Monte Carlo methods）
    - 常见的应用
    - 统计学中的应用
    - 蒙特卡罗优化
- 如何获得随机数
    -伪随机数生成器
        - 线性同余发生器（Linear congruential generators）
    - 从一个分布中获取另外一个分布
        - 反向变化
        - 拒绝采样方法
        - 混合表示
-  蒙特卡罗积分
    - 基本概念
    - 拟随机数字
- 蒙特卡罗诈骗

上机实验课8
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab08/Exercises08.ipynb 这个文件。

Lecture 15
----------------------------------------

- 重采样方法
    - Bootstrap 算法
	- Jackknife 折刀法
	- 转列采样（Permuation sampling）
	- 交叉验证（Cross-validation）
- 进行模拟实验(案例研究)
	- 实验设计
    - 研究的变量
    - 变量层次（阶乘，超立方）（Levels of variables (factorial, Latin hypercube)）
    - 代码文档
    - 结果记录
    - 汇报
    - 可重现分析，使用 `make`和 $\LaTeX$

Lecture 16
----------------------------------------

- MCMC (1)
	- 玩具温体 - 吃药的老鼠
    - 蒙特卡罗估计
    - 权重抽样
    - Metropolis-Hasting 采样
    - Gibbs 吉布斯采样
    - Hamiltonian 哈密顿采样
	- 收敛性评估
	- 使用 `pystan`
	- 使用 `pymc2`
	- 使用 `emcee`

上机实验课9
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab09/Exercises09.ipynb 这个文件。

Lecture 17
----------------------------------------

- MCMC算法（马尔可夫链蒙特卡罗(Markov Chain Monte Carlo)）(2)
    - 回顾高斯混合模型（Gaussian mixture model revisited）
    - 吉布斯采样（Gibbs sampling）
    - 狄利克雷过程的无限混合模型（Infinite mixture model with the Dirichlet process）

Lecture 18
----------------------------------------

- 分析（Profiling）
    - 过早的优化是万恶之源（Premature optimization is the root of all evil）
    - 使用 `%time` 和 `%timeit`
    - 使用`%prun` 进行分析
	- 行分析（Line profiler）
	- 内存分析（Memory profiler）
- 代码优化
    - 使用合适的数据结构
    - 使用恰当的算法
    - 使用 Python 语言风格
    - 使用优化过的模块
    - 缓存和记忆
    - 向量化和广播
    - 纵览全局（Views）
    - 跨过技巧

上机实验课10
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab10/Exercises10.ipynb 这个文件。

Lecture 19
----------------------------------------

- JIT（just in time） 编译，使用 `numba`
- 利用 `cython` 优化性能
- 包装利用 C 代码
- 包装利用 C++ 代码
- 包装利用 Fortran 代码

Lecture 20
----------------------------------------

- [为什么现代 CPU 性能依然不够用，以及该怎么解决这个问题](http://www.pytables.org/docs/CISE-12-2-ScientificPro.pdf)
- 并行编程模式
- Amdahl's and GustClassifying points with the Gustafson's laws
- 并行编程样例
    - JIT（just in time） 编译，使用 `numba`
	- 玩具样例 - 分型
    - 使用 `joblib`
    - 使用 `multiprocessing`
    - 使用 `IPython.Parallel`
    - 使用 `MPI4py`

上机实验课11
----------------------------------------

参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab11/Exercises11.ipynb 这个文件。

Lecture 21
----------------------------------------

- GPU 运算
	- 简介 CUDA
	- Vanilla 矩阵乘法
	- 共享内存的矩阵乘法
	- JIT（just in time） 编译，使用 `numba`
	- 样例: 利用 CUDA 的大尺度高斯模型（GMMs）

Lecture 22
----------------------------------------

-  "Map（映射）"和"Reduce（归约）" 以及 Spark ([AWS](https://aws.amazon.com/articles/4926593393724923))

    - 问题 - k-mer DNA序列计数
    - 使用 Python 进行小规模 "Map（映射）"和"Reduce（归约）"
    -  通过 `mrjob`使用`hadoop`
    - 通过`pyspark`使用`spark`
    - 使用`MLib` 进行大规模机器学习

上机实验课12
----------------------------------------


参考本课程 Github 项目中[上机实验目录下](https://github.com/Kivy-CN/Computational-statistics-with-Python-CN/tree/master/Labs/) Lab12/Exercises12.ipynb 这个文件。


参考资料
========================================

第一部分
----------------------------------------

- 使用命令后
    - Unix 哲学以及 `bash`
    - 使用 `ssh` 远程登录
    - 使用 `git` 进行版本控制
    - 利用 $\LaTeX$ 编写文档
    - 通过`make`实现自动化

第二部分
----------------------------------------

- Python 中的可视化
    - 使用 `matplotlib`
    - 使用  `seaborn`
    - 使用  `bokeh`
    - [使用  `daft`](http://daft-pgm.org/)

