话题覆盖范围
========================================
统计计算和计算机统计学（Statistical computing and computational statistics）
----------------------------------------

什么是统计计算（statistical computing）?

- 理解用于统计计算交互的一些基本计算方法
    - 数据结构和经典算法Data structures and classical algorithms
    - 效率和权衡妥协
    - 数值分析算法

什么是计算机统计学（computational statistics）?

- 利用计算机来生成适用于计算界面的统计实验（Use of the computer to conduct statistical experiments for computational inference）
    - 从模型到数据
    - 从数据抽象到模型
    - 用模型来作为数据生成过程


统计计算（computational statistics）和计算机统计学（statistical computing）都各自有什么工具？

- 优化，Optimization
- 模拟，Simulation
- 函数逼近和平滑，Function approximation and smoothing
- 结构发现，Finding structure
- 依赖关系建模，Modeling dependencies
- 统计可视化，Statistical visualization

数值分析
----------------------------------------

- 数值计算
    - 计算机中的数值表示，Representation
    - 数值稳定性，Stability
    - 条件判断，Conditioning
- 线性代数
    - 距离量度和范数
    - 向量空间和内积
    - 矩阵操作，行、列、空白（row, column and nul space）
    - 矩阵分解

数据结构和算法
----------------------------------------

- 数据结构
    - 列表 List
    - 元组 Tuple
    - 字典 Dictionary 和 集合 set
    - 堆 Heap
	- 树 Trees
- 算法
    - 复杂度和大O标识符
    - 算法设计
        - 贪婪法搜索 Greedy search
        - 局部搜索 Local search
        - 动态规划和缓存 Dynamic programming and caching
        - 逼近和嵌入 Approximation and embedding
    - 查找和排序
    - 组合算法
    - 图和网络

在工作流程中使用 Git 进行版本控制
----------------------------------------

- 5 分钟快速指南
    - 创建一个空的 Github 项目（repository）
    - 把这个 repo 克隆岛本地硬盘
    - `git config --global user.name "My name"`
    - `git config --global user.email "My email"`
    - 创建按两个新文件
    - `git add`
    - `git commit`
    - `git push`
    - 在 Github 上面检查
    - 创建一个分支（branch）
    - `git checkout foo`
    - 在分支中进行一些修改
    - 删除分支
    - `git branch -d foo`
    - 创建分支
    - `git checkout bar`
    - 在分支中进行一些更改
    - `git add`
    - `git commit`
    - 合并分支到 master
    - `git checkout master`
    - `git merge master`
    - 创建分支
    - `git checkout baz`
    - 在分支中进行一些更改
    - 临时保存更改
    - `git stash`
    - 转换到主分支
    - `git checkout master`
    - 在主分支中进行一些修改
    - 恢复到对分支 baz 进行操作的状态
    - `git checkout baz`
    - `git stash pop`
- 在启动 IPython notebook 的时候加上--script 参数来进行版本控制


测试
----------------------------------------

- 使用 pytest 来写单元测试
- 使用 IPython 来写单元测试<https://github.com/zonca/pytest-ipynb>
- 使用 TravisCI 进行持续集成

调试
----------------------------------------

- 在 IPython 中使用 pdb
- 使用 GUI 开发环境 (比如 PyCharm)

使用虚拟环境以实现可重现分析
----------------------------------------

参考 [虚拟环境指南](http://astropy.readthedocs.org/en/latest/development/workflow/virtual_pythons.html)

常规文档
----------------------------------------

- 在 IPython 内使用 [章节编号](https://github.com/dpsanders/ipython_extensions/tree/master/section_numbering) 以及 [ipyBibtex](https://gist.github.com/z-m-k/6080008)
- 使用`nbconvert`
- 使用 PythonTeX

打包并发布 Python 模块
----------------------------------------

- 使用 binstar
- 使用 PyPI

使用 Python
----------------------------------------

- 基础知识
    - 数据类型和数据集
    - 循环
    - 编写自定义函数
- 函数式编程
    - 函数首先是类的对象
    - 高优先级函数
    - 迭代器和生成器
	- 推导（列表理解），Comprehensions
    - 偏函数应用(Partial Application)和函数加里化(Currying)
    - 递归
    - 习语 Idioms - 底图 map, 筛选 filter, 折叠？fold
	- 函数库
	- 解析器
	- 倒计时
- 用 Python 做生产力工具
	- 用 IPython notebook 快速开发原型
	    - 文学编程，Literate programming
        - 内联可视化，Inline visualization
        - 对多种工具的快捷访问
            - 命令行
            - 编辑器
            - 其他编程语言
    - 用 Python 作为胶水语言
		- 获取和修缮数据
		- Python 和 LaTeX - PythonTeX
        - 与 R, Matlab 以及 Julia 交互
        - 用户界面开发
    - 从 Python 到 C
    - 单线程到并行，乃至大规模并行
- 与数据科学相关的 Python
    - IPython
	    - 使用内核 kernal 和多个客户端
        - [Jupyter](http://jupyter.org/)
		- 使用 `%qtconsole`
		- [通过 JavaScript 来自定义配置 IPython](https://github.com/ipython-contrib/IPython-notebook-extensions/wiki)
    - PythonTeX
    - 命令行工具
    - 获取数据
        - 使用关系型数据库
            - SQLite
            - [数据集 dataset](http://dataset.readthedocs.org/en/latest/)
			- [PonyORM](http://ponyorm.com/)
        - 使用分层数据库 (HDF5)
        - 处理 Excel 和 CSV
        - 处理 JSON 和 XML
        - 处理远程数据
    - 修整数据
        - 文字处理
        - 正则表达式，Regex
        - 使用 Pandas
	- Virtualenv
- 与统计学相关的 Python 内容
    - Numpy
        - Ufuncs
        - 向量化 vectorization (EM example)
        - tensordot
        - einsum
    - Scipy
    - Scikits-learn, Scikits-monaco, Scikits-bootstrap, Scikits-image
    - Pandas
    - Patsy
    - StatsModels
    - SymPy
    - PyStan, PyMC, emcee
- 使用 Python 对数据进行可视化
    - Matplotlib
	- plt.rcdefaults() after e.g. xkcd()
	- 或者使用
	```
	with plt.xkcd():
        fig1 = plt.figure()
    ```
	- Seaborn
		- seaborn.reset_orig() 来重设 Matplotlib 的默认值
	- Vispy（译者注，这个项目很久不更新了，现在推荐 GlumPy）
	- ggplot
	- Bokeh
	- 通过 IPython 块来展示矩阵乘法
	- 通过 IPython Notebook 的控件和交互功能来进行演示
	- 通过 [Asymptote](https://github.com/jrjohansson/ipython-asymptote) 达到出版要求的高质量图像
- 用 Python 进行高性能计算
	- [用 Joblib 进行缓存以及实现并行任务](https://pythonhosted.org/joblib/)
    - Cython, bottleneck, CythonGSL
    - 多线程处理，Multiprocessing
    - NumExpr, Numba 以及 NumbaPro
    - Theano
    - MPI4py
- 用 Python 做胶水语言
    - 结合 R, Matlab, Julia

优化
----------------------------------------

- 标准方法
    - 最小残差法，Minimize residuals
    - 最大或然法，Maximum likelihood
    - 矩量动差法，Method of moments
    - 期望最大化，[x] Expectation-maximization (EM)
- 方法
    - Nelder-Mead
    - 一阶方法，First-order methods
		- 梯度下降法，[x]Gradient descent
		- 随机梯度下降法，Stochastic gradient descent
    - 二阶方法，Second-order methods
        - 牛顿法，Newton
        - 高斯-牛顿法，Gauss-Newton
        - IRLS
        - 拟牛顿法，Quasi-Newton
        - 共轭梯度法，Conjugate gradient
        - Levenberg-Marquadt
    - 全局方法，Global methods
        - 模拟退火法，Simulated annealing
        - Basin hopping

模拟
----------------------------------------

- 随机数的生成
- 数据分隔
- 统计学上的支持度计算？Bootstrap and Jackknife
- 置换取样
- 蒙特卡罗方法
    - 拒绝性采样，Rejection sampling
    - 重要性采样，Importance sampling
- 马尔科夫链蒙特卡罗方法，Markov chain Monte Carlo
    - 温度模拟？Simulated tempering
    - MCMCMC：Metropolis-coupled Markov chain Monte Carlo？
- 微粒过滤器，Particle filters
- 模拟实验
    - 实验设计
        - 随机采样
        - 拉丁超立方体抽样，Latin Hypercube sampling
        - 正交化，Orthogonal
	- 文档和可重复分析

近似函数（Approximating functions）
----------------------------------------

- 函数之间的距离
- 求积分，Quadrature
    - 梯形法和辛普森法，Trapezoid and Simpsons
    - 牛顿-柯提斯求积法，Newton-Cotes
    - 自适应数值积分，Adaptive quadrature
- 参数函数的一般类型，General families of parametric functions
- 核心方法，Kernel methods
- 正交基函数，Orthogonal basis functions
    - 多项式，Polynomials
    - 傅里叶，Fourier
    - 厄米特，Hermite
- 贝叶斯非参数模型，Bayesian non-parametric models

结构（Structure）
----------------------------------------

- Linear structure
    -  PCA
    -  ICA
    -  Factor analysis
    -  Latent semantic analysis
- Class discovery
    - Distance between clusters
    - Bottom-up
    - Top-down
    - K-means
- Mixture models
    - Parametric mixture models
    - Non-parametric mixture models
- Graphical structure

依赖关系
----------------------------------------

- Transformations
- Generalized models
    - Linear regression
    - Logistic regression
- Hierarchical structure
- Sequential structure

数据可视化
----------------------------------------

- Conditional plots
- Scatter matrix
- Image map
- Brushing and linked displays
- Parallel coordinates and Andrews curves
- Dimension reduction
    - Multi-dimensional scaling (MDS)
- Projection pursuit

关键算法
----------------------------------------

- GD and SGD (1st order)
- Newton type methods (2nd order)
- EM
- PCA, ICA, Factor anaalyis, LSI
- Manifold learning - MDS
- Monte Carlo methods - bootstrap, permutation resampling, cross-validation
- MCMC
- Particle filters
- Smoothing concepts
- Kernel density estimation
- Viterbi algorihtm as example of dynamic progrramming
- Time series - convolution, smoothing, FFT, wavelets
- Matrix decomposition
- Pairwise metrics - clustering
