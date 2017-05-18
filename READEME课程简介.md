Title: Introduction of STA663
Date: 2017-04-16
Category: Duke
Tags: Translation,Lesson,Programming,Python



基于 Python 的计算机统计学
=======

# 简介


这一套课件实际上是一些相当粗糙的讲座笔记的草稿，基于 IPython notebook ，这门课程从 2015 年春季开始的，针对硕士生的统计学课程，[课程地址原本在这里，但是这个链接实际上已经不能访问了](https://stat.duke.edu/courses/statistical-computing-and-computation)。译者注：现在的链接[可能应该是在这个链接](http://people.duke.edu/~ccc14/sta-663/index.html)。这个课程主要介绍的内容是在**优化**和**模拟**这两方面的各种算法的开发，这些内容也是计算统计学的核心内容。各种各样的算法和数据集的复杂度都会逐渐增加，（一维 -> 多维，固定 -> 自适应，线性 -> 并行 -> 大规模并行，少量数据 -> 巨量数据），这是为了让学这门课的学生能够掌握并联系下面这些内容：

* 练习可重现的分析（reproducible analysis）
* 掌握基本的数据管理和处理技能
* 使用 Python 语言来进行统计计算
* 使用数学和统计学的链接库来提高效率
* 能够理解和优化线性代码（serial code）
* 能够掌握不同的并行开发范式并高效利用

算法上，主要集中在以下两方面：

* 优化（Optimization）
    * 牛顿-拉普森算法（Newton-Raphson，NR算法），函数式编程和向量化
    * 积分算法（Quadrature），自适应方法
    * 梯度下降法（Gradient descent），用于多个变量的情况
    * 广义线性模型算法（Solving GLMs），多变量 + C/C++  的接口
    * 期望最大化算法，多变量 + 有限混合模型
* 仿真和重采样（Simulation and resampling0
    * Bootstrap法，并行编程的基础
    * "Map（映射）"和"Reduce（归约），适用于针对大规模数据的统计
	* 蒙特卡罗模拟算法（Monte Carlo simulations），更大规模的并行编程
    * MCMC算法（马尔可夫链蒙特卡罗(Markov Chain Monte Carlo)），多重采样，使用 GPU 进行编程


我（原作者）相信在 Duke 大学本专业内，这应该是第一次基于 Python 来讲的这门课，所以这门课会非常有意思。这也意味着教这门课需要准备很多新内容，我（原作者）就直接从已有的各种公有领域的 IPython notebook 里面来摘抄了。
