Computational Statistics with Python
=======

Very rough drafts  of IPython notebook based lecture notes for the MS Statistical Science course on [Statistical Computing and Computation](https://stat.duke.edu/courses/statistical-computing-and-computation), to be taught in Spring 2015.  The course will focus on the development of various algorithms for *optimization* and *simulation*, the workhorses of much of computational statistics. A variety of algorithms and data sets of gradually increasing complexity (1 dimension $\rightarrow$ many dimensions, fixed $\rightarrow$ adaptive, serial $\rightarrow$ parallel $\rightarrow$ massively parallel, small data $\rightarrow$ big data) will allow students to develop and practise the following skills:

* Practices for reproducible analysis
* Fundamentals of data management and munging
* Use Python as a language for statistical computing
* Use mathematical and statistical libraries effectively
* Profile and optimize serial code
* Effective use of different parallel programming paradigms

In particular, the focus in on algorithms for:

* Optimization
    * Newton-Raphson (functional programming and vectorization)
    * Quadrature (adaptive methods)
    * Gradient descent (multivariable)
    * Solving GLMs (multivariable  + interface to C/C++)
    * Expectation-maximization (multivariable + finite mixture models)
* Simulation and resampling
    * Bootstrap (basics of parallel programming)
    * Map-reduce applications in statistics for big data
	* Monte Carlo simulations (more parallel programming)
    * MCMC (various samplers - GPU programming)

I believe that this is the first time a python based course will be offered in the Department, so it is really exciting. It also means a lot of new material needs to be developed, and I am borrowing freely from existing public domain IPython notebooks.
