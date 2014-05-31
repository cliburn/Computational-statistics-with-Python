STA 633 Statistical Computing and Computation
========================================

Pre-requisites (with testing)
----------------------------------------

* Working knowledge of linear algebra
* Working knowledge of multivariable calculus
* Able to use a Unix shell
* Able to code in a high level language such as R or Matlab
* Able to write a simple C/C++ program and build it using a Makefile
* Basic understanding of version control and use of Git

Learning objectives
----------------------------------------

The course will focus on the development of various algorithms for *optimization* and *simulation*, the workhorses of much of computational statistics. A variety of algorithms and data sets of gradually increasing complexity (1 dimension $\rightarrow$ many dimensions, fixed $\rightarrow$ adaptive, serial $\rightarrow$ parallel $\rightarrow$ massively parallel, small data $\rightarrow$ big data) will allow students to develop and practise the following skills:

* Practices for reproducible analysis
* Fundamentals of data management and munging
* Use Python as a language for statistical computing
* Use mathematical and statistical libraries effectively
* Profile and optimize serial code
* Effective use of different parallel programming paradigms

In particular, the following algorithms will be covered:

* Optimization
    * Newton-Raphson (functional programming and vectorization)
    * Quadrature (adaptive methods)
    * Gradient descent (multivariable)
    * Solving GLMs (multivariable  + interface to C/C++)
    * Expectation-maximization (multivariable + finite mixture models )
* Simulation
    * Bootstrap (basics of parallel programming)
    * Bag of little bootstraps (map-reduce)
	* Monte Carlo simulations (more parallel programming)
    * MCMC (Gibbs sampler - GPU programming)

Course outline
----------------------------------------

1. Overview and setting up of local and cloud compute environment
2. Introduction to Python, the IPython notebook and PyCharm $\checkmark$
3. Functional and recursive programs in Python <font color=red>$\checkmark$</font>
4. Numerical computing in Python (numpy, blaze)
5. Statistical computing in Python (Rmagic, pandas, statsmodels)
6. Bayesian statistics in Python (pymc, emcee, multinest, pystan ) $\checkmark$
7. Graphics in Python (matplotlib, bokeh, seashore) <font color=red>$\checkmark$</font>
8. Testing, debugging and optimization 
9. The Newton-Raphson method $\checkmark$
10. Divide and conquer - adaptive quadrature
11. Numerical optimization in $k$-dimensions - gradient descent algorithm $\checkmark$
12. Solving generalized linear models with IRLS
13. Expectation-maximization for finite mixture models
14. Overview of parallel and high performance computing
15. Embarrassingly parallel problems with multiprocessing and IPython.Parallel
16. Parallel tasks with MPI $\checkmark$
17. Map-reduce for data transforms and summaries
18. Map-reduce and the bag of little bootstraps
19. Introduction to massively parallel programming
20. MCMC on GPUs $\checkmark$






Environment
----------------------------------------

* Unix shell
* Anaconda Python distribution with the Accelerate package
* PyCharm IDE
* C/C++ compiler
* R/RStudio

Examples of statistical applications
----------------------------------------

* Linear algebra (?)
    * Matrix multiplication
    * Matrix decomposition
    * Matrix inversion
    * Eigenvalues and eigenvectors of a covariance matrix
* Optimization
    * Solving least squares and GLM type methods
	* Newton method
    * Gradient descent
    * Lagrange multipliers
* Simulation
	* Bootstrap
	* Permutation sampling
	* Monte Carlo
* Smoothing and density estimation
    * Kernel density estimation
    * Expectation-Maximization
    * MCMC
    * Adaptive quadrature
* Possible data examples
    * Genomics data from 1000 Genomes project
    * Electronic health records from kaggle.com
    * US Census Data

Programming
----------------------------------------

* Defensive programming
    * Reproducible analysis
    * Version control
    * Testing
    * Literate programming
    * Automation with Makefiles
* Data munging
    * Extracting data from text files (e.g. Fasta, VCF)
    * Extracting data from JSON files
    * Extracting data from XML files
    * Extracting data from relational databases
* Data structures and algorithms
    * Computational complexity and Big O notation
	* Iterators and generators
	* Divide and conquer and recursion
    * Online or streaming algorithms
* Coding in a functional style
    * Higher order functions
	    * Apply, map, reduce
    * Split-apply-combine
   * Partial application
* Computational linear algebra
    * Solving linear systems
    * Eigenstructure of covariance matrix
    * Libraries for linear algebra (BLAS, LAPACK, Armadillo, Eigen)
* Polyglot programming
    * Python
    * Interfacing with R/Matlab
    * C/C++
* Structured data and databases (?)
* Statistical graphics and visualization (?)

High performance computing
----------------------------------------

* Benchmarking and profiling
    * Heuristic hierarchy of things to try
        * Solve a simpler problem
        * Choose more appropriate data structure
        * Choose more appropriate serial algorithm
        * Vectorize
        * Compile bottleneck to native code
		    * Use JIT where available
			* Intermediate language (e.g. Cython)
			* Write native code that interfaces to R/Python - e.g. Rcpp, bitey, f2py
        * Find a machine with more RAM
        * Implement appropriate parallel programming solution for problem (moving to a cloud platform is often most economical solution)
            * Large-grain, embarrassingly parallel - distributed computing
            * Large-grain with communication - MPI
            * Fine-grain (millions of tasks) - GPU computing
            * Massive data sets that won't fit into memory with simple calculations - MapReduce
* Vectorization
* Compiling to native code
    * Writing C/C++ modules to call from Python
    * Cython
    * JIT with Numba and NumbaPro
    * Interfacing R with Rcpp
* Parallel programming concepts
    * Computation
    * Memory and latency
    * Common parallel programming patterns and when to apply them
* Cloud computing and virtual machines
* Multi-core programming in IPython
* MPI programming
* Massively parallel programming with CUDA
* MapReduce for big data
