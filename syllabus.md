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

The course will focus on the development of various algorithms for *optimization* and *simulation*, the workhorses of much of computational statistics. The emphasis is on *comptutation* for statistics  - how to prototype, optimize and develop high performance computing (HPC) algorithms in Python and C/C++. A variety of algorithms and data sets of gradually increasing complexity (1 dimension $\rightarrow$ many dimensions, fixed $\rightarrow$ adaptive, serial $\rightarrow$ parallel $\rightarrow$ massively parallel, small data $\rightarrow$ big data) will allow students to develop and practise the following skills:

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
2. Introduction to Python, the IPython notebook and PyCharm [jm]
3. Functional programming [cc]
4. Numerical building blocks [cc]
5. Statistical building blocks (Rmagic, pandas, statsmodels) [jm]
7. Graphics (matplotlib, bokeh, seashore, bokeh) [cc]
8. Profiling and optimization [cc]
9. Interfacing with compiled languages [cc]
9. The Newton-Raphson method [jm]
12. Numerical optimization in $k$-dimensions - gradient descent algorithm [jm]
11. Solving generalized linear models with IRLS [cc]
12. Divide and conquer - adaptive quadrature [cc]
13. Expectation-maximization for finite mixture models [cc/jm]
6. Bayesian models (I) [cc]
7. Bayesian models (II) [cc]
14. Overview of parallel and high performance computing [cc/jm]
15. Beginning parallel programming with multiprocessing and IPython.Parallel [cc]
16. Parallel tasks with MPI [jm]
17. Map-reduce for data transforms and summaries [cc]
18. Map-reduce and the bag of little bootstraps [cc]
19. Introduction to massively parallel programming [jm]
20. MCMC on GPUs [jm]
21. Data management [cc]


Course details
----------------------------------------

Course outline
----------------------------------------

* Overview and setting up of local and cloud compute environment
    * Local environment
        * Anaconda distribution
        * Extra python packages
        * Extra IPython extensions
        * Compiler support (C/C++, Fortran)
        * R
    * Cloud compute
        * Logging in to EC2 instance
        * Using ssh
        * Using tmux
        * AWS console
        * Shutting down virtual machine
* Introduction to Python, the IPython notebook and PyCharm
* Functional programming
    * Motivation
    * Iterators
    * List comprehensions and Generator expressions
    * Generators
    * The `functools` package
        * map
        * reduce
        * partial
    * The `operator` package
    * The `itertools` package
    * The `functional` package
		* compose
    * Decorators
	* Recursion
    * **Example**: counting words in books with map and reduce
    * **Example**: a generic bootstrap function
    * **Example**: recursion on trees
* Numerical computing
    * Introduction to numpy arrays
        * Dimension and axis
	    * Indexing and slicing
        * Broadcasting
    * Random number generators
    * Linear algebra
        * [Vector and matrix operations](http://cs.bc.edu/~alvarez/Randomness/Notes/matrixOperations)
            * Matrix multiplication as weighed combinations
            * Inner product
            * Outer product
            * Einstein summation convention
        * Matrix decomposition
            * Eigensystem
            * LU
            * QR
            * SVD
            * Cholesky
        * Low level functions
            * BLAS
            * Lapack
    * **Example**: Pitfalls: How computers handle numbers
        * Integer and floating point division in Python
        * Calculating the pseduo-inverse and division by almost zero
        * Not working in log space for probabilities
        * Calculating a negative variance using the textbook definition
    * **Example**: Calculating the covariance matrix with matrix operations
    * **Example**: Solving least squares problems
* Statistical computing (Rmagic, pandas, statsmodels)
    * Managing data with Pandas
        * Series
        * DataFrame
        * Panels
        * I/O
        * Split-apply-combine
    * Statistical models
        * Getting data sets
        * Basic statistics
        * Linear Regression
        * Generalized linear models
    * Using R from Python
    * [Machine learning](http://www.cbinsights.com/blog/python-tools-machine-learning)
        * Scikits-learn
* Bayesian models (pymc, emcee, multinest, pystan )
	* Ideas
	    * Estimating integrals
			* Generating univariate and multivariate random deviates
		   * Quadrature
		   * Rejection sampling
		   * Importance sampling
		   * Grid-based sampling
		   * Monte Carlo integration and the Strong Law of Large Numbers
		   * Markov chains, MCMC and the ergodic theorem
           * Primer on probabilistic graphical models
		   * Metropolis-Hastings - random walk and independent
		   * Gibbs sampling, full conditionals and conjugate priors
		   * Hamiltonian sampling
		   * Multiple walkers
    * Computation
        * Samplers with animation
            * Metropolis sampler
            * Gibbs sampler
            * Hamiltonian sampler
        * Convergence diagnostics
        * Posterior predictions
        * Estimating parameters of simple models
        * Hierarchical models
    * Implementation
        * Python and Cython
        * PyMC
        * PyStan
        * Emcee
	* **Example**: [Gibbs sampler shootout](http://nbviewer.ipython.org/gist/fonnesbeck/4166681)
    * **Example**: Logistic regression for classification
    * **Example**: [Gaussian mixture models for density estimation](http://www.nehalemlabs.net/prototype/blog/2014/04/03/quick-introduction-to-gaussian-mixture-models-with-python/)
* [Graphics](http://2014.pycon.se/assets/slides/Plotly-Pycon-Sweden.pdf) (matplotlib, bokeh, seashore)
    * Graphics with Matplotlib
    * Graphics with Pandas
    * Graphics with Seaborn
    * Graphics with ggplot
    * Interactive plots with mpld3 and plotly
    * Volume rendering with yt (?)
    * [Interactive web plotting](http://nbviewer.ipython.org/github/damianavila/bokeh_overview/blob/master/Bokeh%20Overview.ipynb?create=1) with `Bokeh`
* Profiling and optimization
    * [Profiling](https://docs.python.org/2/library/debug.html)
        * Manual timing with the `timeit` package
        * `%time` and `%timeit` functions
        * Using the `hotshot` profiler
        * Line profilers
        * Memory profilers
    * Optimization
        * Premature optimization
        * Measuring complexity
            * Big O notation
            * Time and space trade-offs
    * Understanding performance
           * Fundamental data structures
           * Fundamental algorithms
    * Python performance idioms
    * Vectorization
* Interfacing with compiled languages
    * Pure python
    * Numpy
    * JIT compilation
    * Cython (incremental optimization)
    * C/C++
    * Fortran
    * Julia
    * **Example**: Using C libraries (R standalone)
    * **Example**: Using C++ libraries (Eigen)
* The Newton-Raphson method
* Numerical optimization in $k$-dimensions - gradient descent algorithm
* Solving generalized linear models with IRLS
    * <http://scipystats.blogspot.com/2009/07/iterated-reweighted-least-squares.html>
    * <http://puzlet.com/m/b007h>
* Divide and conquer - adaptive quadrature
    * <http://en.wikipedia.org/wiki/Adaptive_quadrature>
    * <http://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method>
* Expectation-maximization for finite mixture models
    * <http://nbviewer.ipython.org/github/tritemio/notebooks/blob/master/Mixture_Model_Fitting.ipynb>
    * <http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf>
* Overview of parallel and high performance computing
    * [Starving CPUs](http://www.blosc.org/docs/StarvingCPUs-CISE-2010.pdf)
    * What is theoretically possible?
    * Parallel programming patterns
    * Parallel programming with IPython.Parallel
    * Using multiprocessing
    * **Example**: Independent data sets
    * **Example**: Monte Carlo simulations
* Parallel tasks with MPI
    * [A Python Introduction to Parallel Programming with MPI](http://jeremybejarano.zzl.org/MPIwithPython/)
* Parallel Cython with OpenMP
* Map-reduce for data transforms and summaries
* Map-reduce and the bag of little bootstraps
* Introduction to massively parallel programming
* MCMC on GPUs
* [Maybe] Working with bad data - getting data from various sources (Excel, JSON, XML, RDBMS, NoSQL), data munging, data validation and imputation.



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
