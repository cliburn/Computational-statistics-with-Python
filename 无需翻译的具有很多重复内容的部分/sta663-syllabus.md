Syllabus for STA 663
========================================

Instructor:: Cliburn Chan <cliburn.chan@duke..edu>
Instructor: Janice McCarthy <janice.mccarthy@duke.edu>
TA: Matt Johnson <mcj15@stat.duke.edu>

Github Repository for course materials
<https://github.com/cliburn/STA663-2015.git>

Overview
----------------------------------------

The goal of STA 663 is to learn statistical programming - how to write code to solve statistical problems. In general, statistical problems have to do with the estimation of some characteristic derived from data - this can be a point estimate, an interval, or an entire function. Almost always, solving such statistical problems involves writing code to collect, organize, explore, analyze and present the data. For obvious reasons, we would like to write good code that is readable, correct and efficient, preferably without reinventing the wheel.

This course will cover general ideas relevant for high-performance code (data structures, algorithms, code optimization including parallelization) as well as specific numerical methods important for data analysis (computer numbers, matrix decompositions, linear and nonlinear regression, numerial optimization, function estimation, Monte Carlo methods). We will mostly assume that you are comfortable with basic programming concepts (functions, classes, loops), have good habits (literate programming, testing, version control) and a decent mathematical and statistical background (linear algebra, calculus, probability).

To solve statistical problems, you will typically need to (1) have the basic skills to collect, organize, explore and present the data, (2) apply specific numerical methods to analyze the data and (3) optimize the code to make it run acceptably fast (increasingly important in this era of "big data"). STA 663 is organized in 3 parts to reflect these stages of statistical programming - basics (20%), numerical methods (60%) and high performance computing (20%).

Learning objectives
----------------------------------------

The course will focus on the development of various algorithms for *optimization* and *simulation*, the workhorses of much of computational statistics. The emphasis is on *comptutation* for statistics  - how to prototype, optimize and develop high performance computing (HPC) algorithms in Python and C/C++. A variety of algorithms and data sets of gradually increasing complexity (1 dimension $\rightarrow$ many dimensions, serial $\rightarrow$ parallel $\rightarrow$ massively parallel, small data $\rightarrow$ big data) will allow students to develop and practise the following skills:

* Practices for reproducible analysis
* Fundamentals of data management and munging
* Use Python as a language for statistical computing
* Use mathematical and statistical libraries effectively
* Profile and optimize serial code
* Effective use of different parallel programming paradigms

Pre-requisites
----------------------------------------

Review the following if you are not familiar with them

- Unix commands
- Using `git` for version control
- Writing Markdown
- Writing $\LaTeX$
- Using `make` to build programs

The course will cover the basics of Python at an extremely rapid pace. Unless you are an experienced programmer, you should probably review basic Python programming skills from the [Think Python](http://www.greenteapress.com/thinkpython/html/index.html) book. This is also useful as a reference when doing assignments.

Another very useful as a reference is the official [Python tutorial](https://docs.python.org/2/tutorial/)

Grading
----------------------------------------

- Computer lab quizzes (25%)
- Programming project 1 (25%)
- Programming project 2 (25%)
- Final exam (25%)

Computing Platform
----------------------------------------

Each student will be provided with access to a virtual machine image running Ubuntu - URLs for inidividual students will be provided on the first day. For GPU computing and map-reduce examples, we will be using the Amazon Web Services (AWS) cloud platform. Again, details for how to acccess will be provided when appropriate.

All code developed for the course should be in a personal Github repository called sta-663-firstname-lastname. Make the instructors and TA collaborators so that we have full access to your code. We trust that you can figure out how to do this on your own.

Lecture 1
----------------------------------------

- The IPython notebook
    - Markdown cells
    - Code cells
    - The display system
    - IPython magic
    - Interfacing with other languages
- Programming in Python
    - Basic types
    - Basic collections (tuples, lists, sets, dicts, strings)
    - Control flow
    - Functions
    - Classes
    - Modules
    - The standard library
    - PyPI and `pip`   
    - Importing other modules

Lecture 2
----------------------------------------

- Functional programming
	- Functions are first class objects
	- Pure functions
    - Iterators
    - Generators
    - Anonymous functions with lambda
    - Recursion
	- Decorators
	- The `operator` module
	- The `itertools` module
	- The `functools` module
	- The `toolz` module
	- Constructing a lazy data pipeline
- Working with text
    - string methods
    - The `string` module
    - The `re` module

Computer lab 1
----------------------------------------

See Lab01/Exercises01.ipynb in the course Github repository.

Lecture 3
----------------------------------------

- Obtaining data
    - CSV with `csv`
    - JSON with `json`
    - Web scraping with `scrapy`
    - HDF5 with `pyhdf`
    - Relational databases and SQL with `sqlite3`
    - The `datasets` module
- Scrubbing data
	- Removing comments
	- Filtering rows
	- Filtering columns
	- Fixing inconsistencies
	- Handling missing data
	- Removing unwanted information
	- Derived information
	- Sanity check and visualization 

Lecture 4
----------------------------------------

- Using `numpy`
	- Data types
    - Creating arrays
    - Indexing
	- Broadcasting
		- Outer product
    - Ufuncs
    - Generalized Ufuncs
    - Linear algebra in numpy
        - Calculating covariance matrix
        - Solving least squares linear regression
    - I/O in numpy
- Using `pandas`
	- Reading and writing data
	- Split-apply-combine
	- Merging and joining
	- Working with time series
- Using `blaze`

Computer lab 2
----------------------------------------

See Lab02/Exercises02.ipynb in the course Github repository.

Lecture 5
----------------------------------------

- From math to computing
    - Computer representation of numbers
    - Overflow, underflow, catastrophic cancellation
    - Stability
    - Conditioning
	- Direct translation of symbols to code is dangerous
- The purpose of computing is insight not numbers
- Examples of computation in statistics
    - Estimating parameters (point and interval estimates)
    - Estimating functions
    - Feature extraction, class discovery and dimension reduction
    - Classification and regression
    - Simulations and computational inference
- Algorithmic efficiency and big $\mathcal{O}$ notation
    - Examples from classic data structures and algorithms

Lecture 6
----------------------------------------

- Numerical linear algebra
    - Simultaneous linear equations
    - Column space, row space and rank
    - Rank, basis, span
    - Norms and distance
    - Trace and determinant
    - Eigenvalues and eigenvectors
    - Inner product
    - Outer product
    - Einstein summation notation
- Matrices as linear transforms
    - Types of matrices
        - Square and non-square
        - Singular
        - Positive definite
        - Idempotent and projections
        - Orthogonal and orthonormal
        - Symmetric
        - Transition
    - Matrix geometry illustrated

Computer lab 3
----------------------------------------

Coming soon

Lecture 7
----------------------------------------

- Matrix decompositions
    - LU (Gaussian elimination)
    - QR
    - Spectral
    - SVD
    - Cholesky
- Using `scipy.linalg`
- BLAS and LAPACK

Lecture 8
----------------------------------------

- Projections, ordination, change of coordinates
    - PCA in detail
    - PCA with eigendecomposition
    - PCA with SVD
   	- Related methods
        - ICA
 		- LSA
   		- Factor analysis

Computer lab 4
----------------------------------------

Coming soon

Lecture 9
----------------------------------------

- Regression and maximum likelihood as optimization problems
- Local and global extrema
- Univariate root finding and optimization
    - Golden section search
    - Bisection
	- Newton-Raphson and secant methods
- Packages for optimization
    - Using `scipy.optimization`
    - Using `statsmodels`
	- Using `scikits-learn`
- General approach to optimization
	- Know the problem
	- Multiple random starts
	- Combining algorithms
	- Graphing progress

Lecture 10
----------------------------------------

Coming soon

Computer lab 5
----------------------------------------

- Exercise 1: Given a sequence of function values, write a program to perform kernel density estimation using several kernels
- Exercise 2: Use line search optimization to solve a 1D logistic regression problem
- Exercise 3: Implement the secant method for finding roots in 1D
- Exercise 4: Implement Newton's method and find an approximate solution to several equations (including ones that diverge)

Lecture 11
----------------------------------------

- Constrained optimization
    - Lagrange multipliers
	- Barriers and penalties
	- Transfomations
	- Convex optimization
	- Packages: `cvxopt`, `cvxpy` and `pyopt`


Lecture 12
----------------------------------------

- The EM algorithm (1)
    - Convex and concave functions
    - Jensen's inequality
    - Missing data setup
    - Toy example - coin flipping with 2 biased coins

Computer lab 6
----------------------------------------

Coming soon

Lecture 13
----------------------------------------

- The EM algorithm (2)
    - Gaussian mixture model
    - EM for Bayesians - MAP of posterior distribution
	- Other applications of EM
    - EM variants

Lecture 14
----------------------------------------

- Monte Carlo methods
	- Random number generators
    - Generating random variates from a distribution
	- Quadrature, Monte Carlo estimation and Monte Carlo swindles
    - Estimate confidence intervals (bootstrap)
    - Compare competing statistics (statistical simulation - e.g. power)
    - Compare models (cross-validation)
    - Hypothesis testing (permutation-resampling)

Computer lab 7
----------------------------------------

Coming soon

Lecture 15
----------------------------------------

- Conducting a simulation experiment (case study)
- Experimental design
    - Variables to study
    - Levels of variables (factorial, Latin hypercube)
    - Code documentation
    - Recording results
    - Reporting
    - Reproducible analysis with `make` and $\LaTeX$

Lecture 16
----------------------------------------

- MCMC (1)
	- Toy problem - rats on drugs
    - Monte Carlo estimation
    - Importance sampling
    - Metropolis-Hasting
    - Gibbs sampling
    - Hamiltonian sampling
	- Assessing convergence
	- Using `pystan`
	- Using `pymc2`
	- Using `emcee`

Computer lab 8
----------------------------------------

Coming soon

Lecture 17
----------------------------------------

- MCMC (2)
    - Gaussian mixture model revisited
    - Gibbs sampling
    - Infinite mixture model with the Dirichlet process

Lecture 18
----------------------------------------

- Profiling 
    - Premature optimization is the root of all evil
    - Using `%time` and `%timeit`
    - Profiling with `%prun`
	- Line profiler
	- Memory profiler
- Code optimization
    - Use appropriate data structure
    - Use appropriate algorithm
    - Use known Python idioms
    - Use optimized modules
    - Caching and memorization
    - Vectorize and broadcast
    - Views 
    - Stride tricks

Computer lab 9
----------------------------------------

Coming soon

Lecture 19
----------------------------------------

- JIT compilation with `numba`
- Optimization with `cython`
- Wrapping C code
- Wrapping C++ code
- Wrapping Fortran code

Lecture 20
----------------------------------------

- [Why modern CPUs are starving and what can be done about it](http://www.pytables.org/docs/CISE-12-2-ScientificPro.pdf)
- Parallel programming patterns
- Amdahl's and GustClassifying points with the Gustafson's laws
- Parallel programming examples
    - JIT compilation with `numba`
	- Toy example - fractals 
    - Using `joblib`
    - Using `multiprocessing`
    - Using `IPython.Parallel`
    - Using `MPI4py`

Computer lab 10
----------------------------------------

Coming soon

Lecture 21
----------------------------------------

- GPU computing
	- Introduction to CUDA
	- Vanilla matrix multiplication
	- Matrix multiplication with shared memory
	- JIT compilation with `numba`
	- Example: Large-scale GMMs with CUDA

Lecture 22
----------------------------------------

- Map-reduce and Spark ([AWS](https://aws.amazon.com/articles/4926593393724923))

    - Problem - k-mer counting for DNA sequences
    - Small scale map-reduce using Python
    - Using `hadoop` with `mrjob`
    - Using `spark` with `pyspark`
    - Using `MLib` for large-scale machine learning

Computer lab 11
----------------------------------------

Coming soon


Supplementary Mateiral
========================================

SM 1 
----------------------------------------

- Using the command line
    - The Unix philosophy and `bash`
    - Remote computing with `ssh`
    - Version control with `git`
    - Documents with $\LaTeX$
    - Automation with `make`

SM 2
----------------------------------------

- Graphics in Python
    - Using `matplotlib`
    - Using `seaborn`
    - Using `bokeh`
    - [Using `daft`](http://daft-pgm.org/)

