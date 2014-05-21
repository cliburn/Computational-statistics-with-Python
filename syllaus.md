STA 633 Statistical Computing and Computation
========================================

Environment
----------------------------------------

* Anaconda Python distribution with the Accelerate package
* PyCharm IDE
* C/C++ compiler
* R/RStudio
* Matlab (optional)

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
