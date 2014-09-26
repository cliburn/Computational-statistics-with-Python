Course organization
===========================

Most of science, including statsitcs, revolves around the design of models to represent data of some sort. In statistics, these models are typically probability distributions with parameters to estimate. The typical stages of analysis goes something like this:

1. Data management 
    1. Receive possibly messy and bad data
    2. Clean and filter the data 
2. Expoloratory data analysis
    1. Describe and visualize the data to come up with possible models
3. Statistical inference
    1. Estimate model parameters given data (Model estimation and selection)
    2. Make predictions with model (Model prediction)
4. Reporting

In today's context, statisticians will be doing all three stages in front of a computer. We believe that modern analysiis of massive data sets is best achieved using a combination of complementatry software tools, and the course will cover what we consider to be an essential toolkit comprising `bash`, `git`, `make`, `sqlite3`, `python`, `R`, `C` and `LaTeX`. We use Python as the computational "glue" that integrates these collection of tools, and also in its capacity as an efficient high-level language for scientiific and statistical computing.

To deal with the massive data sets that you will enocounter in your career, the course will emphasize reproducible analysis, code optimization, high-perforamnce computing and cloud computing. Examples will be drawn from the core topics in computational statistics of optimization (e.g. smoothing, interpolation, maximum lkelihood, constrained and unconstrained methods) and simulation (e.g. jackknife, bootstrap, permutation, Monte Carlo integrals, MCMC).

At the end of the course, these are the practical skills every student should learn:

1. How to set up a reproducible analysis pipeline using bash, git, make and LaTeX
2. How to clean, manage and manipulate huge data sets using text processing, relational databases
3. How to explore data sets interactively using the IPython notebook and visualization packages
4. How to code statistical routines efficiently in high level languages 
5. How to optimize statistical routines by compiling to native code
6. How to compute in parallel on multi-core machines, clusters and GPUs

In particular, students should be able to write readable, well-documented, efficient (and if necessary parallel) code to implement a statistical method described in a textbook or paper, and apply it to real-world, possibly messy, data sets.

### Unit 1: Setting up a reproducible analysis pipeline (5 hours)
* Setting up workspace and introduction to bash
* Version control with git
* Document generation with LaTeX
* Automating the pipeline with make
* Introduction to Python and the IPython notebook
* Testing and debugging

**Exercise**: Create a git repository on Github for this project. Write a makefile to automate generation of a LaTeX report with embedded R and Python results. Ensure that git commits are performed regularly and well-documented.

**Exercise**: You are given a Python program that has errors. Fix it.

**Exercise**: Write functions to calculate the mean, median and variance of a set of numbers using test-driven development.

### Unit 2: Data manipulation and munging (5 hours)
* Reading and writing data
* Text processing and regular expressions
* Querying a relational database with SQL
* Database design for statisticians

**Exercise**: You are given a data set that needs to be cleaned and reformatted into a data frame.

**Exercise**: Given an SQLite3 database, use SQL to answer some questions and extract data subsets.

**Exercise**: Given a spreadsheet, design a normalized database to manage the data. Transfer the data from the spreadsheet into the database.

### Unit 3: Exploratory data analysis and visualization (5 hours)
* Manipulating data in a DataFrame
* Visualizing data with matplotlib
* Grammar of graphics with ggplot and bokeh
* Animation - Metropolis, Gibbs and Hamiltonian sampling

**Exercise**: Load a dataset into a DataFrame and use joins and the split-apply-combine pattern to answer some quesiotns.

**Exercise**: Plot and annotate the given dataset to illustrate its key features.

### Unit 4: Efficient statistical routines (15 hours)
* Broadcasting and vectorization in numpy and pandas
* Functional programming
* Representation of numbers and linear algebra
* Introducing BLAS and LAPACK
* Quadrature (Numerical integration)
* Constrained and unconstrained optimization
* Resampling methods
* Monte Carlo simulations
* Markov chain Monte Carlo

**Exercise**: You are given a slow and buggy simulation script. Fix the errors and speed it up using vectorization.

**Exercise**: Write an efficient function to calculate Cook's distance for the influence of data points on a regression.

**Exercise**: Use Newton's method to fit a logistic regression model (aka Iterative reweighted least squares).

**Exercise**: Implement the non-parametric and parametric bootstrap for phylogenetic trees described in <http://www.pnas.org/content/93/23/13429.full.pdf>

**Exercise**: Use simulation to perform evaluate the power of .

**Exercsie**: Use symbolic integration, numerical integration and Monte Carlo integration to evaluate a definite double integral

**Exercise**: Use regular Python, PyMC3 and PyStan to find the posterior distribution for a two-level model.

### Unit 5: Code optimization and native code (5 hours)
* Complexity and performance of algorithms and data structures
* C crash course for statisticians
* Using numexpr, numba and cython
* Using functions from C/C++ libraries
* Writing functions in C/C++ and wrapping for Python/R

**Exercise**: You are given some slow code. Speed it up by using a better algoithm or data structure.

**Exercise**: Write the Newton-Raphson method in C - it should take the following arguments - a function pointer f, a function pointer fprime, an initial point x0, and a tolerance.

**Exercise**: You are given some slow code in Python. Speed it up using Cython.

### Unit 6: High-performance computing (10 hours)
* Parallel progrmming patterns
* Multiprocessing and IPython.Parallel
* Processing big data with MapReduce
* Multi-CPU computing with MPI
* GPU computing with CUDA

**Exercise**: Rewrite the function to calculate Cook's distance using multiprocessing.

**Exercise**: Use Elastic MapReduce to do some massive genomic data manipulation.

**Exercise**: Use MPI to run an MCMC with parallel temperinig (aka $MC^3$) for a long time with some defined swap interval between chains.

**Exercise**: Write a matrix multiplication kernel with and without use of shared memory using CUDA. Test it out using square matrices initialized with random numbers from CURAND. 
