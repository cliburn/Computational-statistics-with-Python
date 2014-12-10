Topic coverage
========================================

Statistical computing and computational statistics
----------------------------------------

What is statistical computing?

- Understanding computational methods useful for statistical inference
    - Data structures and classical algorithms
    - Efficiency and trade-offs
    - Numerical analysis algorithms

What is computational statistics?

- Use of the computer to conduct statistical experiments for computational inference
    - From model to data
    - From data to model
    - Model as a data generating process

What are some tools for computational statistics and statistical computing?

- Optimization
- Simulation
- Function approximation and smoothing
- Finding structure
- Modeling dependencies
- Statistical visualization	

Numerical Analysis
----------------------------------------

- Computer numbers
    - Representation
    - Stability
    - Conditioning
- Linear algebra 
    - Distance measures and norms
    - Vector spaces and inner product
    - Matrix operations - row, column and nul space
    - Matrix decompositions

Data structures and algorithms
----------------------------------------

- Data structures
    - List
    - Tuple
    - Dictionary and set
    - Heap
	- Trees
- Algorithms
    - Complexity and Big O notation
    - Algorithm design
        - Greedy search
        - Local search
        - Dynamic programming and caching
        - Approximation and embedding
    - Searching and sorting
    - Combinatorial algorithms
    - Graphs and networks

Using Git in development workflow
----------------------------------------

- 5 minute tutorial
    - Create an empty githbu repository
    - Clone Github repository to local drive
    - `git config --global user.name "My name"`
    - `git config --global user.email "My email"`
    - Create two files
    - `git add`
    - `git commit`
    - `git push`
    - Check on Github
    - Create a branch
    - `git checkout foo`
    - Make soem changess in branch
    - Delete branch
    - `git branch -d foo`
    - Create a branch
    - `git checkout bar`
    - Make some changes in branch
    - `git add`
    - `git commit`
    - Merge branch to master
    - `git checkout master`
    - `git merge master`
    - create a branch
    - `git checkout baz`
    - Make some changes in branch
    - Temporarily save changes
    - `git stash`
    - Switch to main branch
    - `git checkout master`
    - Make some changes in master
    - Resume work in baz
    - `git checkout baz`
    - `git stash pop`
- Starting IPython notebook with --script argument for version control

Testing
----------------------------------------

- Writing unit tests with pytest
- Writing unit tests in IPython <https://github.com/zonca/pytest-ipynb>
- Continuous integration with TravisCI

Debugging
----------------------------------------

- Using pdb in IPython
- Using a GUI (e.g. PyCharm)

Using a vritual environment for reproducible analysis
----------------------------------------

See [tutorial](http://astropy.readthedocs.org/en/latest/development/workflow/virtual_pythons.html)

Generating documentation
----------------------------------------

- Using IPython  with [Section numberiing](https://github.com/dpsanders/ipython_extensions/tree/master/section_numbering) and  [ipyBibtex](https://gist.github.com/z-m-k/6080008)
- Using `nbconvert`
- Using PythonTeX

Packaging Python moduels for distribution
----------------------------------------

- Using binstar
- Using PyPI

Using Python
----------------------------------------

- Fundamentals
    - Data types and collections
    - Loops
    - Writing custom functions
- Functional programming
    - Functions as first class objects
    - Higher order functions
    - Iterators and generators
	- Comprehensions
    - Partial application and currying
    - Recursion
    - Idioms - map, filter, fold
	- Functional libraries
	- Parsers 
	- The countdown
- Python for productivity
	- The IPython notebook for prototyping
	    - Literate programming
        - Inline visualization
        - Easy access to multiple tools
            - Command line
            - Editors
            - Other languages
    - Python as a glue language
		- Obtaining and scrubbing data
		- Python and LaTeX - PythonTeX
        - Interfacing with R, Matlab and Julia
        - User interface development
    - From Python to C
    - From single threaded to parallel and massively parallel
- Python for data science
    - IPython
	    - Using kernal and multiple clients
        - [Jupyter](http://jupyter.org/)
		- Using `%qtconsole`
		- [Customizing IPython with JavaScript](https://github.com/ipython-contrib/IPython-notebook-extensions/wiki)
    - PythonTeX
    - Command line tools
    - Obtaining data
        - Working with relational databases
            - SQLite
            - [dataset](http://dataset.readthedocs.org/en/latest/)
			- [PonyORM](http://ponyorm.com/)
        - Working with hierarchical databases (HDF5)
        - Working with Excel and CSV 
        - Working with JSON and XML
        - Working with remote data
    - Scrubbing data
        - Text processing
        - Regex
        - Using Pandas
	- Virtualenv
- Python for statistics
    - Numpy
        - Ufuncs
        - vectorization (EM example)
        - tensordot
        - einsum
    - Scipy
    - Scikits-learn, Scikits-monaco, Scikits-bootstrap, Scikits-image
    - Pandas
    - Patsy
    - StatsModels
    - SymPy
    - PyStan, PyMC, emcee
- Python for visualization
    - Matplotlib
	- plt.rcdefaults() after e.g. xkcd()
	- or use
	```
	with plt.xkcd():
        fig1 = plt.figure()
    ```
	- Seaborn
		- seaborn.reset_orig() to restore Matplotlib defaults
	- Vispy
	- ggplot
	- Bokeh
	- Illustrating matrix multiplication with IPython blocks
	- Illustration of IPython notebook widgets and interactive features
	- Publication quality graphics with [Asymptote](https://github.com/jrjohansson/ipython-asymptote)
- Python for HPC
	- [Joblib for caching and embarrassingly parallel tasks](https://pythonhosted.org/joblib/)
    - Cython, bottleneck, CythonGSL
    - Multiprocessing 
    - NumExpr, Numba and NumbaPro
    - Theano
    - MPI4py
- Python as a glue language
    - Combining R, Matlab, Julia

Optimization
----------------------------------------

- Standard approaches
    - Minimize residuals
    - Maximum likelihood
    - Method of moments
    - [x] Expectation-maximization (EM)
- Methods
    - Nelder-Mead
    - First-order methods
		- [x]Gradient descent 
		- Stochastic gradient descent
    - Second-order methods
        - Newton
        - Gauss-Newton
        - IRLS
        - Quasi-Newton
        - Conjugate gradient
        - Levenberg-Marquadt
    - Global methods
        - Simulated annealing
        - Basin hopping

Simulation
----------------------------------------

- Random number generation
- Data partitioning
- Bootstrap and Jackknife
- Permutation sampling
- Monte Carlo methods
    - Rejection sampling 
    - Importance sampling 
- Markov chain Monte Carlo
    - Simulated tempering
    - MCMCMC 	
- Particle filters
- Simulation as an experiment
    - Design
        - Random sampling
        - Latin Hypercube sampling
        - Orthogonal
	- Documentation and reproducible analysis

Approximating functions
----------------------------------------

- Distance between functions
- Quadrature
    - Trapezoid and Simpsons
    - Newton-Cotes
    - Adaptive quadrature
- General families of parametric functions
- Kernel methods
- Orthogonal basis functions
    - Polynomials
    - Fourier
    - Hermite
- Bayesian non-parametric models

Structure
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

Dependency
----------------------------------------

- Transformations
- Generalized models
    - Linear regression
    - Logistic regression
- Hierarchical structure
- Sequential structure

Visualization
----------------------------------------

- Conditional plots
- Scatter matrix
- Image map
- Brushing and linked displays
- Parallel coordinates and Andrews curves
- Dimension reduction
    - Multi-dimensional scaling (MDS)
- Projection pursuit 

Key algorithms
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
