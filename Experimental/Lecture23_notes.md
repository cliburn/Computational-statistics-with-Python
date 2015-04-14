### Statistical foundations

- Experimental design
    - Usualy want to isolate a main effect from confounders
    - Can we use a randomized experiment design?
        - Batch effects
- Replication
    - Essential for science
- Exploratory data analysis
    - Always eyeball the data
    - Facility with graphics librareis is essential
    - Even better are interactive graphics libraries (IPython notebook is ideal)
        - Bokeh
- As amount of data grows
    - Simple algorithms may perform better than complex ones
    - Non-parametric models may perform better than parametric ones
    - But big data can often be interpreted as many pieces of small data

### Computing foundations

- Polyglot programming
    - R and/or SAS (for statistical libraries)
    - Python (for glue and data munging)
	- C/C++ (for high performance)
    - Command line tools and [Unix philosophy](http://www.faqs.org/docs/artu/ch01s06.html)
    - SQL (for managing data)
	- Scala (for Spark)
- Need for concurrency
    - Functional style is increasingly important
        - Prefer immutable data structures
	    - Prefer pure functions
            - Same input always gives same output
	        - Does not cause any side effects
- With big data, lazy evaluation can be helpful
    - Prefer generators to lists
    - Look at the `itertools` standaard library in Python 2
- Composability for maintainability and extensibility
    - Small pieces, loosely joined
    - Combinator pattern
	- Again, all this was in the original [Unix philosophy](http://www.faqs.org/docs/artu/ch01s06.html)

### Mathematical foundations

- Core: probability and linear algebra
- Calculus is important but secondary
- Graphs and networks increasingly relevant

### Statistical algorithms

- Numbers as leaky abstractions
- Don't just use black boxes
    - Make an effort to understand what each algorithm you call is doing
    - At minimum, can you explain what the algorithm is doing in plain English?
    - Can you implement a simple version from the ground up?
- Categories of algorithms
    - Big matrix manipulations (matrix decomposition is key)
    - Continuous optimization - order 0, 1, 2
	- EM algorithm has wide applicability in both frequentist and Bayesian domains
	- Monte Carlo methods, MCMC and simulations
- Making code fast
    - Make it run, make it right, make it fast
    - Python has amazing profiling tools - use them
	- For profiling C code, try [gperftools](https://code.google.com/p/gperftools/)
	- Compilation: Try numba or Cython in preference to writing raw C/C++
	- Parallel programming
	    - Python GIL
	    - Use Queue from threading or multiprocessing to build a pipeline
		- Skip OpenMP (except within Cython) and MPI
- Big data
    - [Spark](http://spark.apache.org/) is the killer app

### Libraries worth knowing about

### Data management

- [cvskit](https://csvkit.readthedocs.org/en/0.9.1/)
- [SQLalchemy](http://www.sqlalchemy.org/)
- [redis](https://github.com/andymccurdy/redis-py)

#### Statisticcs

- [pandas](http://pandas.pydata.org/)
- [statsmodels](http://statsmodels.sourceforge.net/)
- [Rpy2](http://rpy.sourceforge.net/)

#### Graphics

- [seaborn](http://stanford.edu/~mwaskom/software/seaborn/)
- [ggplot](https://github.com/yhat/ggplot)
- [Bokeh](http://bokeh.pydata.org/en/latest/)

#### Optimization

- [scipy.optimize](http://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize)
- [cvxpy](http://www.cvxpy.org/en/latest/)

####  Graph algorithms

- [networkx](http://networkx.github.io/)
-  [graph-tool](https://graph-tool.skewed.de/)
- [Spark GraphX](https://spark.apache.org/graphx/))

####  Machine learning

- [Machine learning](https://www.cbinsights.com/blog/python-tools-machine-learning/)
- 	[sklearn](http://scikit-learn.org/stable/)
- MADLib in [SQL](https://github.com/madlib/madlib) or [Python](https://github.com/pivotalsoftware/pymadlib)
- [Spark MLLib](https://spark.apache.org/mllib/)
