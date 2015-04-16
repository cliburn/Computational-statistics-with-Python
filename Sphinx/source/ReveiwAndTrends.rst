
What you should know and learn more about
=========================================

Statistical foundations
-----------------------

-  Experimental design

   -  Usualy want to isolate a main effect from confounders
   -  Can we use a randomized experiment design?

      -  Batch effects

-  Replication

   -  Essential for science

-  Exploratory data analysis

   -  Always eyeball the data
   -  Facility with graphics librareis is essential
   -  Even better are interactive graphics libraries (IPython notebook
      is ideal)

      -  Bokeh

-  As amount of data grows

   -  Simple algorithms may perform better than complex ones
   -  Non-parametric models may perform better than parametric ones
   -  But big data can often be interpreted as many pieces of small data

Computing foundations
---------------------

-  Polyglot programming

   -  R and/or SAS (for statistical libraries)
   -  Python (for glue and data munging)
   -  C/C++ (for high performance)
   -  Command line tools and `Unix
      philosophy <http://www.faqs.org/docs/artu/ch01s06.html>`__
   -  SQL (for managing data)
   -  Scala (for Spark)

-  Need for concurrency

   -  Functional style is increasingly important

      -  Prefer immutable data structures
      -  Prefer pure functions

         -  Same input always gives same output
         -  Does not cause any side effects

-  With big data, lazy evaluation can be helpful

   -  Prefer generators to lists
   -  Look at the ``itertools`` standaard library in Python 2

-  Composability for maintainability and extensibility

   -  Small pieces, loosely joined
   -  Combinator pattern
   -  Again, all this was in the original `Unix
      philosophy <http://www.faqs.org/docs/artu/ch01s06.html>`__

Mathematical foundations
------------------------

-  Core: probability and linear algebra
-  Calculus is important but secondary
-  Graphs and networks increasingly relevant

Statistical algorithms
----------------------

-  Numbers as leaky abstractions
-  Don't just use black boxes

   -  Make an effort to understand what each algorithm you call is doing
   -  At minimum, can you explain what the algorithm is doing in plain
      English?
   -  Can you implement a simple version from the ground up?

-  Categories of algorithms

   -  Big matrix manipulations (matrix decomposition is key)
   -  Continuous optimization - order 0, 1, 2
   -  EM algorithm has wide applicability in both frequentist and
      Bayesian domains
   -  Monte Carlo methods, MCMC and simulations

-  Making code fast

   -  Make it run, make it right, make it fast
   -  Python has amazing profiling tools - use them
   -  For profiling C code, try
      `gperftools <https://code.google.com/p/gperftools/>`__
   -  Compilation: Try numba or Cython in preference to writing raw
      C/C++
   -  Parallel programming

      -  Python GIL
      -  Use Queue from threading or multiprocessing to build a pipeline
      -  Skip OpenMP (except within Cython) and MPI

-  Big data

   -  `Spark <http://spark.apache.org/>`__ is the killer app

Libraries worth knowing about after numpy, scipy and matplotlib
---------------------------------------------------------------

Data management
~~~~~~~~~~~~~~~

-  `csvkit for working with CSV
   files <https://csvkit.readthedocs.org/en/0.9.1/>`__
-  `SQLalchemy for interacting with relational
   databases <http://www.sqlalchemy.org/>`__
-  `redis when you need a fast, scalable, persistent
   store <https://github.com/andymccurdy/redis-py>`__

Statisticcs
^^^^^^^^^^^

-  `pandas for dataframe manipulations <http://pandas.pydata.org/>`__
-  `statsmodels: Statistics in
   Python <http://statsmodels.sourceforge.net/>`__
-  `Rpy2 - using R in Python <http://rpy.sourceforge.net/>`__

Graphics
^^^^^^^^

-  `seaborn provides statistical
   graphics <http://stanford.edu/~mwaskom/software/seaborn/>`__
-  `ggplot is a port of ggplot2 <https://github.com/yhat/ggplot>`__
-  `Bokeh provides a modern graphics library callable from Python, R,
   Scala and Julia <http://bokeh.pydata.org/en/latest/>`__

Optimization
^^^^^^^^^^^^

-  `scipy.optimize for root finding, optimization and linear
   programming <http://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize>`__
-  `cvxpy for convex optimization <http://www.cvxpy.org/en/latest/>`__

Graph algorithms
^^^^^^^^^^^^^^^^

-  `networkx - pure Python library for moderate sized
   graphs <http://networkx.github.io/>`__
-  `graph-tool: fastest graph library on a single
   computer <https://graph-tool.skewed.de/>`__
-  `Spark GraphX for graphs on the
   cluster <https://spark.apache.org/graphx/>`__)

Machine learning
^^^^^^^^^^^^^^^^

-  `Comprehensive post on Machine
   learning <https://www.cbinsights.com/blog/python-tools-machine-learning/>`__
-  `sklearn is the standard package for ML in
   Python <http://scikit-learn.org/stable/>`__
-  MADLib in `SQL <https://github.com/madlib/madlib>`__ or
   `Python <https://github.com/pivotalsoftware/pymadlib>`__
-  `Spark MLLib for ML on the
   cluster <https://spark.apache.org/mllib/>`__

Text processing
^^^^^^^^^^^^^^^

-  `Natural language toolkit <http://www.nltk.org/>`__
-  `Topic modeling with Gensim <http://radimrehurek.com/gensim/>`__

