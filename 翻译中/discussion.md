What would you like your students to learn in a 2nd course on statistical computing?
========================================

> Make it run, make it right, make it fast.

Background
----------------------------------------

* Offering [STA 633](http://people.duke.edu/~ccc14/sta-633-2015/): Statistical computing and computation in Spring 2015
* Schedule is 2 lectures (75 mins) and 1 lab (75 mins) per week
* This is a *2nd* course in statistical computing - pre-req is Colin Rundel's class
    * [STA 523](https://stat.duke.edu/~cr173/Sta523_Fa14/)
    * Quite fast-paced - recommended books are
        * Advanced R - Wickham 
        * R Packages - Wickham
    * Will cover ```Unix shell```, ```make```, ```git```, ```markdown``` and programming in R
    * We will have a pretest to determine eligibility if students have not taken STA 523

Proposed learning objectives
----------------------------------------

* Basically teach *all the computing that we would personally like to see in a PhD student or postdoc working with us*
    * Comfortable using both high (Python/Julia?/R) and low level languages (C/C++)
    * Understand data management and use of relational database
		* Working with "bad" data
		    * <font color=red>Examples?</font>
        * Hands-on exercise building a normalized database from a spreadsheet and querying it via SQL
    * Can build reproducible data analysis pipelines (testing + make + literate programming)
    * Can convert a statistical model (e.g. from manuscript or textbook) into a numerical algorithm
        * Understanding of basic algorithms for optimization, simulation and smoothing
			   * Building blocks for large classes of statistical algorithms
               * <font color=red>What algorithms should students know?</font>
		* Pragmatic usage of libraries for established numerical routines
               * <font color=red>Recommendations for C/C++ libraries</font>
	* Can write code that is *correct*
	    * How much and what kind of testing is appropriate?
	    * How to test code with stochastic elements
	* Can write code that runs *fast*
		* Trade-off between computation and programmer time (premature optimization)
	    * Some understanding of complexity trade-offs for algorithms and data structures
        * Benchmarking and profiling
        * JIT compilation
        * Writing native code
        * Exploiting multiple cores (threading, multiprocessing, OpenMP)
        * Exploiting multiple machines (MPI)
        * Exploiting GPUs (CUDA, maybe OpenCL)
        * Working with really big data (MapReduce)

[Units](http://localhost:8888/tree)
----------------------------------------

Unit 1: Reproducible analysis and introducing Python as a glue language (10%)
Unit 2: Working with data - data munging and relational databases (10%)
Unit 3: Exploratory data analysis and visualization (10%)
Unit 4: Core statistical algorithms and libraries (40%)
Unit 5: C bootcamp, code profiling and writing native code (15%)
Unit 6: Parallel computing and working with big data (15%)

Discussion
----------------------------------------

* <font color=red>Overall course objectives?</font>
* <font color=red>Overall course content?</font>
    * Are there useful classes of topics we have left out?
    * Within each topic, what content should students learn?
        * Unit 1: Reproducible analysis and introducing Python as a glue language (10%)
        * Unit 2: Working with data - data munging and relational databases (10%) 
        * Unit 3: Exploratory data analysis and visualization (10%)
        * Unit 4: Core statistical algorithms and libraries (40%)
        * Unit 5: C bootcamp, code profiling and writing native code (15%)
        * Unit 6: Parallel computing and working with big data (15%)
* <font color=red>How can programming be taught effectively?</font>
    * Every good programmer I know is self-taught ...
    * MCQs for rapid sanity check on level of understanding each week
    * Less talking, more doing - mini-project after each unit
    * Individual or group work?
* <font color=red>What are statistical algorithms students should know?</font>
    * Know the theory and how to use a good implementation
        * Teach understanding with toy example
		* Use library to solve more realistic problem
    * Examples
	    * Linear algebra e.g. projection, normal equations
        * Optimization - e.g. Newton, IRLS, multivariate gradient descent, EM
        * Simulation - resampling methods, Monte Carlo, MCMC
        * Others? Smoothing, interpolation etc
* <font color=red>What are good data sets and problems to use for teaching?</font>
	* Bad data
	* Big data
	* Slow and fast versions 
