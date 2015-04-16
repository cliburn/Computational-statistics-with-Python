
.. code:: python

    from __future__ import division
    import os
    import sys
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    %matplotlib inline
    %precision 4
    plt.style.use('ggplot')


.. code:: python

    %load_ext cythonmagic

.. code:: python

    from numba import jit, typeof, int32, int64, float32, float64

.. code:: python

    import random

Writing Parallel Code
=====================

The goal is to desing parallel programs that are flexible, efficient and
simple.

**Step 0**: Start by profiling a serial program to identify bottlenecks

**Step 1**: Are there for opportunities for parallism?

-  Can tasks be perforemd in parallel?

   -  Function calls
   -  Loops

-  Can data be split and operated on in parallel?

   -  Decomposition of arrays along rows, columns, blocks
   -  Decomposition of trees into sub-trees

-  Is there a pipeline with a sequence of stages?

   -  Data preprocesing and analysis
   -  Graphics rendering

**Step 2**: What is the nature of the parallelism?

-  Linear

   -  Embarassingly parallel programs

-  Recursive

   -  Adaptive partitioning methods

**Step 3**: What is the granularity?

-  10s of jobs
-  1000s of jobs

**Step 4**: Choose an algorihtm

-  Organize by tasks

   -  Task parallelism
   -  Dvidie and conquer

-  Organize by data

   -  Geometric decomposition
   -  Recursvie decomposition

-  Organize by flow

   -  Pipeline
   -  Event-based processing

**Step 5**: Map to program and data structures

-  Program structures

   -  Single program multiple data (SPMD)
   -  Master/worker
   -  Loop parallelism
   -  Fork/join

-  Data structures

   -  Shared data
   -  Shared queue
   -  Distributed array

**Step 6**: Map to parallel environment

-  Multi-core shared memrory

   -  Cython with OpenMP
   -  multiprocessing
   -  IPython.cluster

-  Multi-computer

   -  IPython.cluster
   -  MPI
   -  Hadoop / Spark

-  GPU

   -  CUDA
   -  OpenCL

**Step 7**: Execute, debug, tune in parallel environment

Concepts
--------

-  A **task** is a chunk of work that a parallel Unit of Execution can
   do
-  A **Unit of Execution (UE)** is a process or thread
-  A **Processing Element (PE)** is a hardware computational unit - e.g.
   a hyperthreaded core
-  **Load balance** refers to how tasks are distributed to Processing
   Eleements
-  **Synchronization** occurs when execution must stop at the same point
   for all Units of Execution
-  **Race conditions** occur when different Units of Executions compete
   for the same resource and the output depends on who gets the resource
   first
-  **Dead locks** occur when A is waiting for B and B is waiting for A

Embarassingly parallel programs
-------------------------------

Many statistical problems are embarassingly parallel and cna be easily
decomposed into independent tasks or data sets. Here are several
examples:

-  Monte Carlo integration
-  Mulitiple chains of MCMC
-  Boostrap for condence intervals
-  Power calculations by simulation
-  Permuatation-resampling tests
-  Fitting same model on multiple data sets

Other problems are serial at small scale, but can be parallelized at
large scales. For example, EM and MCMC iterations are inherently serial
since there is a dependence on the previous state, but within a single
iteration, there can be many thousands of density calculations (one for
each data poinnt to calculate the likelihood), and this is an
embarassingly parallel problem within a single itneration.

These "low hanging fruits" are great because they offer a path to easy
parallelism with minimal complexity.

Estimating :math:`\pi` using Monte Carlo integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is clearly a toy example, but the template cna be used for most
embarassingly parallel problems. First we see how much we can speed-up
the serial code by the use of compilation, then we apply parallel
processing for a furhter linear speed-up in the number of processors.

.. code:: python

    def pi_python(n):
        s = 0
        for i in range(n):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if (x**2 + y**2) < 1:
                s += 1
        return s/n

.. code:: python

    stats = %prun -r -q pi_python(1000000)


.. parsed-literal::

     

.. code:: python

    stats.sort_stats('time').print_stats(5);


.. parsed-literal::

             4000004 function calls in 2.329 seconds
    
       Ordered by: internal time
       List reduced from 6 to 5 due to restriction <5>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    1.132    1.132    2.329    2.329 <ipython-input-5-fe39fab6b921>:1(pi_python)
      2000000    0.956    0.000    1.141    0.000 random.py:358(uniform)
      2000000    0.185    0.000    0.185    0.000 {method 'random' of '_random.Random' objects}
            1    0.056    0.056    0.056    0.056 {range}
            1    0.000    0.000    2.329    2.329 <string>:1(<module>)
    
    


.. code:: python

    def pi_numpy(n):
        xs = np.random.uniform(-1, 1, (n,2))
        return 4.0*((xs**2).sum(axis=1).sum() < 1)/n

.. code:: python

    @jit
    def pi_numba(n):
        s = 0
        for i in range(n):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x**2 + y**2 < 1:
                s += 1
        return s/n

This usse the GNU Scientific lirbary. You may need to instal it

.. code:: bash

    wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz
    tar -xzf gsl-latest.tar.gz
    cd gsl-1.16
    ./configure --prefilx=/usr/local
    make
    make install

and then

.. code:: bash

    pip install cythongsl

.. code:: python

    %%cython -a -lgsl
    import cython
    import numpy as np
    cimport numpy as np
    from cython_gsl cimport gsl_rng_mt19937, gsl_rng, gsl_rng_alloc, gsl_rng_uniform
    
    cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
    
    @cython.cdivision
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pi_cython(int n):
        cdef int[:] s = np.zeros(n, dtype=np.int32)
        cdef int i = 0
        cdef double x, y
        for i in range(n):
            x = gsl_rng_uniform(r)*2 - 1
            y = gsl_rng_uniform(r)*2 - 1
            s[i] = x**2 + y**2 < 1
        cdef int hits = 0
        for i in range(n):
            hits += s[i]
        return 4.0*hits/n




.. raw:: html

    <!DOCTYPE html>
    <!-- Generated by Cython 0.22 -->
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
        <style type="text/css">
        
    body.cython { font-family: courier; font-size: 12; }
    
    .cython.tag  {  }
    .cython.line { margin: 0em }
    .cython.code  { font-size: 9; color: #444444; display: none; margin: 0px 0px 0px 20px;  }
    
    .cython.code .py_c_api  { color: red; }
    .cython.code .py_macro_api  { color: #FF7000; }
    .cython.code .pyx_c_api  { color: #FF3000; }
    .cython.code .pyx_macro_api  { color: #FF7000; }
    .cython.code .refnanny  { color: #FFA000; }
    .cython.code .error_goto  { color: #FFA000; }
    
    .cython.code .coerce  { color: #008000; border: 1px dotted #008000 }
    .cython.code .py_attr { color: #FF0000; font-weight: bold; }
    .cython.code .c_attr  { color: #0000FF; }
    .cython.code .py_call { color: #FF0000; font-weight: bold; }
    .cython.code .c_call  { color: #0000FF; }
    
    .cython.score-0 {background-color: #FFFFff;}
    .cython.score-1 {background-color: #FFFFe7;}
    .cython.score-2 {background-color: #FFFFd4;}
    .cython.score-3 {background-color: #FFFFc4;}
    .cython.score-4 {background-color: #FFFFb6;}
    .cython.score-5 {background-color: #FFFFaa;}
    .cython.score-6 {background-color: #FFFF9f;}
    .cython.score-7 {background-color: #FFFF96;}
    .cython.score-8 {background-color: #FFFF8d;}
    .cython.score-9 {background-color: #FFFF86;}
    .cython.score-10 {background-color: #FFFF7f;}
    .cython.score-11 {background-color: #FFFF79;}
    .cython.score-12 {background-color: #FFFF73;}
    .cython.score-13 {background-color: #FFFF6e;}
    .cython.score-14 {background-color: #FFFF6a;}
    .cython.score-15 {background-color: #FFFF66;}
    .cython.score-16 {background-color: #FFFF62;}
    .cython.score-17 {background-color: #FFFF5e;}
    .cython.score-18 {background-color: #FFFF5b;}
    .cython.score-19 {background-color: #FFFF57;}
    .cython.score-20 {background-color: #FFFF55;}
    .cython.score-21 {background-color: #FFFF52;}
    .cython.score-22 {background-color: #FFFF4f;}
    .cython.score-23 {background-color: #FFFF4d;}
    .cython.score-24 {background-color: #FFFF4b;}
    .cython.score-25 {background-color: #FFFF48;}
    .cython.score-26 {background-color: #FFFF46;}
    .cython.score-27 {background-color: #FFFF44;}
    .cython.score-28 {background-color: #FFFF43;}
    .cython.score-29 {background-color: #FFFF41;}
    .cython.score-30 {background-color: #FFFF3f;}
    .cython.score-31 {background-color: #FFFF3e;}
    .cython.score-32 {background-color: #FFFF3c;}
    .cython.score-33 {background-color: #FFFF3b;}
    .cython.score-34 {background-color: #FFFF39;}
    .cython.score-35 {background-color: #FFFF38;}
    .cython.score-36 {background-color: #FFFF37;}
    .cython.score-37 {background-color: #FFFF36;}
    .cython.score-38 {background-color: #FFFF35;}
    .cython.score-39 {background-color: #FFFF34;}
    .cython.score-40 {background-color: #FFFF33;}
    .cython.score-41 {background-color: #FFFF32;}
    .cython.score-42 {background-color: #FFFF31;}
    .cython.score-43 {background-color: #FFFF30;}
    .cython.score-44 {background-color: #FFFF2f;}
    .cython.score-45 {background-color: #FFFF2e;}
    .cython.score-46 {background-color: #FFFF2d;}
    .cython.score-47 {background-color: #FFFF2c;}
    .cython.score-48 {background-color: #FFFF2b;}
    .cython.score-49 {background-color: #FFFF2b;}
    .cython.score-50 {background-color: #FFFF2a;}
    .cython.score-51 {background-color: #FFFF29;}
    .cython.score-52 {background-color: #FFFF29;}
    .cython.score-53 {background-color: #FFFF28;}
    .cython.score-54 {background-color: #FFFF27;}
    .cython.score-55 {background-color: #FFFF27;}
    .cython.score-56 {background-color: #FFFF26;}
    .cython.score-57 {background-color: #FFFF26;}
    .cython.score-58 {background-color: #FFFF25;}
    .cython.score-59 {background-color: #FFFF24;}
    .cython.score-60 {background-color: #FFFF24;}
    .cython.score-61 {background-color: #FFFF23;}
    .cython.score-62 {background-color: #FFFF23;}
    .cython.score-63 {background-color: #FFFF22;}
    .cython.score-64 {background-color: #FFFF22;}
    .cython.score-65 {background-color: #FFFF22;}
    .cython.score-66 {background-color: #FFFF21;}
    .cython.score-67 {background-color: #FFFF21;}
    .cython.score-68 {background-color: #FFFF20;}
    .cython.score-69 {background-color: #FFFF20;}
    .cython.score-70 {background-color: #FFFF1f;}
    .cython.score-71 {background-color: #FFFF1f;}
    .cython.score-72 {background-color: #FFFF1f;}
    .cython.score-73 {background-color: #FFFF1e;}
    .cython.score-74 {background-color: #FFFF1e;}
    .cython.score-75 {background-color: #FFFF1e;}
    .cython.score-76 {background-color: #FFFF1d;}
    .cython.score-77 {background-color: #FFFF1d;}
    .cython.score-78 {background-color: #FFFF1c;}
    .cython.score-79 {background-color: #FFFF1c;}
    .cython.score-80 {background-color: #FFFF1c;}
    .cython.score-81 {background-color: #FFFF1c;}
    .cython.score-82 {background-color: #FFFF1b;}
    .cython.score-83 {background-color: #FFFF1b;}
    .cython.score-84 {background-color: #FFFF1b;}
    .cython.score-85 {background-color: #FFFF1a;}
    .cython.score-86 {background-color: #FFFF1a;}
    .cython.score-87 {background-color: #FFFF1a;}
    .cython.score-88 {background-color: #FFFF1a;}
    .cython.score-89 {background-color: #FFFF19;}
    .cython.score-90 {background-color: #FFFF19;}
    .cython.score-91 {background-color: #FFFF19;}
    .cython.score-92 {background-color: #FFFF19;}
    .cython.score-93 {background-color: #FFFF18;}
    .cython.score-94 {background-color: #FFFF18;}
    .cython.score-95 {background-color: #FFFF18;}
    .cython.score-96 {background-color: #FFFF18;}
    .cython.score-97 {background-color: #FFFF17;}
    .cython.score-98 {background-color: #FFFF17;}
    .cython.score-99 {background-color: #FFFF17;}
    .cython.score-100 {background-color: #FFFF17;}
    .cython.score-101 {background-color: #FFFF16;}
    .cython.score-102 {background-color: #FFFF16;}
    .cython.score-103 {background-color: #FFFF16;}
    .cython.score-104 {background-color: #FFFF16;}
    .cython.score-105 {background-color: #FFFF16;}
    .cython.score-106 {background-color: #FFFF15;}
    .cython.score-107 {background-color: #FFFF15;}
    .cython.score-108 {background-color: #FFFF15;}
    .cython.score-109 {background-color: #FFFF15;}
    .cython.score-110 {background-color: #FFFF15;}
    .cython.score-111 {background-color: #FFFF15;}
    .cython.score-112 {background-color: #FFFF14;}
    .cython.score-113 {background-color: #FFFF14;}
    .cython.score-114 {background-color: #FFFF14;}
    .cython.score-115 {background-color: #FFFF14;}
    .cython.score-116 {background-color: #FFFF14;}
    .cython.score-117 {background-color: #FFFF14;}
    .cython.score-118 {background-color: #FFFF13;}
    .cython.score-119 {background-color: #FFFF13;}
    .cython.score-120 {background-color: #FFFF13;}
    .cython.score-121 {background-color: #FFFF13;}
    .cython.score-122 {background-color: #FFFF13;}
    .cython.score-123 {background-color: #FFFF13;}
    .cython.score-124 {background-color: #FFFF13;}
    .cython.score-125 {background-color: #FFFF12;}
    .cython.score-126 {background-color: #FFFF12;}
    .cython.score-127 {background-color: #FFFF12;}
    .cython.score-128 {background-color: #FFFF12;}
    .cython.score-129 {background-color: #FFFF12;}
    .cython.score-130 {background-color: #FFFF12;}
    .cython.score-131 {background-color: #FFFF12;}
    .cython.score-132 {background-color: #FFFF11;}
    .cython.score-133 {background-color: #FFFF11;}
    .cython.score-134 {background-color: #FFFF11;}
    .cython.score-135 {background-color: #FFFF11;}
    .cython.score-136 {background-color: #FFFF11;}
    .cython.score-137 {background-color: #FFFF11;}
    .cython.score-138 {background-color: #FFFF11;}
    .cython.score-139 {background-color: #FFFF11;}
    .cython.score-140 {background-color: #FFFF11;}
    .cython.score-141 {background-color: #FFFF10;}
    .cython.score-142 {background-color: #FFFF10;}
    .cython.score-143 {background-color: #FFFF10;}
    .cython.score-144 {background-color: #FFFF10;}
    .cython.score-145 {background-color: #FFFF10;}
    .cython.score-146 {background-color: #FFFF10;}
    .cython.score-147 {background-color: #FFFF10;}
    .cython.score-148 {background-color: #FFFF10;}
    .cython.score-149 {background-color: #FFFF10;}
    .cython.score-150 {background-color: #FFFF0f;}
    .cython.score-151 {background-color: #FFFF0f;}
    .cython.score-152 {background-color: #FFFF0f;}
    .cython.score-153 {background-color: #FFFF0f;}
    .cython.score-154 {background-color: #FFFF0f;}
    .cython.score-155 {background-color: #FFFF0f;}
    .cython.score-156 {background-color: #FFFF0f;}
    .cython.score-157 {background-color: #FFFF0f;}
    .cython.score-158 {background-color: #FFFF0f;}
    .cython.score-159 {background-color: #FFFF0f;}
    .cython.score-160 {background-color: #FFFF0f;}
    .cython.score-161 {background-color: #FFFF0e;}
    .cython.score-162 {background-color: #FFFF0e;}
    .cython.score-163 {background-color: #FFFF0e;}
    .cython.score-164 {background-color: #FFFF0e;}
    .cython.score-165 {background-color: #FFFF0e;}
    .cython.score-166 {background-color: #FFFF0e;}
    .cython.score-167 {background-color: #FFFF0e;}
    .cython.score-168 {background-color: #FFFF0e;}
    .cython.score-169 {background-color: #FFFF0e;}
    .cython.score-170 {background-color: #FFFF0e;}
    .cython.score-171 {background-color: #FFFF0e;}
    .cython.score-172 {background-color: #FFFF0e;}
    .cython.score-173 {background-color: #FFFF0d;}
    .cython.score-174 {background-color: #FFFF0d;}
    .cython.score-175 {background-color: #FFFF0d;}
    .cython.score-176 {background-color: #FFFF0d;}
    .cython.score-177 {background-color: #FFFF0d;}
    .cython.score-178 {background-color: #FFFF0d;}
    .cython.score-179 {background-color: #FFFF0d;}
    .cython.score-180 {background-color: #FFFF0d;}
    .cython.score-181 {background-color: #FFFF0d;}
    .cython.score-182 {background-color: #FFFF0d;}
    .cython.score-183 {background-color: #FFFF0d;}
    .cython.score-184 {background-color: #FFFF0d;}
    .cython.score-185 {background-color: #FFFF0d;}
    .cython.score-186 {background-color: #FFFF0d;}
    .cython.score-187 {background-color: #FFFF0c;}
    .cython.score-188 {background-color: #FFFF0c;}
    .cython.score-189 {background-color: #FFFF0c;}
    .cython.score-190 {background-color: #FFFF0c;}
    .cython.score-191 {background-color: #FFFF0c;}
    .cython.score-192 {background-color: #FFFF0c;}
    .cython.score-193 {background-color: #FFFF0c;}
    .cython.score-194 {background-color: #FFFF0c;}
    .cython.score-195 {background-color: #FFFF0c;}
    .cython.score-196 {background-color: #FFFF0c;}
    .cython.score-197 {background-color: #FFFF0c;}
    .cython.score-198 {background-color: #FFFF0c;}
    .cython.score-199 {background-color: #FFFF0c;}
    .cython.score-200 {background-color: #FFFF0c;}
    .cython.score-201 {background-color: #FFFF0c;}
    .cython.score-202 {background-color: #FFFF0c;}
    .cython.score-203 {background-color: #FFFF0b;}
    .cython.score-204 {background-color: #FFFF0b;}
    .cython.score-205 {background-color: #FFFF0b;}
    .cython.score-206 {background-color: #FFFF0b;}
    .cython.score-207 {background-color: #FFFF0b;}
    .cython.score-208 {background-color: #FFFF0b;}
    .cython.score-209 {background-color: #FFFF0b;}
    .cython.score-210 {background-color: #FFFF0b;}
    .cython.score-211 {background-color: #FFFF0b;}
    .cython.score-212 {background-color: #FFFF0b;}
    .cython.score-213 {background-color: #FFFF0b;}
    .cython.score-214 {background-color: #FFFF0b;}
    .cython.score-215 {background-color: #FFFF0b;}
    .cython.score-216 {background-color: #FFFF0b;}
    .cython.score-217 {background-color: #FFFF0b;}
    .cython.score-218 {background-color: #FFFF0b;}
    .cython.score-219 {background-color: #FFFF0b;}
    .cython.score-220 {background-color: #FFFF0b;}
    .cython.score-221 {background-color: #FFFF0b;}
    .cython.score-222 {background-color: #FFFF0a;}
    .cython.score-223 {background-color: #FFFF0a;}
    .cython.score-224 {background-color: #FFFF0a;}
    .cython.score-225 {background-color: #FFFF0a;}
    .cython.score-226 {background-color: #FFFF0a;}
    .cython.score-227 {background-color: #FFFF0a;}
    .cython.score-228 {background-color: #FFFF0a;}
    .cython.score-229 {background-color: #FFFF0a;}
    .cython.score-230 {background-color: #FFFF0a;}
    .cython.score-231 {background-color: #FFFF0a;}
    .cython.score-232 {background-color: #FFFF0a;}
    .cython.score-233 {background-color: #FFFF0a;}
    .cython.score-234 {background-color: #FFFF0a;}
    .cython.score-235 {background-color: #FFFF0a;}
    .cython.score-236 {background-color: #FFFF0a;}
    .cython.score-237 {background-color: #FFFF0a;}
    .cython.score-238 {background-color: #FFFF0a;}
    .cython.score-239 {background-color: #FFFF0a;}
    .cython.score-240 {background-color: #FFFF0a;}
    .cython.score-241 {background-color: #FFFF0a;}
    .cython.score-242 {background-color: #FFFF0a;}
    .cython.score-243 {background-color: #FFFF0a;}
    .cython.score-244 {background-color: #FFFF0a;}
    .cython.score-245 {background-color: #FFFF0a;}
    .cython.score-246 {background-color: #FFFF09;}
    .cython.score-247 {background-color: #FFFF09;}
    .cython.score-248 {background-color: #FFFF09;}
    .cython.score-249 {background-color: #FFFF09;}
    .cython.score-250 {background-color: #FFFF09;}
    .cython.score-251 {background-color: #FFFF09;}
    .cython.score-252 {background-color: #FFFF09;}
    .cython.score-253 {background-color: #FFFF09;}
    .cython.score-254 {background-color: #FFFF09;}.cython .hll { background-color: #ffffcc }
    .cython  { background: #f8f8f8; }
    .cython .c { color: #408080; font-style: italic } /* Comment */
    .cython .err { border: 1px solid #FF0000 } /* Error */
    .cython .k { color: #008000; font-weight: bold } /* Keyword */
    .cython .o { color: #666666 } /* Operator */
    .cython .cm { color: #408080; font-style: italic } /* Comment.Multiline */
    .cython .cp { color: #BC7A00 } /* Comment.Preproc */
    .cython .c1 { color: #408080; font-style: italic } /* Comment.Single */
    .cython .cs { color: #408080; font-style: italic } /* Comment.Special */
    .cython .gd { color: #A00000 } /* Generic.Deleted */
    .cython .ge { font-style: italic } /* Generic.Emph */
    .cython .gr { color: #FF0000 } /* Generic.Error */
    .cython .gh { color: #000080; font-weight: bold } /* Generic.Heading */
    .cython .gi { color: #00A000 } /* Generic.Inserted */
    .cython .go { color: #888888 } /* Generic.Output */
    .cython .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
    .cython .gs { font-weight: bold } /* Generic.Strong */
    .cython .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
    .cython .gt { color: #0044DD } /* Generic.Traceback */
    .cython .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
    .cython .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
    .cython .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
    .cython .kp { color: #008000 } /* Keyword.Pseudo */
    .cython .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
    .cython .kt { color: #B00040 } /* Keyword.Type */
    .cython .m { color: #666666 } /* Literal.Number */
    .cython .s { color: #BA2121 } /* Literal.String */
    .cython .na { color: #7D9029 } /* Name.Attribute */
    .cython .nb { color: #008000 } /* Name.Builtin */
    .cython .nc { color: #0000FF; font-weight: bold } /* Name.Class */
    .cython .no { color: #880000 } /* Name.Constant */
    .cython .nd { color: #AA22FF } /* Name.Decorator */
    .cython .ni { color: #999999; font-weight: bold } /* Name.Entity */
    .cython .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
    .cython .nf { color: #0000FF } /* Name.Function */
    .cython .nl { color: #A0A000 } /* Name.Label */
    .cython .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
    .cython .nt { color: #008000; font-weight: bold } /* Name.Tag */
    .cython .nv { color: #19177C } /* Name.Variable */
    .cython .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
    .cython .w { color: #bbbbbb } /* Text.Whitespace */
    .cython .mf { color: #666666 } /* Literal.Number.Float */
    .cython .mh { color: #666666 } /* Literal.Number.Hex */
    .cython .mi { color: #666666 } /* Literal.Number.Integer */
    .cython .mo { color: #666666 } /* Literal.Number.Oct */
    .cython .sb { color: #BA2121 } /* Literal.String.Backtick */
    .cython .sc { color: #BA2121 } /* Literal.String.Char */
    .cython .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
    .cython .s2 { color: #BA2121 } /* Literal.String.Double */
    .cython .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
    .cython .sh { color: #BA2121 } /* Literal.String.Heredoc */
    .cython .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
    .cython .sx { color: #008000 } /* Literal.String.Other */
    .cython .sr { color: #BB6688 } /* Literal.String.Regex */
    .cython .s1 { color: #BA2121 } /* Literal.String.Single */
    .cython .ss { color: #19177C } /* Literal.String.Symbol */
    .cython .bp { color: #008000 } /* Name.Builtin.Pseudo */
    .cython .vc { color: #19177C } /* Name.Variable.Class */
    .cython .vg { color: #19177C } /* Name.Variable.Global */
    .cython .vi { color: #19177C } /* Name.Variable.Instance */
    .cython .il { color: #666666 } /* Literal.Number.Integer.Long */
        </style>
        <script>
        function toggleDiv(id) {
            theDiv = id.nextElementSibling
            if (theDiv.style.display != 'block') theDiv.style.display = 'block';
            else theDiv.style.display = 'none';
        }
        </script>
    </head>
    <body class="cython">
    <p>Generated by Cython 0.22</p>
    <div class="cython"><pre class='cython line score-11' onclick='toggleDiv(this)'>+01: <span class="k">import</span> <span class="nn">cython</span></pre>
    <pre class='cython code score-11'>  __pyx_t_1 = <span class='py_c_api'>PyDict_New</span>();<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_test, __pyx_t_1) &lt; 0) <span class='error_goto'>{__pyx_filename = __pyx_f[0]; __pyx_lineno = 1; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    </pre><pre class='cython line score-8' onclick='toggleDiv(this)'>+02: <span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
    <pre class='cython code score-8'>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_Import</span>(__pyx_n_s_numpy, 0, -1);<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 2; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_np, __pyx_t_1) &lt; 0) <span class='error_goto'>{__pyx_filename = __pyx_f[0]; __pyx_lineno = 2; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
    </pre><pre class='cython line score-0'>&#xA0;03: <span class="k">cimport</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></pre>
    <pre class='cython line score-0'>&#xA0;04: <span class="k">from</span> <span class="nn">cython_gsl</span> <span class="k">cimport</span> <span class="n">gsl_rng_mt19937</span><span class="p">,</span> <span class="n">gsl_rng</span><span class="p">,</span> <span class="n">gsl_rng_alloc</span><span class="p">,</span> <span class="n">gsl_rng_uniform</span></pre>
    <pre class='cython line score-0'>&#xA0;05: </pre>
    <pre class='cython line score-0' onclick='toggleDiv(this)'>+06: <span class="k">cdef</span> <span class="kt">gsl_rng</span> *<span class="nf">r</span> <span class="o">=</span> <span class="n">gsl_rng_alloc</span><span class="p">(</span><span class="n">gsl_rng_mt19937</span><span class="p">)</span></pre>
    <pre class='cython code score-0'>  __pyx_v_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_r = gsl_rng_alloc(gsl_rng_mt19937);
    </pre><pre class='cython line score-0'>&#xA0;07: </pre>
    <pre class='cython line score-0'>&#xA0;08: <span class="nd">@cython</span><span class="o">.</span><span class="n">cdivision</span></pre>
    <pre class='cython line score-0'>&#xA0;09: <span class="nd">@cython</span><span class="o">.</span><span class="n">boundscheck</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
    <pre class='cython line score-0'>&#xA0;10: <span class="nd">@cython</span><span class="o">.</span><span class="n">wraparound</span><span class="p">(</span><span class="bp">False</span><span class="p">)</span></pre>
    <pre class='cython line score-24' onclick='toggleDiv(this)'>+11: <span class="k">def</span> <span class="nf">pi_cython</span><span class="p">(</span><span class="nb">int</span> <span class="n">n</span><span class="p">):</span></pre>
    <pre class='cython code score-24'>/* Python wrapper */
    static PyObject *__pyx_pw_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_1pi_cython(PyObject *__pyx_self, PyObject *__pyx_arg_n); /*proto*/
    static PyMethodDef __pyx_mdef_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_1pi_cython = {"pi_cython", (PyCFunction)__pyx_pw_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_1pi_cython, METH_O, 0};
    static PyObject *__pyx_pw_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_1pi_cython(PyObject *__pyx_self, PyObject *__pyx_arg_n) {
      int __pyx_v_n;
      PyObject *__pyx_r = 0;
      <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
      <span class='refnanny'>__Pyx_RefNannySetupContext</span>("pi_cython (wrapper)", 0);
      assert(__pyx_arg_n); {
        __pyx_v_n = <span class='pyx_c_api'>__Pyx_PyInt_As_int</span>(__pyx_arg_n);<span class='error_goto'> if (unlikely((__pyx_v_n == (int)-1) &amp;&amp; PyErr_Occurred())) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 11; __pyx_clineno = __LINE__; goto __pyx_L3_error;}</span>
      }
      goto __pyx_L4_argument_unpacking_done;
      __pyx_L3_error:;
      <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_f0d9ca082c7b25a08598049bfef2323c.pi_cython", __pyx_clineno, __pyx_lineno, __pyx_filename);
      <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
      return NULL;
      __pyx_L4_argument_unpacking_done:;
      __pyx_r = __pyx_pf_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_pi_cython(__pyx_self, ((int)__pyx_v_n));
      int __pyx_lineno = 0;
      const char *__pyx_filename = NULL;
      int __pyx_clineno = 0;
    
      /* function exit code */
      <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
      return __pyx_r;
    }
    
    static PyObject *__pyx_pf_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_pi_cython(CYTHON_UNUSED PyObject *__pyx_self, int __pyx_v_n) {
      __Pyx_memviewslice __pyx_v_s = { 0, 0, { 0 }, { 0 }, { 0 } };
      int __pyx_v_i;
      double __pyx_v_x;
      double __pyx_v_y;
      int __pyx_v_hits;
      PyObject *__pyx_r = NULL;
      <span class='refnanny'>__Pyx_RefNannyDeclarations</span>
      <span class='refnanny'>__Pyx_RefNannySetupContext</span>("pi_cython", 0);
    /* … */
      /* function exit code */
      __pyx_L1_error:;
      <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_1);
      <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_2);
      <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_3);
      <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_4);
      <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_t_5);
      __PYX_XDEC_MEMVIEW(&amp;__pyx_t_6, 1);
      <span class='pyx_c_api'>__Pyx_AddTraceback</span>("_cython_magic_f0d9ca082c7b25a08598049bfef2323c.pi_cython", __pyx_clineno, __pyx_lineno, __pyx_filename);
      __pyx_r = NULL;
      __pyx_L0:;
      __PYX_XDEC_MEMVIEW(&amp;__pyx_v_s, 1);
      <span class='refnanny'>__Pyx_XGIVEREF</span>(__pyx_r);
      <span class='refnanny'>__Pyx_RefNannyFinishContext</span>();
      return __pyx_r;
    }
    /* … */
      __pyx_tuple__19 = <span class='py_c_api'>PyTuple_Pack</span>(7, __pyx_n_s_n, __pyx_n_s_n, __pyx_n_s_s, __pyx_n_s_i, __pyx_n_s_x, __pyx_n_s_y, __pyx_n_s_hits);<span class='error_goto'> if (unlikely(!__pyx_tuple__19)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 11; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_tuple__19);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_tuple__19);
    /* … */
      __pyx_t_1 = PyCFunction_NewEx(&amp;__pyx_mdef_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_1pi_cython, NULL, __pyx_n_s_cython_magic_f0d9ca082c7b25a085);<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 11; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_d, __pyx_n_s_pi_cython, __pyx_t_1) &lt; 0) <span class='error_goto'>{__pyx_filename = __pyx_f[0]; __pyx_lineno = 11; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
      __pyx_codeobj__20 = (PyObject*)<span class='pyx_c_api'>__Pyx_PyCode_New</span>(1, 0, 7, 0, 0, __pyx_empty_bytes, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_tuple__19, __pyx_empty_tuple, __pyx_empty_tuple, __pyx_kp_s_Users_cliburn_ipython_cython__c, __pyx_n_s_pi_cython, 11, __pyx_empty_bytes);<span class='error_goto'> if (unlikely(!__pyx_codeobj__20)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 11; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
    </pre><pre class='cython line score-37' onclick='toggleDiv(this)'>+12:     <span class="k">cdef</span> <span class="kt">int</span>[<span class="p">:]</span> <span class="n">s</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int32</span><span class="p">)</span></pre>
    <pre class='cython code score-37'>  __pyx_t_1 = <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      __pyx_t_2 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_1, __pyx_n_s_zeros);<span class='error_goto'> if (unlikely(!__pyx_t_2)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_2);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
      __pyx_t_1 = <span class='pyx_c_api'>__Pyx_PyInt_From_int</span>(__pyx_v_n);<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      __pyx_t_3 = <span class='py_c_api'>PyTuple_New</span>(1);<span class='error_goto'> if (unlikely(!__pyx_t_3)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_3);
      <span class='py_macro_api'>PyTuple_SET_ITEM</span>(__pyx_t_3, 0, __pyx_t_1);
      <span class='refnanny'>__Pyx_GIVEREF</span>(__pyx_t_1);
      __pyx_t_1 = 0;
      __pyx_t_1 = <span class='py_c_api'>PyDict_New</span>();<span class='error_goto'> if (unlikely(!__pyx_t_1)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_1);
      __pyx_t_4 = <span class='pyx_c_api'>__Pyx_GetModuleGlobalName</span>(__pyx_n_s_np);<span class='error_goto'> if (unlikely(!__pyx_t_4)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_4);
      __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_GetAttrStr</span>(__pyx_t_4, __pyx_n_s_int32);<span class='error_goto'> if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_4); __pyx_t_4 = 0;
      if (<span class='py_c_api'>PyDict_SetItem</span>(__pyx_t_1, __pyx_n_s_dtype, __pyx_t_5) &lt; 0) <span class='error_goto'>{__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_t_5 = <span class='pyx_c_api'>__Pyx_PyObject_Call</span>(__pyx_t_2, __pyx_t_3, __pyx_t_1);<span class='error_goto'> if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_2); __pyx_t_2 = 0;
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_3); __pyx_t_3 = 0;
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_1); __pyx_t_1 = 0;
      __pyx_t_6 = <span class='pyx_c_api'>__Pyx_PyObject_to_MemoryviewSlice_ds_int</span>(__pyx_t_5);
      if (unlikely(!__pyx_t_6.memview)) <span class='error_goto'>{__pyx_filename = __pyx_f[0]; __pyx_lineno = 12; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='pyx_macro_api'>__Pyx_DECREF</span>(__pyx_t_5); __pyx_t_5 = 0;
      __pyx_v_s = __pyx_t_6;
      __pyx_t_6.memview = NULL;
      __pyx_t_6.data = NULL;
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+13:     <span class="k">cdef</span> <span class="kt">int</span> <span class="nf">i</span> <span class="o">=</span> <span class="mf">0</span></pre>
    <pre class='cython code score-0'>  __pyx_v_i = 0;
    </pre><pre class='cython line score-0'>&#xA0;14:     <span class="k">cdef</span> <span class="kt">double</span> <span class="nf">x</span><span class="p">,</span> <span class="nf">y</span></pre>
    <pre class='cython line score-0' onclick='toggleDiv(this)'>+15:     <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span></pre>
    <pre class='cython code score-0'>  __pyx_t_7 = __pyx_v_n;
      for (__pyx_t_8 = 0; __pyx_t_8 &lt; __pyx_t_7; __pyx_t_8+=1) {
        __pyx_v_i = __pyx_t_8;
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+16:         <span class="n">x</span> <span class="o">=</span> <span class="n">gsl_rng_uniform</span><span class="p">(</span><span class="n">r</span><span class="p">)</span><span class="o">*</span><span class="mf">2</span> <span class="o">-</span> <span class="mf">1</span></pre>
    <pre class='cython code score-0'>    __pyx_v_x = ((gsl_rng_uniform(__pyx_v_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_r) * 2.0) - 1.0);
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+17:         <span class="n">y</span> <span class="o">=</span> <span class="n">gsl_rng_uniform</span><span class="p">(</span><span class="n">r</span><span class="p">)</span><span class="o">*</span><span class="mf">2</span> <span class="o">-</span> <span class="mf">1</span></pre>
    <pre class='cython code score-0'>    __pyx_v_y = ((gsl_rng_uniform(__pyx_v_46_cython_magic_f0d9ca082c7b25a08598049bfef2323c_r) * 2.0) - 1.0);
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+18:         <span class="n">s</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">x</span><span class="o">**</span><span class="mf">2</span> <span class="o">+</span> <span class="n">y</span><span class="o">**</span><span class="mf">2</span> <span class="o">&lt;</span> <span class="mf">1</span></pre>
    <pre class='cython code score-0'>    __pyx_t_9 = __pyx_v_i;
        *((int *) ( /* dim=0 */ (__pyx_v_s.data + __pyx_t_9 * __pyx_v_s.strides[0]) )) = ((pow(__pyx_v_x, 2.0) + pow(__pyx_v_y, 2.0)) &lt; 1.0);
      }
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+19:     <span class="k">cdef</span> <span class="kt">int</span> <span class="nf">hits</span> <span class="o">=</span> <span class="mf">0</span></pre>
    <pre class='cython code score-0'>  __pyx_v_hits = 0;
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+20:     <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">n</span><span class="p">):</span></pre>
    <pre class='cython code score-0'>  __pyx_t_7 = __pyx_v_n;
      for (__pyx_t_8 = 0; __pyx_t_8 &lt; __pyx_t_7; __pyx_t_8+=1) {
        __pyx_v_i = __pyx_t_8;
    </pre><pre class='cython line score-0' onclick='toggleDiv(this)'>+21:         <span class="n">hits</span> <span class="o">+=</span> <span class="n">s</span><span class="p">[</span><span class="n">i</span><span class="p">]</span></pre>
    <pre class='cython code score-0'>    __pyx_t_10 = __pyx_v_i;
        __pyx_v_hits = (__pyx_v_hits + (*((int *) ( /* dim=0 */ (__pyx_v_s.data + __pyx_t_10 * __pyx_v_s.strides[0]) ))));
      }
    </pre><pre class='cython line score-6' onclick='toggleDiv(this)'>+22:     <span class="k">return</span> <span class="mf">4.0</span><span class="o">*</span><span class="n">hits</span><span class="o">/</span><span class="n">n</span></pre>
    <pre class='cython code score-6'>  <span class='pyx_macro_api'>__Pyx_XDECREF</span>(__pyx_r);
      __pyx_t_5 = <span class='py_c_api'>PyFloat_FromDouble</span>(((4.0 * __pyx_v_hits) / __pyx_v_n));<span class='error_goto'> if (unlikely(!__pyx_t_5)) {__pyx_filename = __pyx_f[0]; __pyx_lineno = 22; __pyx_clineno = __LINE__; goto __pyx_L1_error;}</span>
      <span class='refnanny'>__Pyx_GOTREF</span>(__pyx_t_5);
      __pyx_r = __pyx_t_5;
      __pyx_t_5 = 0;
      goto __pyx_L0;
    </pre></div></body></html>



.. code:: python

    n = int(1e5)
    %timeit pi_python(n)
    %timeit pi_numba(n)
    %timeit pi_numpy(n)
    %timeit pi_cython(n)


.. parsed-literal::

    10 loops, best of 3: 127 ms per loop
    1 loops, best of 3: 146 ms per loop
    100 loops, best of 3: 5.18 ms per loop
    100 loops, best of 3: 1.95 ms per loop


**The bigger the problem, the more scope there is for parallelism**

**Amhdahls' law** says that the speedup from parallelization is bounded
by the ratio of parallelizable to irreducibly serial code in the
aloorithm. However, for big data analysis, **Gustafson's Law** is more
relevant. This says that we are nearly always interested in increasing
the size of the parallelizable bits, and the ratio of parallelizable to
irreducibly serial code is not a static quantity but depends on data
size. For example, Gibbs sampling has an irreducibly serial nature, but
for large samples, each iteration may be able perform PDF evaluations in
parallel for zillions of data points.

Using Multiprocessing
---------------------

-  `Documentation <https://docs.python.org/2/library/multiprocessing.html>`__
-  `Tutorial - kerndel density esitmation with
   multiprocessing <http://sebastianraschka.com/Articles/2014_multiprocessing_intro.html>`__

.. code:: python

    import multiprocessing
    
    num_procs = multiprocessing.cpu_count()
    num_procs




.. parsed-literal::

    4



.. code:: python

    def pi_multiprocessing(n):
        """Split a job of length n into num_procs pieces."""
        import multiprocessing
        m = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(m)
        results = pool.map(pi_cython, [n/m]*m)
        pool.close()
        return np.mean(results)

For small jobs, the cost of spawning processes dominates

.. code:: python

    n = int(1e5)
    %timeit pi_cython(n)
    %timeit pi_multiprocessing(n)


.. parsed-literal::

    100 loops, best of 3: 1.95 ms per loop
    10 loops, best of 3: 32.6 ms per loop


For larger jobs, we see the expected linear speedup

.. code:: python

    n = int(1e7)
    %timeit pi_numpy(n)
    %timeit pi_multiprocessing(n)


.. parsed-literal::

    1 loops, best of 3: 718 ms per loop
    10 loops, best of 3: 148 ms per loop


Communication across parallel workers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all tasks are embarassingly parallel. In these problems, we need to
communicate across parallel workers. There are two ways to do this - via
shared memory (exemplar is OpenMP) and by explicit communication
mechanisms (exemplar is MPI). Multiprocessing (and GPU computing) can
use both mechanisms.

See `MOTW <http://pymotw.com/2/multiprocessing/communication.html>`__
for examples of communicating across processes with multiprocessing.

**Using shared memory can lead to race conditions**

.. code:: python

    from multiprocessing import Pool, Value, Array, Lock, current_process
    
    n = 4
    val = Value('i')
    arr = Array('i', n)
    
    val.value = 0
    for i in range(n):
        arr[i] = 0
    
    def count1(i):
        "Everyone competes to write to val."""
        val.value += 1
        
    def count2(i):
        """Each process has its own slot in arr to write to."""
        ix = current_process().pid % n
        arr[ix] += 1
        
    pool = Pool(n)
    pool.map(count1, range(1000))
    pool.map(count2, range(1000))
    
    pool.close()
    print val.value
    print sum(arr)


.. parsed-literal::

    500
    1000


Using IPython parallel for interactive parallel computing
---------------------------------------------------------

Start a cluster of workers using IPython notebook interface.
Alternatively, enter

``ipcluster start -n 4``

at the command line.

.. code:: python

    from IPython.parallel import Client, interactive

**Direct view**

.. code:: python

    rc = Client()
    print rc.ids
    dv = rc[:]


.. parsed-literal::

    [0, 1, 2, 3]


The %%px (parallel execute) magic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a cell is marked with %%px, all commands in that cell get executed
on all engines simultaneously. We'll use it to load ``numpy`` on all
engines.

.. code:: python

    %px import numpy as np

We can refer to indivudal engines using indexing and slice notation on
the client - for example, to set random seeds.

.. code:: python

    for i, r in enumerate(rc):
        r.execute('np.random.seed(123)')

.. code:: python

    %%px
    
    np.random.random(3)



.. parsed-literal::

    Out[0:2]: array([ 0.69646919,  0.28613933,  0.22685145])



.. parsed-literal::

    Out[1:2]: array([ 0.69646919,  0.28613933,  0.22685145])



.. parsed-literal::

    Out[2:2]: array([ 0.69646919,  0.28613933,  0.22685145])



.. parsed-literal::

    Out[3:2]: array([ 0.69646919,  0.28613933,  0.22685145])


Another way to do this is via the ``scatter`` operation.

.. code:: python

    dv.scatter('seed', [1,1,2,2], block=True)

.. code:: python

    dv['seed']




.. parsed-literal::

    [[1], [1], [2], [2]]



.. code:: python

    %%px 
    
    np.random.seed(seed)
    np.random.random(3)



.. parsed-literal::

    Out[0:3]: array([ 0.13436424,  0.84743374,  0.76377462])



.. parsed-literal::

    Out[1:3]: array([ 0.13436424,  0.84743374,  0.76377462])



.. parsed-literal::

    Out[2:3]: array([ 0.95603427,  0.94782749,  0.05655137])



.. parsed-literal::

    Out[3:3]: array([ 0.95603427,  0.94782749,  0.05655137])


We set them to differnet seeds again to do the Monte Carlo integration.

.. code:: python

    for i, r in enumerate(rc):
        r.execute('np.random.seed(%d)' % i)

.. code:: python

    %%px 
    
    np.random.random(3)



.. parsed-literal::

    Out[0:4]: array([ 0.5488135 ,  0.71518937,  0.60276338])



.. parsed-literal::

    Out[1:4]: array([  4.17022005e-01,   7.20324493e-01,   1.14374817e-04])



.. parsed-literal::

    Out[2:4]: array([ 0.4359949 ,  0.02592623,  0.54966248])



.. parsed-literal::

    Out[3:4]: array([ 0.5507979 ,  0.70814782,  0.29090474])


We can collect the individual results of remote computation using a
dictionary lookup syntax or use ``gather`` to concatenate the resutls.

.. code:: python

    %%px
    
    x = np.random.random(3)

.. code:: python

    dv['x']




.. parsed-literal::

    [array([ 0.5449,  0.4237,  0.6459]),
     array([ 0.3023,  0.1468,  0.0923]),
     array([ 0.4353,  0.4204,  0.3303]),
     array([ 0.5108,  0.8929,  0.8963])]



.. code:: python

    dv.gather('x', block=True)




.. parsed-literal::

    array([ 0.5449,  0.4237,  0.6459,  0.3023,  0.1468,  0.0923,  0.4353,
            0.4204,  0.3303,  0.5108,  0.8929,  0.8963])



Finding :math:`\pi` simply involves generating random uniforms on each
processor.

.. code:: python

    %%px
    n = 1e7
    x = np.random.uniform(-1, 1, (n, 2))
    n = (x[:, 0]**2 + x[:,1]**2 < 1).sum()

.. code:: python

    %precision 8
    ns = dv['n']
    4*np.sum(ns)/(1e7*len(rc))




.. parsed-literal::

    3.14143780



Blocking and non-blocking operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In blocking mode (the default), operations on remote engines do not
return until all compuations are done. For long computations, this may
be undesirable and we can ask the engine to return immeidately by using
a non-blocking operation. In this case, what is returned is an Async
type object, which we can query for whether the computation is complete
and if so, retrieve data from it.

.. code:: python

    dv.scatter('s', np.arange(16), block=False)




.. parsed-literal::

    <AsyncResult: scatter>



.. code:: python

    dv['s']




.. parsed-literal::

    [array([0, 1, 2, 3]),
     array([4, 5, 6, 7]),
     array([ 8,  9, 10, 11]),
     array([12, 13, 14, 15])]



.. code:: python

    dv.gather('s')




.. parsed-literal::

    <AsyncMapResult: gather>



.. code:: python

    dv.gather('s').get()




.. parsed-literal::

    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])



.. code:: python

    ar = dv.map_async(lambda x: x+1, range(10))
    ar.ready()




.. parsed-literal::

    False



.. code:: python

    ar.ready()




.. parsed-literal::

    False



.. code:: python

    ar.get()




.. parsed-literal::

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



Load-balanced view
^^^^^^^^^^^^^^^^^^

Sometimes the tasks are very unbalanced - some may complete in a short
time, while others ay take a long time. In this case, using a
load\_balanced view is more efficient to avoid the risk that a single
engine gets allocated all the long-running tasks.

.. code:: python

    lv = rc.load_balanced_view()

.. code:: python

    def wait(n):
        import time
        time.sleep(n)
        return n
    
    dv['wait'] = wait

.. code:: python

    intervals = [5,1,1,1,1,1,1,1,1,1,1,1,1,5,5,5]

.. code:: python

    %%time
    
    ar = dv.map(wait, intervals)
    ar.get()


.. parsed-literal::

    CPU times: user 2.75 s, sys: 723 ms, total: 3.47 s
    Wall time: 16 s


.. code:: python

    %%time
    
    ar = lv.map(wait, intervals, balanced=True)
    ar.get()


.. parsed-literal::

    CPU times: user 1.7 s, sys: 459 ms, total: 2.16 s
    Wall time: 9.1 s


Using Cython with IPython parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to %load\_ext cythonmagic in every engine, and compile the
cython extension in every engine. the simplest way is to do all this in
a %%px cell.

.. code:: python

    %%px
    def python_loop(xs):
        s = 0.0
        for i in range(len(xs)):
            s += xs[i]
        return s

.. code:: python

    %%px
    %load_ext cythonmagic

.. code:: python

    %%px
    %%cython
    
    def cython_loop(double[::1] xs):
        n = xs.shape[0]
        cdef int i
        cdef double s = 0.0
        for i in range(n):
            s += xs[i]
        return s

.. code:: python

    %%time
    %%px
    xs = np.random.random(1e7)
    s = python_loop(xs)


.. parsed-literal::

    CPU times: user 900 ms, sys: 195 ms, total: 1.1 s
    Wall time: 9.12 s


.. code:: python

    dv['s']




.. parsed-literal::

    [4999255.51979800, 5001207.17286485, 5000816.40605527, 4999437.17107215]



.. code:: python

    %%time
    %%px
    xs = np.random.random(1e7)
    s = cython_loop(xs)


.. parsed-literal::

    CPU times: user 37.3 ms, sys: 7.5 ms, total: 44.8 ms
    Wall time: 376 ms


.. code:: python

    dv['s']




.. parsed-literal::

    [5000927.33063748, 4999180.32360687, 5000671.20938849, 4999140.47559244]



Other parallel programming approaches not covered
-------------------------------------------------

-  `MPI: Message Passing Interface <http://www-unix.mcs.anl.gov/mpi/>`__
-  `mpi4py: MPI for Python <http://mpi4py.scipy.org/>`__
-  `OpenMPI: Open MPI <http://www.open-mpi.org/>`__

References
----------

-  `Parallel Processing in
   Python <http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section7_2-Parallel-Processing.ipynb>`__
-  `Tools for high-performance computing
   applications <http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-6B-HPC.ipynb>`__
-  `Using IPython for Parallel
   Computing <Using%20IPython%20for%20parallel%20computing>`__

.. code:: python

    %load_ext version_information

.. code:: python

    %version_information numba, multiprocessing, cython




.. raw:: html

    <table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.9 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>2.2.0</td></tr><tr><td>OS</td><td>Darwin 13.4.0 x86_64 i386 64bit</td></tr><tr><td>numba</td><td>0.17.0</td></tr><tr><td>multiprocessing</td><td>0.70a1</td></tr><tr><td>cython</td><td>0.22</td></tr><tr><td colspan='2'>Thu Mar 26 16:49:30 2015 EDT</td></tr></table>



