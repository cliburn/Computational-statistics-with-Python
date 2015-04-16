
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


Code Optimization
=================

There is a traditional sequence for writing code, and it goes like this:

1. Make it run
2. Make it right (testing)
3. Make it fast (optimization)

Making it fast is the last step, and you should only optimize when it is
necessary. Also, it is good to know when a program is "fast enough" for
your needs. Optimization has a price:

1. Cost in programmer time
2. Optimized code is often more complex
3. Optimized code is oftne less generic

However, having fast code is often necessary for statistical computing,
so we will spend some time learning how to make code run faster. To do
so, we need to understand why our code is slow: Code can be slow because
of differnet resource limitations:

1. CPU-bound - CPU is working flat out
2. Memory-bound - Out of RAM - swapping to hard disk
3. IO-bound - Lots of data transfer to and from hard disk
4. Network-bound - CPU is waiting for data to come over network or from
   memory ("starvation")

Different bottlenekcs may require different appraoches. However, theere
is a natural order to making code fast

1. Cheat

   -  Use a better machine (e.g. if RAM is limititg is - buy more RAM)
   -  Solve a simpler problem (e.g. will a subsample of the data
      suffice?)
   -  Solve a diffrent problem (perhaps solving a toy problem will
      suffice for your JASA paper? If your method is so useful, maybe
      someone else will optimize it for you)

2. Find out what is slowing down the code (profiling)

   -  Using ``timeit``
   -  Using ``time``
   -  Usign ``cProfile``
   -  Using ``line_profiler``
   -  Using ``memory_profiler``

3. Use better algorithms and data structures
4. Using compiled code written in another language

   -  Calling code written in C/C++

      -  Using ``bitey``
      -  Using ``ctypes``
      -  Using ``cython``

   -  Calling code written in Fotran

      -  Using ``f2py``

   -  Calling code written in Julia

      -  Usign ``pyjulia``

5. Converting Python code to compiled code

   -  Using ``numexpr``
   -  Using ``numba``
   -  Using ``cython``

6. Parallel programs

   -  Ahmdahl and Gustafsson's laws
   -  Embarassinlgy parallel problems
   -  Problems requiring communiccation and syncrhonization

      -  Race conditions
      -  Deadlock

   -  Task granularity
   -  Parallel programming idioms

7. Execute in parallel

   -  On multi-core machines
   -  On multiple machines

      -  Using IPython
      -  Using MPI4py
      -  Using Hadoop/SPARK

   -  On GPUs

Profiling
---------

Profiling means to time your code so as to identify bottelnecks. If one
function is taking up 99% of the time in your program, it is sensiblt to
focus on optimizign that function first. It is a truism in computer
science that we are generally hopeless at guessing what the bottlenecks
in complex programs are, so we need to make use of profiling tools to
help us.

Install profling tools:

.. code:: bash

    pip install --pre line-profiler
    pip install psutil
    pip install memory_profiler

References:

1. http://scipy-lectures.github.com/advanced/optimizing/index.html
2. http://pynash.org/2013/03/06/timing-and-profiling.html

.. code:: python

    ! pip install --pre line-profiler &> /dev/null
    ! pip install psutil &> /dev/null
    ! pip install memory_profiler &> /dev/null

Create an Ipython profile

::

    $ ipython profile create

Add the exntesions to ``.ipython/profile_default/ipython_config.py``

::

    c.TerminalIPythonApp.extensions = [
        'line_profiler',
        'memory_profiler',
    ]

Using the timeit modules
~~~~~~~~~~~~~~~~~~~~~~~~

We can measure the time taken by an arbitrary code block by starting
timers before and after the code block, and measuring the difference.

.. code:: python

    def f(nsec=1.0):
        """Function sleeps for nsec seconds."""
        import time
        time.sleep(nsec) 

.. code:: python

    import timeit
    
    start = timeit.default_timer()
    f()
    elapsed = timeit.default_timer() - start
    elapsed




.. parsed-literal::

    1.0014



In the IPython notebook, individual functions can also be timed using
%timeit. Useful options to %timeit include

-  -n: execute the given statement times in a loop. If this value is not
   given, a fitting value is chosen.
-  -r: repeat the loop iteration times and take the best result.
   Default: 3

.. code:: python

    %timeit f(0.5)


.. parsed-literal::

    1 loops, best of 3: 500 ms per loop


.. code:: python

    %timeit -n2 -r4 f(0.5)


.. parsed-literal::

    2 loops, best of 4: 501 ms per loop


We can also measure the time to execute an entire cell with %%time -
this provdes 3 readouts:

-  Wall time - time from start to finish of the call. This is all
   elapsed time including time slices used by other processes and time
   the process spends blocked (for example if it is waiting for I/O to
   complete).

-  User is the amount of CPU time spent in executing the process,
   excluing operating system (kernel) calls.

-  Sys is the executing CPU time spent in system calls within the
   kernel, as opposed to library code.

.. code:: python

    %%time
    
    f(1)
    f(0.5)


.. parsed-literal::

    CPU times: user 543 µs, sys: 762 µs, total: 1.31 ms
    Wall time: 1.5 s


Using cProfile
~~~~~~~~~~~~~~

This can be done in a notebook with %prun, with the following readouts
as column headers:

-  ncalls

   -  for the number of calls,

-  tottime

   -  for the total time spent in the given function (and excluding time
      made in calls to sub-functions),

-  percall

   -  is the quotient of tottime divided by ncalls

-  cumtime

   -  is the total time spent in this and all subfunctions (from
      invocation till exit). This figure is accurate even for recursive
      functions.

-  percall

   -  is the quotient of cumtime divided by primitive calls

-  filename:lineno(function)

   -  provides the respective data of each function

Profiling Newton iterations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    def newton(z, f, fprime, max_iter=100, tol=1e-6):
        """The Newton-Raphson method."""
        for i in range(max_iter):
            step = f(z)/fprime(z)
            if abs(step) < tol:
                return i, z
            z -= step
        return i, z

.. code:: python

    def plot_newton_iters(p, pprime, n=200, extent=[-1,1,-1,1], cmap='hsv'):
        """Shows how long it takes to converge to a root using the Newton-Raphson method."""
        m = np.zeros((n,n))
        xmin, xmax, ymin, ymax = extent
        for r, x in enumerate(np.linspace(xmin, xmax, n)):
            for s, y in enumerate(np.linspace(ymin, ymax, n)):
                z = x + y*1j
                m[s, r] = newton(z, p, pprime)[0]
        plt.imshow(m, cmap=cmap, extent=extent)

.. code:: python

    def f(x):
        return x**3 - 1
    
    def fprime(x):
        return 3*x**2

.. code:: python

    stats = %prun -r -q plot_newton_iters(f, fprime)


.. parsed-literal::

     


.. image:: MakingCodeFast_files/MakingCodeFast_20_1.png


.. code:: python

    # Restrict to 10 lines
    stats.sort_stats('time').print_stats(10);


.. parsed-literal::

             1088832 function calls (1088459 primitive calls) in 1.938 seconds
    
       Ordered by: internal time
       List reduced from 445 to 10 due to restriction <10>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        40000    0.623    0.000    1.343    0.000 <ipython-input-8-3671b81b1850>:1(newton)
            1    0.519    0.519    1.938    1.938 <ipython-input-9-0773c96453fa>:1(plot_newton_iters)
       324388    0.312    0.000    0.312    0.000 <ipython-input-10-dbc2ff3e5adf>:1(f)
       324388    0.290    0.000    0.290    0.000 <ipython-input-10-dbc2ff3e5adf>:4(fprime)
        40004    0.072    0.000    0.072    0.000 {range}
       324392    0.045    0.000    0.045    0.000 {abs}
          421    0.003    0.000    0.008    0.000 path.py:199(_update_values)
          201    0.003    0.000    0.007    0.000 function_base.py:9(linspace)
          837    0.003    0.000    0.004    0.000 weakref.py:47(__init__)
         2813    0.003    0.000    0.003    0.000 __init__.py:871(__getitem__)
    
    


.. code:: python

    # Restrict using regular expression match
    stats.sort_stats('time').print_stats(r'ipython');


.. parsed-literal::

             1088832 function calls (1088459 primitive calls) in 1.938 seconds
    
       Ordered by: internal time
       List reduced from 445 to 4 due to restriction <'ipython'>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        40000    0.623    0.000    1.343    0.000 <ipython-input-8-3671b81b1850>:1(newton)
            1    0.519    0.519    1.938    1.938 <ipython-input-9-0773c96453fa>:1(plot_newton_iters)
       324388    0.312    0.000    0.312    0.000 <ipython-input-10-dbc2ff3e5adf>:1(f)
       324388    0.290    0.000    0.290    0.000 <ipython-input-10-dbc2ff3e5adf>:4(fprime)
    
    


Using the line profiler
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %load_ext line_profiler

.. code:: python

    lstats = %lprun -r -f plot_newton_iters plot_newton_iters(f, fprime)



.. image:: MakingCodeFast_files/MakingCodeFast_25_0.png


.. code:: python

    lstats.print_stats()


.. parsed-literal::

    Timer unit: 1e-06 s
    
    Total time: 2.40384 s
    File: <ipython-input-9-0773c96453fa>
    Function: plot_newton_iters at line 1
    
    Line #      Hits         Time  Per Hit   % Time  Line Contents
    ==============================================================
         1                                           def plot_newton_iters(p, pprime, n=200, extent=[-1,1,-1,1], cmap='hsv'):
         2                                               """Shows how long it takes to converge to a root using the Newton-Raphson method."""
         3         1           81     81.0      0.0      m = np.zeros((n,n))
         4         1            1      1.0      0.0      xmin, xmax, ymin, ymax = extent
         5       201          396      2.0      0.0      for r, x in enumerate(np.linspace(xmin, xmax, n)):
         6     40200        74400      1.9      3.1          for s, y in enumerate(np.linspace(ymin, ymax, n)):
         7     40000       466076     11.7     19.4              z = x + y*1j
         8     40000      1708697     42.7     71.1              m[s, r] = newton(z, p, pprime)[0]
         9         1       154191 154191.0      6.4      plt.imshow(m, cmap=cmap, extent=extent)
    


Using the memory profiler
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes the problem is that too much memory is being used, and we need
to reduce it so that we can avoid disk churning (swapping of memory from
RAM to hard disk). Two useful magic functions are %memit which works
like %timeit but shows space rahter than time consumption, and %mprun
which is like %lprun for memory usage.

Note that %mprun requires that the funciton to be evaluated is in a
file.

.. code:: python

    %load_ext memory_profiler

.. code:: python

    %memit np.random.random((1000, 1000))


.. parsed-literal::

    peak memory: 90.66 MiB, increment: 7.66 MiB


.. code:: python

    %%file foo.py
    
    def foo(n):
        phrase = 'repeat me'
        pmul = phrase * n
        pjoi = ''.join([phrase for x in xrange(n)])
        pinc = ''
        for x in xrange(n):
            pinc += phrase
        del pmul, pjoi, pinc


.. parsed-literal::

    Overwriting foo.py


.. code:: python

    import foo
    
    %mprun -f foo.foo foo.foo(10000);


.. parsed-literal::

    ('',)


Using better algorihtms and data structures
-------------------------------------------

The first major optimization is to see if we can reduce the algorithmic
complexity of our solution, say from :math:`\mathcal{O}(n^2)` to
:math:`\mathcal{O}(n \log(n))`. Unless you are going to invent an
entirely new algorithm (possible but uncommon), this involves research
into whether the data structures used are optimal, or whetther there is
a way to reformulate the problem to take advantage of better algorithms.
If your inital solution is by "brute force", there is sometimes room for
huge performacne gains here.

Taking a course in data structures and algorithms is a very worthwhile
investment of your time if you are developing novel statsitical
algorithms - perhaps Bloom filters, locality sensitive hashing, priority
queues, Barnes-Hut partitionaing, dynamic programming or minimal
spanning trees can be used to solve your problem - in which case you can
expect to see dramatic improvements over naive brute-force
implementations.

Data structures example
^^^^^^^^^^^^^^^^^^^^^^^

Suppose you were given two lists ``xs`` and ``ys`` and asked to find the
unique elements in common between them.

.. code:: python

    xs = np.random.randint(0, 1000, 10000)
    ys = np.random.randint(0, 1000, 10000)

.. code:: python

    # This is easy to solve using a nested loop
    
    def common1(xs, ys):
        """Using lists."""
        zs = []
        for x in xs:
            for y in ys:
                if x==y and x not in zs:
                    zs.append(x)
        return zs
    
    %timeit -n1 -r1 common1(xs, ys)


.. parsed-literal::

    1 loops, best of 1: 14.7 s per loop


.. code:: python

    # However, it is much more efficient to use the set data structure
    
    def common2(xs, ys):
        return list(set(xs) & set(ys))
    
    %timeit -n1 -r1 common2(xs, ys)


.. parsed-literal::

    1 loops, best of 1: 2.82 ms per loop


.. code:: python

    assert(sorted(common1(xs, ys)) == sorted(common2(xs, ys)))

Algorithms example
^^^^^^^^^^^^^^^^^^

We have seen many such examples in the course - for example, numerical
quadrature versus Monte Carlo integration, gradient desceent versus
conjugate gradient descent, random walk Metropolis versus Hamiltonian
Monte Carlo.

I/O Bound problems
------------------

Sometimes the issue is that you need to load or save massive amounts of
data, and the transfer to and from the hard disk is the bootleneck.
Possible solutions include 1) use of binary rather than text data, 2)
use of data compression, 3) use of specialized data structures such as
HDF5.

If you are working wiht huge amounts of data, conisder the use of 1)
relational databases if there are many rleations to manage, 2) HDF5 if a
hiearchical structure is natural, and 3) NoSQL databases such as Redis
if the data relatons are simple and you need to transfer over the
network.

-  `h5py for HDF5 <http://docs.h5py.org/en/latest/index.html>`__
-  `PyTables for HDF5 <https://pytables.github.io/>`__
-  `AQLite3 Relational
   Database <https://docs.python.org/2/library/sqlite3.html>`__
-  `Python for Redis <https://github.com/andymccurdy/redis-py>`__

Pandas also offers convenient access to multiple storage and retrieval
options via its DataFramee object.

Output
^^^^^^

.. code:: python

    def io1(xs):
        """Using loops to write."""
        with open('foo1.txt', 'w') as f:
            for x in xs:
                f.write('%d\t' % x)
        
    def io2(xs):
        """Join before writing."""
        with open('foo2.txt', 'w') as f:
            f.write('\t'.join(map(str, xs)))
            
    def io3(xs):
        """Numpy savetxt is surprisingly slow."""
        np.savetxt('foo3.txt', xs, delimiter='\t')
            
    def io4(xs):
        """NUmpy save is better if binary format is OK."""
        np.save('foo4.npy', xs)
       
    def io5(xs):
        """Using HDF5."""
        import h5py
        with h5py.File("mytestfile1.h5", "w") as f:
            ds = f.create_dataset("xs", (len(xs),), dtype='i')
            ds[:] = xs
    
    def io6(xs):
        """Using HDF5 with compression."""
        import h5py
        with h5py.File("mytestfile2.h5", "w") as f:
            ds = f.create_dataset("xs", (len(xs),), dtype='i', compression="lzf")
            ds[:] = xs
        
    n = 1000*1000
    xs = range(n)
    %timeit -r1 -n1 io1(xs)
    %timeit -r1 -n1 io2(xs)
    %timeit -r1 -n1 io3(xs)
    %timeit -r1 -n1 io4(xs)
    %timeit -r1 -n1 io5(xs)
    %timeit -r1 -n1 io6(xs)


.. parsed-literal::

    1 loops, best of 1: 1.64 s per loop
    1 loops, best of 1: 320 ms per loop
    1 loops, best of 1: 6.7 s per loop
    1 loops, best of 1: 108 ms per loop
    1 loops, best of 1: 154 ms per loop
    1 loops, best of 1: 122 ms per loop


Input
^^^^^

.. code:: python

    def io11(xs):
        """Using basic python."""
        with open('foo1.txt', 'r') as f:
            xs = map(int, f.read().strip().split('\t'))
        return xs
        
    def io12(xs):
        """Using pandsa."""
        xs = pd.read_table('foo2.txt').values.tolist()
        return xs
    
    def io13(xs):
        """Numpy loadtxt."""
        xs = np.loadtxt('foo3.txt',delimiter='\t')
        return xs
    
    def io14(xs):
        """Numpy load."""
        xs = np.load('foo4.npy')
        return xs
    
    def io15(xs):
        """Using HDF5."""
        import h5py
        with h5py.File("mytestfile1.h5", 'r') as f:
            xs = f['xs'][:]
        return xs
    
    def io16(xs):
        """Using HDF5 with compression."""
        import h5py
        with h5py.File("mytestfile2.h5", 'r') as f:
            xs = f['xs'][:]
        return xs
        
    n = 1000*1000
    xs = range(n)
    %timeit -r1 -n1 io11(xs)
    %timeit -r1 -n1 io12(xs)
    %timeit -r1 -n1 io13(xs)
    %timeit -r1 -n1 io14(xs)
    %timeit -r1 -n1 io15(xs)
    %timeit -r1 -n1 io16(xs)


.. parsed-literal::

    1 loops, best of 1: 805 ms per loop
    1 loops, best of 1: 51.3 s per loop
    1 loops, best of 1: 5.56 s per loop
    1 loops, best of 1: 15.2 ms per loop
    1 loops, best of 1: 9.69 ms per loop
    1 loops, best of 1: 16 ms per loop




Problem set for optimization
----------------------------

We will use a few standard examples throughout to illustrate differnt
optimization techniques.

Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def mult(u, v):
        m, n = u.shape
        n, p = v.shape
        w = np.zeros((m, p))
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    w[i, j] += u[i, k] * v[k, j]
        return w

.. code:: python

    u = np.reshape(np.arange(6), (2,3))
    v = np.reshape(np.arange(9), (3,3))
    
    np.testing.assert_array_almost_equal(mult(u, v), u.dot(v))

Pairwise distance matrix
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def dist(u, v):
        n = len(u)
        s = 0
        for i in range(n):
            s += (u[i] - v[i])**2
        return np.sqrt(s)

.. code:: python

    u = np.array([4,5])
    v = np.array([1,1])
    
    np.testing.assert_almost_equal(dist(u, v), np.linalg.norm(u-v))

.. code:: python

    def pdist(vs, dist=dist):
        n = len(vs)
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                m[i, j] = dist(vs[i], vs[j])
        return m

.. code:: python

    from scipy.spatial.distance import squareform, pdist as sp_pdist
    
    vs = np.array([[0,0], [1,2], [2,3], [3,4]])
    
    np.testing.assert_array_almost_equal(pdist(vs), squareform(sp_pdist(vs)))

Word count
~~~~~~~~~~

.. code:: python

    import string
    
    def word_count(docs):
        wc = {}
        for doc in docs:
            words = doc.translate(None, string.punctuation).split()
            for word in words:
                wc[word] = wc.get(word, 0) + 1
        return wc

.. code:: python

    docs = ['hello, there handsome!', 'hi, there, beautiful']
    
    word_count(docs)

