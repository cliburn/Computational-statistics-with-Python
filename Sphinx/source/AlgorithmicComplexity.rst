
.. code:: python

    import os
    import sys
    import glob
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    %matplotlib inline
    %precision 4
    plt.style.use('ggplot')


Algorithmic complexity
======================

Data structures, as the name implies, are abstract structures for
storing data. You are already familiar wiht several - e..g. list and
dict. Algorithms are esssntially *recipes* for manipulating data
structures. As we get into more computationally intensive calculations,
we need to better understand how the performacne of data structures and
algorithms is measured so that we can choose the appropriate strategy.

Profling and benchmarking
-------------------------

We can measure performance in terms of the time taken as well as in
terms of the number of basic operations performed. Measuring the time
taken is known as profiling or benchmarking, and in the IPython notebook
can be done conveniently with matic functions.

.. code:: python

    # Use %timeit to measure function calls
    def f():
        import time
        time.sleep(2)
    
    %timeit -n1 f()


.. parsed-literal::

    1 loops, best of 3: 2 s per loop


.. code:: python

    # Use %%time to measure evaluation cell the Unix way
    %%time
    
    f()


.. parsed-literal::

    CPU times: user 522 µs, sys: 940 µs, total: 1.46 ms
    Wall time: 2 s


Measuring algorithmic complexity
--------------------------------

However, profiling doesn't tell us much about how the algorithm will
perform on a different computer since it is partly determined by the
hardware available. To compare performance in a device-indpendent
fashion, we use what is known as Big O notation (you may or may not have
encountered this before in your Calculus courses). The Big O formalism
characterizes functions in terms of their rates of growth.

A little more formally, we have a comparison function :math:`g(n)` and
another function :math:`f(n)` that returns the number of "elementary
operations" we need to perform in our algorithm given an input of size
:math:`n`. In the example, the elementary oepration is comparison of two
items. In statisitcal algorithms, this is most commonly a floating point
operation (FLOP), such as addition or multiplicaiton of two floats. Now
if the ratio :math:`|f(n)/g(n)|` can be bounded by a finite number
:math:`M` as :math:`n` grows to infinity, we say that :math:`f(n)` has
complexity of order :math:`g(n)`. For example, if :math:`f(n) = 10n^2`
and :math:`g(n) = n`, then there is no such number :math:`M` and
:math:`f(n)` is **not** :math:`\mathcal{O}(n)`, but if
:math:`g(n) = n^2`, then :math:`M = 10` wil do and we say that
:math:`f(n)` is :math:`\mathcal{O}(n^2)`. So our search function is
:math:`\mathcal{O}(n)`. Formally, it is also :math:`\mathcal{O} (n^2)`
and so on, but we always choose the "smallest" function :math:`g`. We
also drop all terms ohter than the larget - so we don't say
:math:`\mathcal{O}(n^3 + n^2 + n)` but simply :math:`\mathcal{O}(n^3)`.

Note that since the constant is not used in big O notation, two
algorithms can have the same big O complexity and have very different
performance! However, the O notation is very helpful for understanding
the *scalability* of our algorithm. Below we show a comparison of an
:math:`\mathcal{O}(n^2)` algorithm (e.g. bubble sort) with an
:math:`\mathcal{O}(n \log{n})` algorithm (e.g. merge sort). Regardless
of the difference in constant factor, there is no competition as
:math:`n` gets large.

Suppsoe you wanted to search for an item in an unsorted list of length
:math:`n`. One way to do this would be to scan from the first position
sequentially until you find it (or not). If the item is in the list, you
will need to scan (:math:`n/2`) items on average to find it. If it is
not in the list, you will need to scan all :math:`n` items. In any case,
the complexity of the search grows linearly with the lenght of the list
:math:`n`. We say that the algorithmic complexity of the search using a
linear scan is :math:`\mathcal{O}(n)`.

Strictly, we should say the *average* complexity is
:math:`\mathcal{O}(n)`. We can also calculate worst case performance
(when the item is not in the list), which is the same class
:math:`\mathcal{O}(n)` as average complexity for this searching example.
Since worst case performance may require a perverse organizaiotn of the
input (e.g. asking a sort function to sort an already sorted list),
randomizaiton of inputs will sometimes suffice to convert it to the
average case.

Question: What is the algorithmic complexity of textbook matrix
multiplication? Why?

Comparing complexity of :math:`\mathcal{O}(n^2)` (e.g. bubble sort) and :math:`\mathcal{O} (n \log n)` (e.g. merge sort).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def f1(n, k):
        return k*n*n
    
    def f2(n, k):
        return k*n*np.log(n)
    
    n = np.arange(0, 20001)
    
    plt.plot(n, f1(n, 1), c='blue')
    plt.plot(n, f2(n, 1000), c='red')
    plt.xlabel('Size of input (n)', fontsize=16)
    plt.ylabel('Number of operations', fontsize=16)
    plt.legend(['$\mathcal{O}(n^2)$', '$\mathcal{O}(n \log n)$'], loc='best', fontsize=20);



.. image:: AlgorithmicComplexity_files/AlgorithmicComplexity_9_0.png


Ranking of common Big O complexity classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  consstant = :math:`\mathcal{O}(1)`
-  logarithmic = :math:`\mathcal{O}(\log n)`
-  linear = :math:`\mathcal{O}(n)`
-  n log n = :math:`\mathcal{O}(n \log n)`
-  quadratic = :math:`\mathcal{O}(n^2)`
-  cubic = :math:`\mathcal{O}(n^3)`
-  polynomial = :math:`\mathcal{O}(n^k)`
-  exponential = :math:`\mathcal{O}(k^n)`
-  factorial =\ :math:`\mathcal{O}(n!)`

.. code:: python

    from IPython.display import Image

.. code:: python

    Image(url='http://bigocheatsheet.com/img/big-o-complexity.png')




.. raw:: html

    <img src="http://bigocheatsheet.com/img/big-o-complexity.png"/>



Complexity of common operations on Python data structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `here <https://wiki.python.org/moin/TimeComplexity>`__ for the
complexity of operations on standard Python data structures. Note for
instance that searching a list is much more expensive than searching a
dicitonary.

.. code:: python

    # Searching a list is O(n)
    
    alist = range(1000000)
    r = np.random.randint(100000)
    %timeit -n3 r in alist


.. parsed-literal::

    3 loops, best of 3: 1.28 ms per loop


.. code:: python

    # Searching a dictionary is O(1)
    
    adict = dict.fromkeys(alist)
    %timeit -n3 r in adict


.. parsed-literal::

    3 loops, best of 3: 318 ns per loop


Space complexity
----------------

We can aslo use big O notnation in the same way to measure the space
complexity of an algorithm. The basic idea is identical. The notion of
space complexity becomes important when you data volume is of the same
magntude orlarger than the memory you have available. In that case, an
algorihtm with high space complexity may end up having to swap memory
constantly, and will perform far worse than its Big O for time
complexity would suggest.

Sometimes, you can trade space complexity for time complexity - caching
and dynamic programming are obvious examples.

How much space do I need?
~~~~~~~~~~~~~~~~~~~~~~~~~

Just as you should have a good idea of how your algorithm will scale
with increasing :math:`n`, you should also be able to know how much
memroy your data structures will require. For example, if you had an
:math:`n \times p` matrix of integers, an :math:`n \times p` matrix of
flaots, and an :math:`n \times p` matrix of complex floats, how large
can :math:`n` and :math:`p` be before you run out of RAM to store them?

.. code:: python

    # Notice how much overhead Python objects have
    # A raw integer should be 64 bits or 8 bytes only
    
    print sys.getsizeof(1)
    print sys.getsizeof(1234567890123456789012345678901234567890)
    print sys.getsizeof(3.14)
    print sys.getsizeof(3j)
    print sys.getsizeof('a')
    print sys.getsizeof('hello world')


.. parsed-literal::

    24
    44
    24
    32
    38
    48


.. code:: python

    print np.ones((100,100), dtype='byte').nbytes
    print np.ones((100,100), dtype='i2').nbytes
    print np.ones((100,100), dtype='int').nbytes # default is 64 bits or 8 bytes
    print np.ones((100,100), dtype='f4').nbytes
    print np.ones((100,100), dtype='float').nbytes # default is 64 bits or 8 bytes
    print np.ones((100,100), dtype='complex').nbytes


.. parsed-literal::

    10000
    20000
    80000
    40000
    80000
    160000

