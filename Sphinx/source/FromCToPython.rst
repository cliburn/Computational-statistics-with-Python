
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

    import bitey

.. code:: python

    %load_ext cythonmagic

Using C code in Python
======================

Example: The Fibonacci Sequence
-------------------------------

.. code:: python

    %%file c_fib.h
    double c_fib(int n);


.. parsed-literal::

    Overwriting c_fib.h


.. code:: python

    %%file c_fib.c
    double c_fib(int n) {
        double tmp, a = 0, b = 1;
        for (int i=0; i<n; i++) {
            tmp = a;
            a = a + b;
            b = tmp;
        }
        return a;
    }


.. parsed-literal::

    Overwriting c_fib.c


Using clang and bitey
---------------------

.. code:: python

    !clang -O3 -emit-llvm -c c_fib.c -o bitey_fib.o

.. code:: python

    import bitey_fib

.. code:: python

    bitey_fib.c_fib(10)




.. parsed-literal::

    55.0000



Using gcc and ctypes
--------------------

.. code:: python

    ! gcc -O3 -bundle -undefined dynamic_lookup c_fib.c -o ctypes_fib.so

.. code:: python

    # For Unix systmes
    # ! gcc -O3 -fPIC -shared -std=c99  c_fib.c -o ctypes_fib.so

.. code:: python

    from ctypes import CDLL, c_int, c_double
    
    def ctypes_fib(n):
        
        # Use ctypes to load the library
        lib = CDLL('./ctypes_fib.so')
    
        # We need to give the argument and return types explicitly
        lib.c_fib.argtypes = [c_int]
        lib.c_fib.restype  = c_double
        
        return lib.c_fib(n)

.. code:: python

    ctypes_fib(10)




.. parsed-literal::

    55.0000



Using Cython
------------

.. code:: python

    %load_ext cythonmagic


.. parsed-literal::

    The cythonmagic extension is already loaded. To reload it, use:
      %reload_ext cythonmagic


.. code:: python

    %%file cy_fib.pxd
    cdef extern from "c_fib.h":
        double c_fib(int n)


.. parsed-literal::

    Overwriting cy_fib.pxd


.. code:: python

    %%file cy_fib.pyx
    cimport cy_fib
    
    cpdef cython_fib(n):
        return cy_fib.c_fib(n)


.. parsed-literal::

    Overwriting cy_fib.pyx


.. code:: python

    %%file setup.py
    from distutils.core import setup, Extension
    from Cython.Build import cythonize
    
    ext = Extension("cy_fib",
                  sources=["cy_fib.pyx", "c_fib.c"])
    
    setup(name = "cython_fib",
          ext_modules = cythonize(ext))


.. parsed-literal::

    Overwriting setup.py


.. code:: python

    ! python setup.py build_ext -i &> /dev/null

Benchmark
---------

.. code:: python

    import cy_fib
    import bitey_fib

.. code:: python

    print ctypes_fib(100)
    print bitey_fib.c_fib(100)
    print cy_fib.cython_fib(100)


.. parsed-literal::

    3.54224848179e+20
    3.54224848179e+20
    3.54224848179e+20


.. code:: python

    %timeit -n 1000 ctypes_fib(100)
    %timeit -n 1000 bitey_fib.c_fib(100)
    %timeit -n 1000 cy_fib.cython_fib(100)


.. parsed-literal::

    1000 loops, best of 3: 92.3 µs per loop
    1000 loops, best of 3: 905 ns per loop
    1000 loops, best of 3: 264 ns per loop



