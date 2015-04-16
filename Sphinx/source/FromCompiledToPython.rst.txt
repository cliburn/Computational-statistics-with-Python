
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

    ! pip install git+https://github.com/dabeaz/bitey.git
    %install_ext https://raw.github.com/mgaitan/fortran_magic/master/fortranmagic.py
    %install_ext https://gist.githubusercontent.com/bfroehle/3458310/raw/biteymagic.py


.. parsed-literal::

    Downloading/unpacking git+https://github.com/dabeaz/bitey.git
      Cloning https://github.com/dabeaz/bitey.git to /var/folders/bh/x038t1s943qftp7jzrnkg1vm0000gn/T/pip-_UHN_B-build
      Running setup.py (path:/var/folders/bh/x038t1s943qftp7jzrnkg1vm0000gn/T/pip-_UHN_B-build/setup.py) egg_info for package from git+https://github.com/dabeaz/bitey.git
        
        warning: no files found matching '*' under directory 'doc'
      Requirement already satisfied (use --upgrade to upgrade): bitey==0.0 from git+https://github.com/dabeaz/bitey.git in /Users/cliburn/anaconda/lib/python2.7/site-packages
    Cleaning up...
    Installed fortranmagic.py. To use it, type:
      %load_ext fortranmagic
    Installed biteymagic.py. To use it, type:
      %load_ext biteymagic


Using functions from various compiled languages in Python
=========================================================

There are 2 main reasons why interpreted Python code is slower than code
in a compiled lanauge such as C (or other compiled langauge):

-  Python executes byte code in a virtual machine (minor effect) while C
   compiles down to machine instructions specific for the processor
-  Python has dynamic typing (major effect) while C is statically typed.
   In a dynamically typed language, the simple expression ``a + b`` can
   mean many, many different things, and the interrpeter has to figure
   out which interpretation is intended. In contrast, ``a`` and ``b``
   must have a type in C such as ``double`` and there is no ambiguity
   about what ``+`` means to resolve.

If speed is critical, it is often necessary to exploit the efficiency of
compiled languges - this can be done while retaining the nice features
of Python in 2 directions

-  From C to Python
-  From Python to C

Here we will look at how to go from C (C++, Fortran, Julia) to Python,

.. code:: python

    def python_fib(n):
        a, b = 0,  1
        for i in range(n):
            a, b = a+b, a
        return a

.. code:: python

    %timeit python_fib(100)


.. parsed-literal::

    100000 loops, best of 3: 8.47 µs per loop


C
-

.. code:: python

    %%file fib.h
    
    double fib(int n);


.. parsed-literal::

    Writing fib.h


.. code:: python

    %%file fib.c
    
    double fib(int n) {
        double a = 0, b = 1;
        for (int i=0; i<n; i++) {
            double tmp = b;
            b = a;
            a += tmp;
         }
        return a;
    }


.. parsed-literal::

    Writing fib.c


Using bitey and clang
~~~~~~~~~~~~~~~~~~~~~

This is perhaps the simplest method, but it only works with the
``clang`` compiler and does not geenrate highly optimized code.

.. code:: python

    import bitey

.. code:: python

    !clang -O3 -emit-llvm -c fib.c -o fib1.o

.. code:: python

    import fib1
    
    fib1.fib(100)




.. parsed-literal::

    354224848179261997056.0000



.. code:: python

    %timeit fib1.fib(100)


.. parsed-literal::

    1000000 loops, best of 3: 941 ns per loop


Using Cython
~~~~~~~~~~~~

I recomment using Cython for all your C/C++ interface needs as it is
highly optimized and can do boht C :math:`\rightarrow` Python and Python
:math:`\rightarrow` C. It is a littel more involved, but the steps
always follow the same template.

Define functions to be imported from C
''''''''''''''''''''''''''''''''''''''

.. code:: python

    %%file fib.pxd
    
    cdef extern from "fib.h":
        double fib(int n)


.. parsed-literal::

    Writing fib.pxd


Define wrapper for calling function from Python
'''''''''''''''''''''''''''''''''''''''''''''''

.. code:: python

    %%file fib2.pyx
    
    cimport fib
    
    def fib(n):
        return fib.fib(n)


.. parsed-literal::

    Writing fib2.pyx


Use distutils to compile shared library for Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the standard way *all* Python modules are compiled for
distribution, and results in a build that is portable over different
platforms.

.. code:: python

    %%file setup.py
    from distutils.core import setup, Extension
    from Cython.Build import cythonize
    
    ext = Extension("fib2",
                  sources=["fib2.pyx", "fib.c"])
    
    setup(name = "cython_fib",
          ext_modules = cythonize(ext))


.. parsed-literal::

    Overwriting setup.py


.. code:: python

    ! python setup.py build_ext -i &> /dev/null

.. code:: python

    import fib2
    
    fib2.fib(100)




.. parsed-literal::

    354224848179261997056.0000



.. code:: python

    %timeit fib2.fib(100)


.. parsed-literal::

    1000000 loops, best of 3: 224 ns per loop


C++
---

C++ is a superset of C - the syntax for the fib program is exactly the
same except for change in the filname extensions.

.. code:: python

    %%file fib.hpp
    
    double fib(int n);


.. parsed-literal::

    Writing fib.hpp


.. code:: python

    %%file fib.cpp
    
    double fib(int n) {
        double a = 0, b = 1;
        for (int i=0; i<n; i++) {
            double tmp = b;
            b = a;
            a += tmp;
         }
        return a;
    }


.. parsed-literal::

    Writing fib.cpp


.. code:: python

    %%file setup.py
    from distutils.core import setup, Extension
    from Cython.Build import cythonize
    
    ext = Extension("fib2cpp",
                  sources=["fib2cpp.pyx", "fib.cpp"],
                  language="c++",)
    
    setup(name = "cython_fibcpp",
          ext_modules = cythonize(ext))


.. parsed-literal::

    Overwriting setup.py


.. code:: python

    %%file fib2cpp.pyx
    
    cimport fib
    
    def fib(n):
        return fib.fib(n)


.. parsed-literal::

    Writing fib2cpp.pyx


.. code:: python

    ! python setup.py build_ext -i &> /dev/null

.. code:: python

    import fib2cpp

.. code:: python

    fib2cpp.fib(100)




.. parsed-literal::

    354224848179261997056.0000



Fortran
-------

This is almost trivial with the Fortran Magic extnesion.

.. code:: python

    ! pip install fortran-magic &> /dev/null

.. code:: python

    %load_ext fortranmagic




.. code:: python

    %%fortran
    
    subroutine fib3(n, a)
        integer, intent(in) :: n
        real, intent(out) :: a
    
        integer :: i
        real :: b, tmp
    
        a = 0
        b = 1
        do i = 1, n
            tmp = b
            b = a
            a = a + tmp
        end do
    end subroutine

.. code:: python

    fib3(100)




.. parsed-literal::

    354224717716315439104.0000



Antoher example from the
`documentation <http://nbviewer.ipython.org/github/mgaitan/fortran_magic/blob/master/documentation.ipynb>`__

.. code:: python

    %%fortran --link lapack
    
    subroutine solve(A, b, x, n)
        ! solve the matrix equation A*x=b using LAPACK
        implicit none
    
        real*8, dimension(n,n), intent(in) :: A
        real*8, dimension(n), intent(in) :: b
        real*8, dimension(n), intent(out) :: x
    
        integer :: pivot(n), ok
    
        integer, intent(in) :: n
        x = b
    
        ! find the solution using the LAPACK routine SGESV
        call DGESV(n, 1, A, n, pivot, x, n, ok)
        
    end subroutine

.. code:: python

    A = np.array([[1, 2.5], [-3, 4]])
    b = np.array([1, 2.5])
    
    solve(A, b)




.. parsed-literal::

    array([-0.1957,  0.4783])



Benchmarking
------------

.. code:: python

    %timeit python_fib(100) # Python
    %timeit fib1.fib(100)   # bitey
    %timeit fib2.fib(100)   # Cython
    %timeit fib3(100)       # Fortran


.. parsed-literal::

    100000 loops, best of 3: 11 µs per loop
    1000000 loops, best of 3: 957 ns per loop
    1000000 loops, best of 3: 253 ns per loop
    1000000 loops, best of 3: 255 ns per loop


Wrapping a function from a C library for use in Python
------------------------------------------------------

Cython ships with a set of standard .pxd files that provide these
declarations in a readily usable way that is adapted to their use in
Cython. The main packages are ``cpython``, ``libc`` and ``libcpp``. The
NumPy library also has a standard .pxd file ``numpy``, as it is often
used in Cython code. See Cython’s Cython/Includes/ source package for a
complete list of provided .pxd files. (From
http://docs.cython.org/src/tutorial/clibraries.html).

Additional .pxd are also avaialbel for:

-  `The Rmath library <https://github.com/nfoti/cythonRMath>`__
-  `The GNU scientific library <https://github.com/twiecki/CythonGSL>`__

However, here is an example of how to write functions from an external C
library if you have to do it yourself. The example is taken from
https://github.com/cythonbook/examples and wraps the Mersenne Twister
from http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html for use in
Python.

.. code:: python

    if not os.path.exists('mt19937ar.h'):
        ! wget http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.sep.tgz
        ! tar -xzvf mt19937ar.sep.tgz


.. parsed-literal::

    --2015-03-26 16:02:41--  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.sep.tgz
    Resolving www.math.sci.hiroshima-u.ac.jp... 133.41.16.48
    Connecting to www.math.sci.hiroshima-u.ac.jp|133.41.16.48|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 15433 (15K) [application/x-gzip]
    Saving to: ‘mt19937ar.sep.tgz’
    
    100%[======================================>] 15,433      37.3KB/s   in 0.4s   
    
    2015-03-26 16:02:42 (37.3 KB/s) - ‘mt19937ar.sep.tgz’ saved [15433/15433]
    
    x mt19937ar.c
    x mt19937ar.h
    x mt19937ar.out
    x mtTest.c
    x readme-mt.txt


.. code:: python

    %%file mt.pxd
    
    cdef extern from "mt19937ar.h":
        void init_genrand(unsigned long s)
        double genrand_real1()


.. parsed-literal::

    Writing mt.pxd


.. code:: python

    %%file mt_random.pyx
    
    cimport mt
    
    def init_state(unsigned long s):
        mt.init_genrand(s)
    
    def rand():
        return mt.genrand_real1()


.. parsed-literal::

    Writing mt_random.pyx


.. code:: python

    %%file setup.py
    
    from distutils.core import setup, Extension
    from Cython.Build import cythonize
    
    ext = Extension("mt_random",
                    sources=["mt_random.pyx", "mt19937ar.c"])
    
    setup(name="mersenne_random",
          ext_modules = cythonize([ext]))


.. parsed-literal::

    Overwriting setup.py


.. code:: python

    ! python setup.py build_ext -i &> /dev/null

.. code:: python

    import mt_random
    
    mt_random.init_state(123)
    for i in range(10):
        print mt_random.rand(),
    print


.. parsed-literal::

    0.696469187433 0.712955321584 0.28613933881 0.428470925062 0.226851454989 0.690884851546 0.55131476525 0.71915030892 0.719468970718 0.491118932723


Wrapping functions from C++ library for use in Pyton
----------------------------------------------------

Example - Andrew Cron (DSS PhD graduate) has a GitHub repository
wrapping the C++ Armadillo linear algebra package with Cython at
https://github.com/andrewcron/cy\_armadillo

