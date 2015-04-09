
.. code:: python

    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    from numba import jit
    import numexpr as ne
    import parakeet
    %precision 2




.. parsed-literal::

    u'%.2f'



Optimization bake-off
=====================

Python is a high-level interpreted language, which greatly reduces the
time taken to prototyte and develop useful statistical programs.
However, the trade-off is that *pure* Python programs can be orders of
magnitude slower than programs in compiled languages such as C/C++ or
Forran. Hence most numerical and statistical programs often include
interfaces to compiled code (e.g. numpy which is written in C) or more
recently, are just-in-time compiled to native machine code (e.g. numba,
pymc3). Fortunately, it is relatively easy to write custom modules that
comple to native machine code and call them from Pytthon, an important
factor in the popularity of Python as a langugae for scientific and
statistical computing.

We will use the example of calculating the pairwsise Euclidean distance
between all points to illustrate the various methods of interfacing with
native code.

Adapted and extended from
http://nbviewer.ipython.org/url/jakevdp.github.io/downloads/notebooks/NumbaCython.ipynb

.. code:: python

    A = np.array([[0.0,0.0],[3.0,4.0]])
    n = 1000
    p = 3
    xs = np.random.random((n, p))

Python version
--------------

.. code:: python

    def pdist_python(xs):
        n, p = xs.shape
        D = np.empty((n, n), np.float)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(p):
                    tmp = xs[i,k] - xs[j,k]
                    s += tmp * tmp
                D[i, j] = s**0.5
        return D

.. code:: python

    print pdist_python(A)
    %timeit -n 1 pdist_python(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    1 loops, best of 3: 3.87 s per loop


Numpy version
-------------

The numpy version makes use of advanced broadcasting. To follow the code
below, we will have to understand numpy broadcasting rules a little
better. Here is the gist from:

From
http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis

When operating on two arrays, NumPy compares their shapes element-wise.
It starts with the trailing dimensions, and works its way forward. Two
dimensions are compatible when

-  they are equal, or
-  one of them is 1

Arrays do not need to have the same number of dimensions. When either of
the dimensions compared is one, the larger of the two is used. In other
words, the smaller of two axes is stretched or “copied” to match the
other.

Distance between scalars
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    x = np.arange(10)
    x




.. parsed-literal::

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



.. code:: python

    # if we insert an extra dimension into x with np.newaxis
    # we get a (10, 1) matrix
    x[:, np.newaxis].shape




.. parsed-literal::

    (10, 1)



Comparing shape

::

    x[:, None] = 10 x 1
    x          =     10

When we subtract the two arrays, broadcasting rules first match the the
trailing axis to 10 (so x[:, None] is stretched to be (10,10)), and then
matching the next axis, x is stretechd to also be (10,10).

.. code:: python

    # This is the pairwise distance matrix!
    x[:, None] - x




.. parsed-literal::

    array([[ 0, -1, -2, -3, -4, -5, -6, -7, -8, -9],
           [ 1,  0, -1, -2, -3, -4, -5, -6, -7, -8],
           [ 2,  1,  0, -1, -2, -3, -4, -5, -6, -7],
           [ 3,  2,  1,  0, -1, -2, -3, -4, -5, -6],
           [ 4,  3,  2,  1,  0, -1, -2, -3, -4, -5],
           [ 5,  4,  3,  2,  1,  0, -1, -2, -3, -4],
           [ 6,  5,  4,  3,  2,  1,  0, -1, -2, -3],
           [ 7,  6,  5,  4,  3,  2,  1,  0, -1, -2],
           [ 8,  7,  6,  5,  4,  3,  2,  1,  0, -1],
           [ 9,  8,  7,  6,  5,  4,  3,  2,  1,  0]])



Distance between vectors
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # Suppose we have a collection of vectors of dimeniosn 2
    # In the example below, there are 5 such 2-vectors
    # We want to calculate the Euclidean distance 
    # for all pair-wise comparisons in a 5 x 5 matrix
    
    x = np.arange(10).reshape(5,2)
    print x.shape
    print x


.. parsed-literal::

    (5, 2)
    [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]


.. code:: python

    x[:, None, :].shape




.. parsed-literal::

    (5, 1, 2)



Comparing shape

::

    x[:, None, :] = 5 x 1 x 2
    x          =        5 x 2

From the rules of broadcasting, we expect the result of subtraction to
be a 5 x 5 x 2 array. To calculate Euclidean distance, we need to find
the square root of the sum of squares for the 5 x 5 collection of
2-vectors.

.. code:: python

    delta = x[:, None, :] - x
    pdist = np.sqrt((delta**2).sum(-1))
    pdist




.. parsed-literal::

    array([[  0.  ,   2.83,   5.66,   8.49,  11.31],
           [  2.83,   0.  ,   2.83,   5.66,   8.49],
           [  5.66,   2.83,   0.  ,   2.83,   5.66],
           [  8.49,   5.66,   2.83,   0.  ,   2.83],
           [ 11.31,   8.49,   5.66,   2.83,   0.  ]])



Finally, we come to the anti-climax - a one-liner function!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    def pdist_numpy(xs):
        return np.sqrt(((xs[:,None,:] - xs)**2).sum(-1))

.. code:: python

    print pdist_numpy(A)
    %timeit pdist_numpy(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    10 loops, best of 3: 94.2 ms per loop


Numexpr version
---------------

.. code:: python

    def pdist_numexpr(xs):
        a = xs[:, np.newaxis, :]
        return np.sqrt(ne.evaluate('sum((a-xs)**2, axis=2)'))

.. code:: python

    print pdist_numexpr(A)
    %timeit pdist_numexpr(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    10 loops, best of 3: 30.7 ms per loop


Numba version
-------------

.. code:: python

    pdist_numba = jit(pdist_python)

.. code:: python

    print pdist_numba(A)
    %timeit pdist_numba(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 11.7 ms per loop


NumbaPro version
----------------

.. code:: python

    import numbapro
    pdist_numbapro = numbapro.jit(pdist_python)

.. code:: python

    pdist_numbapro(A)
    %timeit pdist_numbapro(xs)


.. parsed-literal::

    100 loops, best of 3: 11.6 ms per loop


Parakeet version
----------------

.. code:: python

    pdist_parakeet = parakeet.jit(pdist_python)

.. code:: python

    print pdist_parakeet(A)
    %timeit pdist_parakeet(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 18.1 ms per loop


Cython version
--------------

For more control over the translation to C, most Python scientific
developers will use the Cython package. Essentially, this is a language
that resembles Python with type annotations. The Cython code is then
compiled into native code tranaparently. The great advantage of Cythonn
over ther approaches are:

-  A Python program is also valid Cython program, so optimization can
   occur incrementally
-  Fine degree of control over degree of optimization
-  Easy to use - handles details about the C compiler and shared library
   generation
-  Cythonmagic extension comes built into IPyhton notebook
-  Can run parallel code with the nogil decorator
-  Fully optimized code runs at thee same speed as C in most cases

.. code:: python

    %load_ext cythonmagic


.. parsed-literal::

    The Cython magic has been moved to the Cython package, hence 
    `%load_ext cythonmagic` is deprecated; please use `%load_ext Cython` instead.
    
    Though, because I am nice, I'll still try to load it for you this time.


.. code:: python

    %%cython
    
    import numpy as np
    cimport cython
    from libc.math cimport sqrt
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pdist_cython(double[:, ::1] xs):
        cdef int n = xs.shape[0]
        cdef int p = xs.shape[1]
        cdef double tmp, d
        cdef double[:, ::1] D = np.empty((n, n), dtype=np.float)
        for i in range(n):
            for j in range(n):
                d = 0.0
                for k in range(p):
                    tmp = xs[i, k] - xs[j, k]
                    d += tmp * tmp
                D[i, j] = sqrt(d)
        return np.asarray(D)

.. code:: python

    print pdist_cython(A)
    %timeit pdist_cython(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 7.09 ms per loop


C version
---------

There are many ways to wrap C code for Python, such as
`Cython <http://cython.org/>`__, `Swig <http://www.swig.org/>`__ or
`Boost Python with numpy <https://github.com/ndarray/Boost.NumPy>`__.
However, the standard library comes with
`ctypes <https://docs.python.org/2/library/ctypes.html>`__, a foreign
function library that can be used to wrap C functions for use in pure
python. This involves a little more work than the other approaches as
shown below.

.. code:: python

    %%file pdist_c.c
    #include <math.h>
    
    void pdist_c(int n, int p, double xs[n*p], double D[n*n]) {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                double s = 0.0;
                for (int k=0; k<p; k++) {
                    double tmp = xs[i*p+k] - xs[j*p+k];
                    s += tmp*tmp;
                }
                D[i*n+j] = sqrt(s);
            }
        }
    }


.. parsed-literal::

    Writing pdist_c.c


.. code:: python

    # Compile to a shared library
    # Mac
    ! gcc -O3 -bundle -undefined dynamic_lookup pdist_c.c -o pdist_c.so
    # Linux: 
    # ! gcc -O3 -fPIC -shared -std=c99 -lm pdist_c.c -o pdist_c.so

.. code:: python

    from ctypes import CDLL, c_int, c_void_p
    
    def pdist_c(xs):
        
        # Use ctypes to load the library
        lib = CDLL('./pdist_c.so')
    
        # We need to give the argument adn return types explicitly
        lib.pdist_c.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(dtype = np.float), np.ctypeslib.ndpointer(dtype = np.float)]
        lib.pdist_c.restype  = c_void_p
        
        n, p = xs.shape
        D = np.empty((n, n), np.float)
        
        lib.pdist_c(n, p, xs, D)
        return D

.. code:: python

    print pdist_c(A)
    %timeit pdist_c(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 7.5 ms per loop


C++ version
-----------

Using C++ is almost the same as using C. Just add an extern C statement
and use an appropriate C++ compiler.

.. code:: python

    %%file pdist_cpp.cpp
    #include <cmath>
    
    extern "C" 
    
    // Variable length arrays are OK for C99 but not legal in C++
    // void pdist_cpp(int n, int p, double xs[n*p], double D[n*n]) {
    void pdist_cpp(int n, int p, double *xs, double *D) {
        for (int i=0; i<n; i++) {
            for (int j=0; j<n; j++) {
                double s = 0.0;
                for (int k=0; k<p; k++) {
                    double tmp = xs[i*p+k] - xs[j*p+k];
                    s += tmp*tmp;
                }
                D[i*n+j] = sqrt(s);
            }
        }
    }


.. parsed-literal::

    Writing pdist_cpp.cpp


.. code:: python

    # Compile to a shared library
    ! g++ -O3 -bundle -undefined dynamic_lookup pdist_cpp.cpp -o pdist_cpp.so
    # Linux: 
    # ! g++ -O3 -fPIC -shared pdist_cpp.cpp -o pdist_cpp.so

.. code:: python

    from ctypes import CDLL, c_int, c_void_p
    
    def pdist_cpp(xs):
    
        # Use ctypes to load the library
        lib = CDLL('./pdist_cpp.so')
    
        # We need to give the argument adn return types explicitly
        lib.pdist_cpp.argtypes = [c_int, c_int, np.ctypeslib.ndpointer(dtype = np.float), np.ctypeslib.ndpointer(dtype = np.float)]
        lib.pdist_cpp.restype  = c_void_p
    
        n, p = xs.shape
        D = np.empty((n, n), np.float)
        
        lib.pdist_cpp(n, p, xs, D)
        return D

.. code:: python

    print pdist_cpp(A)
    %timeit pdist_cpp(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 7.56 ms per loop


Fortran version
---------------

.. code:: python

    %%file pdist_fortran.f90
    
    subroutine pdist_fortran (n, p, A, D)
    
        integer, intent(in) :: n
        integer, intent(in) :: p
        real(8), intent(in), dimension(n,p) :: A
        real(8), intent(inout), dimension(n,n) :: D
                
        integer :: i, j, k
        real(8) :: s, tmp
        ! note order of indices is different from C
        do j = 1, n
            do i = 1, n
                s = 0.0
                do k = 1, p
                    tmp = A(i, k) - A(j, k)
                    s = s + tmp*tmp
                end do
                D(i, j) = sqrt(s)
            end do
        end do
    end subroutine


.. parsed-literal::

    Writing pdist_fortran.f90


.. code:: python

    ! f2py -c -m flib pdist_fortran.f90 > /dev/null

.. code:: python

    import flib
    print flib.pdist_fortran.__doc__


.. parsed-literal::

    pdist_fortran(a,d,[n,p])
    
    Wrapper for ``pdist_fortran``.
    
    Parameters
    ----------
    a : input rank-2 array('d') with bounds (n,p)
    d : in/output rank-2 array('d') with bounds (n,n)
    
    Other Parameters
    ----------------
    n : input int, optional
        Default: shape(a,0)
    p : input int, optional
        Default: shape(a,1)
    


.. code:: python

    def pdist_fortran(xs):
        import flib
        n, p = xs.shape
        xs = np.array(xs, order='F')
        D = np.empty((n,n), np.float, order='F')
        flib.pdist_fortran(xs, D)
        return D

.. code:: python

    print pdist_fortran(A)
    %timeit pdist_fortran(xs)


.. parsed-literal::

    [[ 0.  5.]
     [ 5.  0.]]
    100 loops, best of 3: 7.23 ms per loop


Bake-off
--------

.. code:: python

    # Final bake-off 
    
    w = 10
    print 'Python'.ljust(w), 
    %timeit pdist_python(xs)
    print 'Numpy'.ljust(w), 
    %timeit pdist_numpy(xs)
    print 'Numexpr'.ljust(w), 
    %timeit pdist_numexpr(xs)
    print 'Numba'.ljust(w), 
    %timeit pdist_numba(xs)
    print 'Parakeet'.ljust(w), 
    %timeit pdist_parakeet(xs)
    print 'Cython'.ljust(w),
    %timeit pdist_cython(xs)
    print 'C'.ljust(w),
    %timeit pdist_c(xs)
    print 'C++'.ljust(w),
    %timeit pdist_cpp(xs)
    print 'Fortran'.ljust(w),
    %timeit pdist_fortran(xs)
    
    from scipy.spatial.distance import pdist as pdist_scipy
    print 'Scipy'.ljust(w),
    %timeit pdist_scipy(xs)


.. parsed-literal::

    Python    1 loops, best of 3: 3.72 s per loop
     Numpy     10 loops, best of 3: 94.3 ms per loop
     Numexpr   10 loops, best of 3: 30.8 ms per loop
     Numba     100 loops, best of 3: 11.7 ms per loop
     Parakeet  100 loops, best of 3: 22 ms per loop
     Cython    100 loops, best of 3: 7.08 ms per loop
     C         100 loops, best of 3: 7.52 ms per loop
     C++       100 loops, best of 3: 7.58 ms per loop
     Fortran   100 loops, best of 3: 7.28 ms per loop
     Scipy     100 loops, best of 3: 4.26 ms per loop
    


**Final optimization**: Scipy only calculates for i < j < n since the
pairwise distance matrix is symmetric, and hence takes about half the
time of our solution. Can you modify our pdist\_X functions to also
exploit symmetry?

Summary
-------

-  Using C, C++ or Fortran give essentially identcial performance
-  Of the JIT solutions:

   -  Cython is the fastest but needs the extra work of type annotations
   -  numba is almost as fast and simplest to use - just say
      jit(functiion)
   -  numexpr is slightly slower and works best for small numpy
      expressions but is also very convenient

-  A pure numpy solution also perfroms reasonably and will be the
   shortest solutoin (a one-liner in this case)
-  The pure python approach is very slow, but serves as a useful
   template for converting to native langauge directly or via a JIT
   compiler
-  Note that the fsatest alternatives are approximately 1000 times
   faster than the pure python version for the test problem with n=1000
   and p=3.

Recommendations for optimizing Python code
------------------------------------------

-  Does a reliable fast implementiaont already exist? If so, consider
   using that
-  Start with a numpy/python prototype - if this is fast enough, stop
-  See if better use of vectoriazaiton via numpy will help
-  Moving to native code:

   -  Most Python devleopers will use Cython as the tool of choice.
      Cython can also be used to access/wrap C/C++ code
   -  JIT compilation with numba is improving fast and may become
      competitive with Cython in the near future
   -  If the function is "minimal", it is usually worth considering
      numexpr because there is almost no work to be done
   -  Use C/C++/Fortran if you are fluent in those languages - you have
      seen how to call these functions from Python

-  If appropriate, consider parallelization (covered in later session)
-  As you optimize your code, remmeber:

   -  Check that is is giving correct results!
   -  Profile often - it is very hard to preidct the effect of an
      optimizaiton in general
   -  Remember that your time is precious - stop when fast enough
   -  If getting a bigger, faster machine will sovle the problem, that
      is sometimes the best solution

.. code:: python

    %load_ext version_information
    
    %version_information numpy, scipy, numexpr, numba, numbapro, parakeet, cython, f2py,




.. raw:: html

    <table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.9 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>3.1.0</td></tr><tr><td>OS</td><td>Darwin 13.4.0 x86_64 i386 64bit</td></tr><tr><td>numpy</td><td>1.9.2</td></tr><tr><td>scipy</td><td>0.15.1</td></tr><tr><td>numexpr</td><td>2.3.1</td></tr><tr><td>numba</td><td>0.17.0</td></tr><tr><td>numbapro</td><td>0.17.1</td></tr><tr><td>parakeet</td><td>0.23.2</td></tr><tr><td>cython</td><td>0.22</td></tr><tr><td>f2py</td><td>f2py</td></tr><tr><td colspan='2'>Thu Apr 09 09:52:28 2015 EDT</td></tr></table>



