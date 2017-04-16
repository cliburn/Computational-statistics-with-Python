
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

Linear Algebra and Linear Systems
=================================

A lot of problems in statistical computing can be described
mathematically using linear algebra. This lecture is meant to serve as a
review of concepts you have covered in linear algebra courses.

Simultaneous Equations
----------------------

Consider a set of :math:`m` linear equations in :math:`n` unknowns:

.. raw:: latex

   \begin{align*}
   a_{11} x_1 + &a_{12} x_2& +& ... + &a_{1n} x_n &=& b_1\\
   \vdots  && &&\vdots &= &\vdots\\
   a_{m1} x_1 + &a_{m2} x_2& +& ... + &a_{mn} x_n &=&b_m 
   \end{align*}

We can let:

.. raw:: latex

   \begin{align*}
       A=\left[\begin{matrix}a_{11}&\cdots&a_{1n}\\
                  \vdots & &\vdots\\
                  a_{m1}&\cdots&a_{mn}\end{matrix}\right] & &
       x = \left[\begin{matrix}x_1 \\
                  \vdots\\
                  x_n\end{matrix}\right] & \;\;\;\;\textrm{   and } &
       b =  \left[\begin{matrix}b_1\\
                  \vdots\\
                  b_m\end{matrix}\right]
   \end{align*}

And re-write the system:

.. math::  Ax = b

This reduces the problem to a matrix equation, and now solving the
system amounts to finding :math:`A^{-1}` (or sort of). Certain properies
of the matrix :math:`A` yield important information about the linear
system.

Underdetermined System (:math:`m<n`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`m<n`, the linear system is said to be *underdetermined*.
I.e. there are fewer equations than unknowns. In this case, there are
either no solutions (if the system is *inconsistent*) or infinite
solutions. A unique solution is not possible.

Overdetermined System
~~~~~~~~~~~~~~~~~~~~~

When :math:`m>n`, the system may be *overdetermined*. In other words,
there are more equations than unknowns. They system could be
inconsistent, or some of the equations could be redundant.
Statistically, you can translate these situations to 'least squares
solution' or 'overparametrized'.

There are many techniques to solve and analyze linear systems. Our goal
is to understand the theory behind many of the built-in functions, and
how they *efficiently* solve systems of equations.

First, let's review some linear algebra topics:

Linear Independence
-------------------

A collection of vectors :math:`v_1,...,v_n` is said to be *linearly
independent* if

.. math:: c_1v_1 + \cdots c_nv_n = 0

.. math:: \iff

.. math:: c_1=\cdots=c_n=0

In other words, any linear combination of the vectors that results in a
zero vector is trivial.

Another interpretation of this is that no vector in the set may be
expressed as a linear combination of the others. In this sense, linear
independence is an expression of non-redundancy in a set of vectors.

Fact: Any linearly independent set of :math:`n` vectors spans an
:math:`n`-dimensional space. (I.e. the collection of all possible linear
combinations is :math:`\mathbb{R}^n`.) Such a set of vectors is said to
be a *basis* of :math:`\mathbb{R}^n`. Another term for basis is *minimal
spanning set*.

What does that have to do with linear systems?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A LOT!!**

-  If :math:`A` is an :math:`m\times n` matrix and :math:`m>n`, if all
   :math:`m` rows are linearly independent, then the system is
   *overdetermined* and *inconsistent*. The system cannot be solved
   exactly. This is the usual case in data analysis, and why least
   squares is so important.

-  If :math:`A` is an :math:`m\times n` matrix and :math:`m<n`, if all
   :math:`m` rows are linearly independent, then the system is
   *underdetermined* and there are *infinite* solutions.

-  If :math:`A` is an :math:`m\times n` matrix and some of its rows are
   linearly dependent, then the system is *reducible*. We can get rid of
   some equations.

-  If :math:`A` is a square matrix and its rows are linearly
   independent, the system has a unique solution. (:math:`A` is
   invertible.)

Linear algebra has a whole lot more to tell us about linear systems, so
we'll review a few basics.

Norms and Distance of Vectors
-----------------------------

Recall that the 'norm' of a vector :math:`v`, denoted :math:`||v||` is
simply its length. For a vector with components

.. math:: v = \left(v_1,...,v_n\right)

the norm of :math:`v` is given by:

.. math:: ||v|| = \sqrt{v_1^2+...+v_n^2}

The distance between two vectors is the length of their difference:

.. math:: d(v,w) = ||v-w||

Examples
^^^^^^^^

.. code:: python

    import numpy as np
    from scipy import linalg
    
    
    # norm of a vector
    
    v = np.array([1,2])
    linalg.norm(v)




.. parsed-literal::

    2.2361



.. code:: python

    # distance between two vectors
    
    w = np.array([1,1])
    linalg.norm(v-w)




.. parsed-literal::

    1.0000



Inner Products
~~~~~~~~~~~~~~

Inner products are closely related to norms and distance. The (standard)
inner product of two :math:`n` dimensional vectors :math:`v` and
:math:`w` is given by:

.. math:: <v,w> = v_1w_1+...+v_nw_n

I.e. the inner product is just the sum of the product of the components.
Certain 'special' matrices also define inner products, and we will see
some of those later.

Any inner product determines a norm via:

.. math:: ||v|| = <v,v>^{\frac12}

There is a more abstract formulation of an inner product, that is useful
when considering more general vector spaces, especially function vector
spaces:

An inner product on a vector space :math:`V` is a symmetric, positive
definite, bilinear form.

There is also a more abstract definition of a norm - a norm is function
from a vector space to the real numbers, that is positive definite,
absolutely scalable and satisfies the triangle inequality.

What is important here is not to memorize these definitions - just to
realize that 'norm' and 'inner product' can be defined for things that
are not tuples in :math:`\mathbb{R}^n`. (In particular, they can be
defined on vector spaces of *functions*).

Example
^^^^^^^

.. code:: python

    v.dot(w)




.. parsed-literal::

    3



Outer Products
~~~~~~~~~~~~~~

Note that the inner product is just matrix multiplication of a
:math:`1\times n` vector with an :math:`n\times 1` vector. In fact, we
may write:

.. math:: <v,w> = v^tw

The *outer product* of two vectors is just the opposite. It is given by:

.. math:: v\otimes w = vw^t

Note that I am considering :math:`v` and :math:`w` as *column* vectors.
The result of the inner product is a *scalar*. The result of the outer
product is a *matrix*.

Example
^^^^^^^

.. code:: python

    np.outer(v,w)




.. parsed-literal::

    array([[1, 1],
           [2, 2]])



**Extended example**: the covariance matrix is an outer proudct.

.. code:: python

    import numpy as np
    
    # We have n observations of p variables 
    n, p = 10, 4
    v = np.random.random((p,n))

.. code:: python

    # The covariance matrix is a p by p matrix
    np.cov(v)




.. parsed-literal::

    array([[ 0.1055, -0.0437,  0.0352, -0.0152],
           [-0.0437,  0.055 , -0.0126,  0.0324],
           [ 0.0352, -0.0126,  0.1016,  0.0552],
           [-0.0152,  0.0324,  0.0552,  0.1224]])



.. code:: python

    # From the definition, the covariance matrix 
    # is just the outer product of the normalized 
    # matrix where every variable has zero mean
    # divided by the number of degrees of freedom
    w = v - v.mean(1)[:, np.newaxis]
    w.dot(w.T)/(n - 1)




.. parsed-literal::

    array([[ 0.1055, -0.0437,  0.0352, -0.0152],
           [-0.0437,  0.055 , -0.0126,  0.0324],
           [ 0.0352, -0.0126,  0.1016,  0.0552],
           [-0.0152,  0.0324,  0.0552,  0.1224]])



Trace and Determinant of Matrices
---------------------------------

The trace of a matrix :math:`A` is the sum of its diagonal elements. It
is important for a couple of reasons:

-  It is an *invariant* of a matrix under change of basis (more on this
   later).
-  It defines a matrix norm (more on that later)

The determinant of a matrix is defined to be the alternating sum of
permutations of the elements of a matrix. Let's not dwell on that
though. It is important to know that the determinant of a
:math:`2\times 2` matrix is

.. math:: \left|\begin{matrix}a_{11} & a_{12}\\a_{21} & a_{22}\end{matrix}\right| = a_{11}a_{22} - a_{12}a_{21}

This may be extended to an :math:`n\times n` matrix by minor expansion.
I will leave that for you to google. We will be computing determinants
using tools such as:

``np.linalg.det(A)``

What is most important about the determinant:

-  Like the trace, it is also invariant under change of basis
-  An :math:`n\times n` matrix :math:`A` is invertible :math:`\iff`
   det\ :math:`(A)\neq 0`
-  The rows(columns) of an :math:`n\times n` matrix :math:`A` are
   linearly independent :math:`\iff` det\ :math:`(A)\neq 0`

.. code:: python

    n = 6
    M = np.random.randint(100,size=(n,n))
    print(M)
    np.linalg.det(M)


.. parsed-literal::

    [[61 36 46 92 50 76]
     [83 63 14 97 17 62]
     [17 26 12 94 61 50]
     [66  9 11 73  1 13]
     [37 98 82 69  3 65]
     [51 15  7 25 85 72]]




.. parsed-literal::

    36971990469.0001



Column space, Row space, Rank and Kernel
----------------------------------------

Let :math:`A` be an :math:`m \times n` matrix. We can view the columns
of :math:`A` as vectors, say :math:`a_1, \dots,, a_n`. The space of all
linear combinations of the :math:`a_i` are the *column space* of the
matrix :math:`A`. Now, if :math:`a_1, \dots ,a_n` are *linearly
independent*, then the column space is of dimension :math:`n`.
Otherwise, the dimension of the column space is the size of the maximal
set of linearly independent :math:`a_i`. Row space is exactly analogous,
but the vectors are the *rows* of :math:`A`.

The *rank* of a matrix *A* is the dimension of its column space - and -
the dimension of its row space. These are equal for any matrix. Rank can
be thought of as a measure of non-degeneracy of a system of linear
equations, in that it is the *dimension of the image of the linear
transformation* determined by :math:`A`.

The *kernel* of a matrix *A* is the dimension of the space mapped to
zero under the linear transformation that :math:`A` represents. The
dimension of the kernel of a linear transformation is called the
*nullity*.

Index theorem: For an :math:`m\times n` matrix :math:`A`,

rank(\ :math:`A`) + nullity(\ :math:`A`) = :math:`n`.

Matrices as Linear Transformations
----------------------------------

Let's consider: what does a matrix *do* to a vector? Matrix
multiplication has a *geometric* interpretation. When we multiply a
vector, we either rotate, reflect, dilate or some combination of those
three. So multiplying by a matrix *transforms* one vector into another
vector. This is known as a *linear transformation*.

Important Facts:

-  Any matrix defines a linear transformation
-  The matrix form of a linear transformation is NOT unique
-  We need only define a transformation by saying what it does to a
   *basis*

Suppose we have a matrix :math:`A` that defines some transformation. We
can take any invertible matrix :math:`B` and

.. math:: BAB^{-1}

defines the same transformation. This operation is called a *change of
basis*, because we are simply expressing the transformation with respect
to a different basis.

This is what we do in PCA. We express the matrix in a basis of
eigenvectors (more on this later).

Example:
^^^^^^^^

Let :math:`f(x)` be the linear transformation that takes
:math:`e_1=(1,0)` to :math:`f(e_1)=(2,3)` and :math:`e_2=(0,1)` to
:math:`f(e_2) = (1,1)`. A matrix representation of :math:`f` would be
given by:

.. math:: A = \left(\begin{matrix}2 & 1\\3&1\end{matrix}\right)

This is the matrix we use if we consider the vectors of
:math:`\mathbb{R}^2` to be linear combinations of the form

.. math:: c_1 e_1 + c_2 e_2

Now, consider a second pair of (linearly independent) vectors in
:math:`\mathbb{R}^2`, say :math:`v_1=(1,3)` and :math:`v_2=(4,1)`. We
first find the transformation that takes :math:`e_1` to :math:`v_1` and
:math:`e_2` to :math:`v_2`. A matrix representation for this is:

.. math:: B = \left(\begin{matrix}1 & 4\\3&1\end{matrix}\right)

Our original transformation :math:`f` can be expressed with respect to
the basis :math:`v_1, v_2` via

.. math:: BAB^{-1}

.. code:: python

    
    
    A = np.array([[2,1],[3,1]])  # transformation f in standard basis
    e1 = np.array([1,0])         # standard basis vectors e1,e2
    e2 = np.array([0,1])
    
    print(A.dot(e1))             # demonstrate that Ae1 is (2,3)
    print(A.dot(e2))             # demonstrate that Ae2 is (1,1) 
                                  
    # new basis vectors
    v1 = np.array([1,3])         
    v2 = np.array([4,1])
    
    # How v1 and v2 are transformed by A
    print("Av1: ")
    print(A.dot(v1))   
    print("Av2: ")
    print(A.dot(v2))
    
    # Change of basis from standard to v1,v2
    B = np.array([[1,4],[3,1]])
    print(B)
    B_inv = linalg.inv(B)
    
    print("B B_inv ")
    print(B.dot(B_inv))   # check inverse
    
    # transform e1 under change of coordinates
    T = B.dot(A.dot(B_inv))        # B A B^{-1}  
    coeffs = T.dot(e1)
    
    print(coeffs[0]*v1 + coeffs[1]*v2)
    
    



.. parsed-literal::

    [2 3]
    [1 1]
    Av1: 
    [5 6]
    Av2: 
    [ 9 13]
    [[1 4]
     [3 1]]
    B B_inv 
    [[  1.0000e+00   0.0000e+00]
     [  5.5511e-17   1.0000e+00]]
    [ 1.1818  0.5455]


.. code:: python

    def plot_vectors(vs):
        """Plot vectors in vs assuming origin at (0,0)."""
        n = len(vs)
        X, Y = np.zeros((n, 2))
        U, V = np.vstack(vs).T
        plt.quiver(X, Y, U, V, range(n), angles='xy', scale_units='xy', scale=1)
        xmin, xmax = np.min([U, X]), np.max([U, X])
        ymin, ymax = np.min([V, Y]), np.max([V, Y])
        xrng = xmax - xmin
        yrng = ymax - ymin
        xmin -= 0.05*xrng
        xmax += 0.05*xrng
        ymin -= 0.05*yrng
        ymax += 0.05*yrng
        plt.axis([xmin, xmax, ymin, ymax])

.. code:: python

    e1 = np.array([1,0])
    e2 = np.array([0,1])
    A = np.array([[2,1],[3,1]])

.. code:: python

    # Here is a simple plot showing Ae_1 and Ae_2
    # You can show other transofrmations if you like
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plot_vectors([e1, e2])
    plt.subplot(1,2,2)
    plot_vectors([A.dot(e1), A.dot(e2)])
    plt.tight_layout()



.. image:: LinearAlgebraReview_files/LinearAlgebraReview_52_0.png


Matrix Norms
------------

We can extend the notion of a norm of a vector to a norm of a matrix.
Matrix norms are used in determining the *condition* of a matrix (we
will define this in the next lecture.) There are many matrix norms, but
three of the most common are so called 'p' norms, and they are based on
p-norms of vectors. So, for an :math:`n`-dimensional vector :math:`v`
and for :math:`1\leq p <\infty`

.. math:: ||v||_p = \left(\sum\limits_{i=1}^n |v_i|^p\right)^\frac1p

and for :math:`p =\infty`:

.. math:: ||v||_\infty = \max{|v_i|}

Similarly, the corresponding matrix norms are:

.. math:: ||A||_p = \sup_x \frac{||Ax||_p}{||x||_p}

.. math:: ||A||_{1} = \max_j\left(\sum\limits_{i=1}^n|a_{ij}|\right)

(column sum)

.. math:: ||A||_{\infty} = \max_i\left(\sum\limits_{j=1}^n|a_{ij}|\right)

(row sum)

FACT: The matrix 2-norm, :math:`||A||_2` is given by the largest
eigenvalue of :math:`\left(A^TA\right)^\frac12` - otherwise known as the
largest singular value of :math:`A`. We will define eigenvalues and
singular values formally in the next lecture.

Another norm that is often used is called the Frobenius norm. It one of
the simplests to compute:

.. math:: ||A||_F = \left(\sum\sum \left(a_{ij}\right)^2\right)^\frac12

Special Matrices
----------------

Some matrices have interesting properties that allow us either simplify
the underlying linear system or to understand more about it.

Square Matrices
^^^^^^^^^^^^^^^

Square matrices have the same number of columns (usually denoted
:math:`n`). We refer to an arbitrary square matrix as and
:math:`n\times n` or we refer to it as a 'square matrix of dimension
:math:`n`'. If an :math:`n\times n` matrix :math:`A` has *full rank*
(i.e. it has rank :math:`n`), then :math:`A` is invertible, and its
inverse is unique. This is a situation that leads to a unique solution
to a linear system.

Diagonal Matrices
^^^^^^^^^^^^^^^^^

A diagonal matrix is a matrix with all entries off the diagonal equal to
zero. Strictly speaking, such a matrix should be square, but we can also
consider rectangular matrices of size :math:`m\times n` to be diagonal,
if all entries :math:`a_{ij}` are zero for :math:`i\neq j`

Symmetric and Skew Symmetric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A matrix :math:`A` is (skew) symmetric iff :math:`a_{ij} = (-)a_{ji}`.

Equivalently, :math:`A` is (skew) symmetric iff

.. math:: A = (-)A^T

Upper and Lower Triangular
^^^^^^^^^^^^^^^^^^^^^^^^^^

A matrix :math:`A` is (upper\|lower) triangular if :math:`a_{ij} = 0`
for all :math:`i (>|<) j`

Banded and Sparse Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^

These are matrices with lots of zero entries. Banded matrices have
non-zero 'bands', and this structure can be exploited to simplify
computations. Sparse matrices are matrices where there are 'few'
non-zero entries, but there is no pattern to where non-zero entries are
found.

Orthogonal and Orthonormal
^^^^^^^^^^^^^^^^^^^^^^^^^^

A matrix :math:`A` is *orthogonal* iff

.. math:: A A^T = I

In other words, :math:`A` is orthogonal iff

.. math:: A^T=A^{-1}

Facts:

-  The rows and columns of an orthogonal matrix are an orthonormal set
   of vectors.
-  Geometrically speaking, orthogonal transformations preserve lengths
   and angles between vectors

Positive Definite
^^^^^^^^^^^^^^^^^

Positive definite matrices are an important class of matrices with very
desirable properties. A square matrix :math:`A` is positive definite if

.. math:: u^TA u > 0

for any non-zero n-dimensional vector :math:`u`.

A symmetric, positive-definite matrix :math:`A` is a positive-definite
matrix such that

.. math:: A = A^T

IMPORTANT:

-  Symmetric, positive-definite matrices have 'square-roots' (in a
   sense)
-  Any symmetric, positive-definite matrix is *diagonizable*!!!
-  Co-variance matrices are symmetric and positive-definite

Now that we have the basics down, we can move on to numerical methods
for solving systems - aka matrix decompositions.

Exercises
---------

**1**. Determine whether the following system of equations has no
solution, infinite solutions or a unique solution *without solving the
system*

.. raw:: latex

   \begin{eqnarray}
   x+2y-z+w &=& 2\\
   3x-4y+2 w &=& 3\\
   2y+z &=& 4\\
   2x+2y-3z+2w&=&0\\
   -2x+6y-z-w&=&-1
   \end{eqnarray}

.. code:: python

    A = np.array([[1,2,-1,1,2],[3,-4,0,2,3],[0,2,1,0,4],[2,2,-3,2,0],[-2,6,-1,-1,-1]])
    
    np.linalg.matrix_rank(A)
    np.linalg.det(A)




.. parsed-literal::

    0.0000



**2**. Let :math:`f(x)` be a linear transformation of
:math:`\mathbb{R}^3` such that

.. raw:: latex

   \begin{eqnarray}
   f(e_1) &=& (1,1,3)\\
   f(e_2) &=& (1,0,4)\\
   f(e_3) &=& (0,2,1)
   \end{eqnarray}

-  Find a matrix representation for :math:`f`.
-  Compute the matrix representation for :math:`f` in the basis

.. raw:: latex

   \begin{eqnarray}
   v_1 &=& (2,3,3)\\
   v_2 &=& (8,5,2)\\
   v_3 &=& (1,0,5)
   \end{eqnarray}

