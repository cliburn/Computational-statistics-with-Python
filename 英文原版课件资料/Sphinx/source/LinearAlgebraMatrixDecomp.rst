
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


**Reference**

`SciPy's official tutorial on Linear
algebra <http://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html>`__

Large Linear Systems
====================

This is the age of Big Data. Every second of every day, data is being
recorded in countless systems over the world. Our shopping habits, book
and movie preferences, key words typed into our email messages, medical
records, NSA recordings of our telephone calls, genomic data - and none
of it is any use without analysis.

Enormous data sets carry with them enormous challenges in data
processing. Solving a system of :math:`10` equations in :math:`10`
unknowns is easy, and one need not be terribly careful about methodolgy.
But as the size of the system grows, algorithmic complexity and
efficiency become critical.

Example: Netflix Competition (circa 2006-2009)
----------------------------------------------

For a more complete description:

http://en.wikipedia.org/wiki/Netflix\_Prize

The whole technical story

http://www.stat.osu.edu/~dmsl/GrandPrize2009\_BPC\_BigChaos.pdf

In 2006, Netflix opened a competition where it provided ratings of over
:math:`400,000` for :math:`18,000` movies. The goal was to make predict
a user's rating of a movie, based on previous ratings *and* ratings of
'similar' users. The task amounted to analysis of a
:math:`400,000\times 18,000` matrix! The wikipedia link above describes
the contest and the second link is a very detailed description of the
method (which took into account important characteristics such as how
tastes may change over time). Part of the analysis is related to matrix
decomposition - we won't go into the details of the winning algorithm,
but we will spend some time on basic matrix decompositions.


Matrix Decompositions
=====================

Matrix decompositions are an important step in solving linear systems in
a computationally efficient manner.

LU Decomposition and Gaussian Elimination
-----------------------------------------

LU stands for 'Lower Upper', and so an LU decomposition of a matrix
:math:`A` is a decomposition so that

.. math:: A= LU

where :math:`L` is lower triangular and :math:`U` is upper triangular.

Now, LU decomposition is essentially gaussian elimination, but we work
only with the matrix :math:`A` (as opposed to the augmented matrix).

Let's review how gaussian elimination (ge) works. We will deal with a
:math:`3\times 3` system of equations for conciseness, but everything
here generalizes to the :math:`n\times n` case. Consider the following
equation:

.. math:: \left(\begin{matrix}a_{11}&a_{12} & a_{13}\\a_{21}&a_{22}&a_{23}\\a_{31}&a_{32}&a_{33}\end{matrix}\right)\left(\begin{matrix}x_1\\x_2\\x_3\end{matrix}\right) = \left(\begin{matrix}b_1\\b_2\\b_3\end{matrix}\right)

For simplicity, let us assume that the leftmost matrix :math:`A` is
non-singular. To solve the system using ge, we start with the 'augmented
matrix':

.. math:: \left(\begin{array}{ccc|c}a_{11}&a_{12} & a_{13}& b_1 \\a_{21}&a_{22}&a_{23}&b_2\\a_{31}&a_{32}&a_{33}&b_3\end{array}\right)

We begin at the first entry, :math:`a_{11}`. If :math:`a_{11} \neq 0`,
then we divide the first row by :math:`a_{11}` and then subtract the
appropriate multiple of the first row from each of the other rows,
zeroing out the first entry of all rows. (If :math:`a_{11}` is zero, we
need to permute rows. We will not go into detail of that here.) The
result is as follows:

.. math::

   \left(\begin{array}{ccc|c}
   1 & \frac{a_{12}}{a_{11}} & \frac{a_{13}}{a_{11}} & \frac{b_1}{a_{11}} \\
   0 & a_{22} - a_{21}\frac{a_{12}}{a_{11}} & a_{23} - a_{21}\frac{a_{13}}{a_{11}}  & b_2 - a_{21}\frac{b_1}{a_{11}}\\
   0&a_{32}-a_{31}\frac{a_{12}}{a_{11}} & a_{33} - a_{31}\frac{a_{13}}{a_{11}}  &b_3- a_{31}\frac{b_1}{a_{11}}\end{array}\right)

We repeat the procedure for the second row, first dividing by the
leading entry, then subtracting the appropriate multiple of the
resulting row from each of the third and first rows, so that the second
entry in row 1 and in row 3 are zero. We *could* continue until the
matrix on the left is the identity. In that case, we can then just 'read
off' the solution: i.e., the vector :math:`x` is the resulting column
vector on the right. Usually, it is more efficient to stop at *reduced
row eschelon* form (upper triangular, with ones on the diagonal), and
then use *back substitution* to obtain the final answer.

Note that in some cases, it is necessary to permute rows to obtain
reduced row eschelon form. This is called *partial pivoting*. If we also
manipulate columns, that is called *full pivoting*.

It should be mentioned that we may obtain the inverse of a matrix using
ge, by reducing the matrix :math:`A` to the identity, with the identity
matrix as the augmented portion.

Now, this is all fine when we are solving a system one time, for one
outcome :math:`b`. Many applications involve solutions to multiple
problems, where the left-hand-side of our matrix equation does not
change, but there are many outcome vectors :math:`b`. In this case, it
is more efficient to *decompose* :math:`A`.

First, we start just as in ge, but we 'keep track' of the various
multiples required to eliminate entries. For example, consider the
matrix

.. math::

   A = \left(\begin{matrix} 1 & 3 & 4 \\
                              2& 1& 3\\
                              4&1&2
                              \end{matrix}\right)

We need to multiply row :math:`1` by :math:`2` and subtract from row
:math:`2` to eliminate the first entry in row :math:`2`, and then
multiply row :math:`1` by :math:`4` and subtract from row :math:`3`.
Instead of entering zeroes into the first entries of rows :math:`2` and
:math:`3`, we record the multiples required for their elimination, as
so:

.. math::

   \left(\begin{matrix} 1 & 3 & 4 \\
                              (2)& -5 & -5\\
                              (4)&-11&-14
                              \end{matrix}\right)

And then we eliminate the second entry in the third row:

.. math::

   \left(\begin{matrix} 1 & 3 & 4 \\
                              (2)& -5 & -5\\
                              (4)&(\frac{-11}{5})&-3
                              \end{matrix}\right)

And now we have the decomposition:

.. math::

   L= \left(\begin{matrix} 1 & 0 & 0 \\
                              2& 1 & 0\\
                              4&\frac{-11}5&1
                              \end{matrix}\right)
                             U = \left(\begin{matrix} 1 & 3 & 4 \\
                              0& -5 & -5\\
                              0&0&-3
                              \end{matrix}\right)

We can solve the system by solving two back-substitution problems:

.. math:: Ly = b

and

.. math:: Ux=y

These are both :math:`O(n^2)`, so it is more efficient to decompose when
there are multiple outcomes to solve for.

Let do this with numpy:

.. code:: python

    import numpy as np
    import scipy.linalg as la
    np.set_printoptions(suppress=True) 
    
    A = np.array([[1,3,4],[2,1,3],[4,1,2]])
    
    print(A)
    
    P, L, U = la.lu(A)
    print(np.dot(P.T, A))
    print
    print(np.dot(L, U))
    print(P)
    print(L)
    print(U)


.. parsed-literal::

    [[1 3 4]
     [2 1 3]
     [4 1 2]]
    [[ 4.  1.  2.]
     [ 1.  3.  4.]
     [ 2.  1.  3.]]
    
    [[ 4.  1.  2.]
     [ 1.  3.  4.]
     [ 2.  1.  3.]]
    [[ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]
    [[ 1.      0.      0.    ]
     [ 0.25    1.      0.    ]
     [ 0.5     0.1818  1.    ]]
    [[ 4.      1.      2.    ]
     [ 0.      2.75    3.5   ]
     [ 0.      0.      1.3636]]


Note that the numpy decomposition uses *partial pivoting* (matrix rows
are permuted to use the largest pivot). This is because small pivots can
lead to numerical instability. Another reason why one should use library
functions whenever possible!

Cholesky Decomposition
----------------------

Recall that a square matrix :math:`A` is positive definite if

.. math:: u^TA u > 0

for any non-zero n-dimensional vector :math:`u`,

and a symmetric, positive-definite matrix :math:`A` is a
positive-definite matrix such that

.. math:: A = A^T

Let :math:`A` be a symmetric, positive-definite matrix. There is a
unique decomposition such that

.. math:: A = L L^T

where :math:`L` is lower-triangular with positive diagonal elements and
:math:`L^T` is its transpose. This decomposition is known as the
Cholesky decompostion, and :math:`L` may be interpreted as the 'square
root' of the matrix :math:`A`.

Algorithm:
~~~~~~~~~~

Let :math:`A` be an :math:`n\times n` matrix. We find the matri
:math:`L` using the following iterative procedure:

.. math::

   A = \left(\begin{matrix}a_{11}&A_{12}\\A_{12}&A_{22}\end{matrix}\right) =
   \left(\begin{matrix}\ell_{11}&0\\
   L_{12}&L_{22}\end{matrix}\right)
   \left(\begin{matrix}\ell_{11}&L_{12}\\0&L_{22}\end{matrix}\right)

1.) Let :math:`\ell_{11} = \sqrt{a_{11}}`

2.) :math:`L_{12} = \frac{1}{\ell_{11}}A_{12}`

3.) Solve :math:`A_{22} - L_{12}L_{12}^T = L_{22}L_{22}^T` for
:math:`L_{22}`

Example:
~~~~~~~~

.. math:: A = \left(\begin{matrix}1&3&5\\3&13&23\\5&23&42\end{matrix}\right)

.. math:: \ell_{11} = \sqrt{a_{11}} = 1

.. math:: L_{12} = \frac{1}{\ell_{11}} A_{12} = A_{12}

:math:`\begin{eqnarray} A_22 - L_{12}L_{12}^T &=& \left(\begin{matrix}13&23\\23&42\end{matrix}\right) - \left(\begin{matrix}9&15\\15&25\end{matrix}\right)\\ &=& \left(\begin{matrix}4&8\\8&17\end{matrix}\right)\\ &=& \left(\begin{matrix}2&0\\4&\ell_{33}\end{matrix}\right) \left(\begin{matrix}2&4\\0&\ell_{33}\end{matrix}\right)\\ &=& \left(\begin{matrix}4&8\\8&16+\ell_{33}^2\end{matrix}\right) \end{eqnarray}`

And so we conclude that :math:`\ell_{33}=1`.

This yields the decomposition:

.. math::

   \left(\begin{matrix}1&3&5\\3&13&23\\5&23&42\end{matrix}\right) = 
   \left(\begin{matrix}1&0&0\\3&2&0\\5&4&1\end{matrix}\right)\left(\begin{matrix}1&3&5\\0&2&4\\0&0&1\end{matrix}\right)

Now, with numpy:

.. code:: python

    A = np.array([[1,3,5],[3,13,23],[5,23,42]])
    L = la.cholesky(A)
    print(np.dot(L.T, L))
    
    print(L)
    print(A)


.. parsed-literal::

    [[  1.   3.   5.]
     [  3.  13.  23.]
     [  5.  23.  42.]]
    [[ 1.  3.  5.]
     [ 0.  2.  4.]
     [ 0.  0.  1.]]
    [[ 1  3  5]
     [ 3 13 23]
     [ 5 23 42]]


Cholesky decomposition is about twice as fast as LU decomposition
(though both scale as :math:`n^3`).

Matrix Decompositions for PCA and Least Squares
-----------------------------------------------

Eigendecomposition
------------------

Eigenvectors and Eigenvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First recall that an *eigenvector* of a matrix :math:`A` is a non-zero
vector :math:`v` such that

.. math:: Av = \lambda v

for some scalar :math:`\lambda`

The value :math:`\lambda` is called an *eigenvalue* of :math:`A`.

If an :math:`n\times n` matrix :math:`A` has :math:`n` linearly
independent eigenvectors, then :math:`A` may be decomposed in the
following manner:

.. math:: A = B\Lambda B^{-1}

where :math:`\Lambda` is a diagonal matrix whose diagonal entries are
the eigenvalues of :math:`A` and the columns of :math:`B` are the
corresponding eigenvectors of :math:`A`.

Facts:

-  An :math:`n\times n` matrix is diagonizable :math:`\iff` it has
   :math:`n` linearly independent eigenvectors.
-  A symmetric, positive definite matrix has only positive eigenvalues
   and its eigendecomposition

   .. math:: A=B\Lambda B^{-1}

is via an orthogonal transformation :math:`B`. (I.e. its eigenvectors
are an orthonormal set)

Calculating Eigenvalues
^^^^^^^^^^^^^^^^^^^^^^^

It is easy to see from the definition that if :math:`v` is an
eigenvector of an :math:`n\times n` matrix :math:`A` with eigenvalue
:math:`\lambda`, then

.. math:: Av - \lambda I = \bf{0}

where :math:`I` is the identity matrix of dimension :math:`n` and
:math:`\bf{0}` is an n-dimensional zero vector. Therefore, the
eigenvalues of :math:`A` satisfy:

.. math:: \det\left(A-\lambda I\right)=0

The left-hand side above is a polynomial in :math:`\lambda`, and is
called the *characteristic polynomial* of :math:`A`. Thus, to find the
eigenvalues of :math:`A`, we find the roots of the characteristic
polynomial.

Computationally, however, computing the characteristic polynomial and
then solving for the roots is prohibitively expensive. Therefore, in
practice, numerical methods are used - both to find eigenvalues and
their corresponding eigenvectors. We won't go into the specifics of the
algorithms used to calculate eigenvalues, but here is a numpy example:

.. code:: python

    A = np.array([[0,1,1],[2,1,0],[3,4,5]])
    
    u, V = la.eig(A)
    print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
    print(u)



.. parsed-literal::

    [[-0.+0.j  1.+0.j  1.+0.j]
     [ 2.+0.j  1.+0.j  0.+0.j]
     [ 3.+0.j  4.+0.j  5.+0.j]]
    [ 5.8541+0.j -0.8541+0.j  1.0000+0.j]


**NB:** Many matrices are *not* diagonizable, and many have *complex*
eigenvalues (even if all entries are real).

.. code:: python

    A = np.array([[0,1],[-1,0]])
    print(A)
    
    u, V = la.eig(A)
    print(np.dot(V,np.dot(np.diag(u), la.inv(V))))
    print(u)


.. parsed-literal::

    [[ 0  1]
     [-1  0]]
    [[ 0.+0.j  1.+0.j]
     [-1.+0.j  0.+0.j]]
    [ 0.+1.j  0.-1.j]


.. code:: python

    # If you know the eigenvalues must be reeal 
    # because A is a positive definite (e.g. covariance) matrix 
    # use real_if_close
    
    A = np.array([[0,1,1],[2,1,0],[3,4,5]])
    u, V = la.eig(A)
    print(u)
    print np.real_if_close(u)


.. parsed-literal::

    [ 5.8541+0.j -0.8541+0.j  1.0000+0.j]
    [ 5.8541 -0.8541  1.    ]


Singular Values
^^^^^^^^^^^^^^^

For any :math:`m\times n` matrix :math:`A`, we define its *singular
values* to be the square root of the eigenvalues of :math:`A^TA`. These
are well-defined as :math:`A^TA` is always symmetric, positive-definite,
so its eigenvalues are real and positive. Singular values are important
properties of a matrix. Geometrically, a matrix :math:`A` maps the unit
sphere in :math:`\mathbb{R}^n` to an ellipse. The singular values are
the lengths of the semi-axes.

Singular values also provide a measure of the *stabilty* of a matrix.
We'll revisit this in the end of the lecture.

QR decompositon
---------------

As with the previous decompositions, :math:`QR` decomposition is a
method to write a matrix :math:`A` as the product of two matrices of
simpler form. In this case, we want:

.. math::  A= QR

where :math:`Q` is an :math:`m\times n` matrix with :math:`Q Q^T = I`
(i.e. :math:`Q` is *orthogonal*) and :math:`R` is an :math:`n\times n`
upper-triangular matrix.

This is really just the matrix form of the Gram-Schmidt
orthogonalization of the columns of :math:`A`. The G-S algorithm itself
is unstable, so various other methods have been developed to compute the
QR decomposition. We won't cover those in detail as they are a bit
beyond our scope.

The first :math:`k` columns of :math:`Q` are an orthonormal basis for
the column space of the first :math:`k` columns of :math:`A`.

Iterative QR decomposition is often used in the computation of
eigenvalues.

Singular Value Decomposition
----------------------------

Another important matrix decomposition is singular value decomposition
or SVD. For any :math:`m\times n` matrix :math:`A`, we may write:

.. math:: A= UDV

where :math:`U` is a unitary (orthogonal in the real case)
:math:`m\times m` matrix, :math:`D` is a rectangular, diagonal
:math:`m\times n` matrix with diagonal entries :math:`d_1,...,d_m` all
non-negative. :math:`V` is a unitary (orthogonal) :math:`n\times n`
matrix. SVD is used in principle component analysis and in the
computation of the Moore-Penrose pseudo-inverse.

Stabilty and Condition Number
-----------------------------

It is important that numerical algorithms be *stable* and *efficient*.
Efficiency is a property of an algorithm, but stability can be a
property of the system itself.

Example
~~~~~~~

.. math:: \left(\begin{matrix}8&6&4&1\\1&4&5&1\\8&4&1&1\\1&4&3&6\end{matrix}\right)x = \left(\begin{matrix}19\\11\\14\\14\end{matrix}\right)

.. code:: python

    A = np.array([[8,6,4,1],[1,4,5,1],[8,4,1,1],[1,4,3,6]])
    b = np.array([19,11,14,14])
    la.solve(A,b)




.. parsed-literal::

    array([ 1.,  1.,  1.,  1.])



.. code:: python

    b = np.array([19.01,11.05,14.07,14.05])
    la.solve(A,b)




.. parsed-literal::

    array([-2.34 ,  9.745, -4.85 , -1.34 ])



Note that the *tiny* perturbations in the outcome vector :math:`b` cause
*large* differences in the solution! When this happens, we say that the
matrix :math:`A` *ill-conditioned*. This happens when a matrix is
'close' to being singular (i.e. non-invertible).

Condition Number
~~~~~~~~~~~~~~~~

A measure of this type of behavior is called the *condition number*. It
is defined as:

.. math::  cond(A) = ||A||\cdot ||A^{-1}|| 

In general, it is difficult to compute.

Fact:

.. math:: cond(A) = \frac{\lambda_1}{\lambda_n}

where :math:`\lambda_1` is the maximum singular value of :math:`A` and
:math:`\lambda_n` is the smallest. The higher the condition number, the
more unstable the system. In general if there is a large discrepancy
between minimal and maximal singular values, the condition number is
large.

Example
^^^^^^^

.. code:: python

    U, s, V = np.linalg.svd(A)
    print(s)
    print(max(s)/min(s))


.. parsed-literal::

    [ 15.5457   6.9002   3.8363   0.0049]
    3198.6725812


Preconditioning
^^^^^^^^^^^^^^^

We can sometimes improve on this behavior by 'pre-conditioning'. Instead
of solving

.. math:: Ax=b

we solve

.. math:: D^{-1}Ax=D^{-1}b

 where :math:`D^{-1}A` has a lower condition number than :math:`A`
itself.

Preconditioning is a *very* involved topic, quite out of the range of
this course. It is mentioned here only to make you aware that such a
thing exists, should you ever run into an ill-conditioned problem!

Exercises
---------

**1**. Compute the LU decomposition of the following matrix by hand and
using numpy

.. math:: \left(\begin{matrix}1&2&3\\2&-4&6\\3&-9&-3\end{matrix}\right)

Solution:

First by hand:

**2**. Compute the Cholesky decomposition of the following matrix by
hand and using numpy

.. math:: \left(\begin{matrix}1&2&3\\2&-4&6\\3&6&-3\end{matrix}\right)

.. code:: python

    # Your code here

**3**. Write a function in Python to solve a system

.. math:: Ax = b

using SVD decomposition. Your function should take :math:`A` and
:math:`b` as input and return :math:`x`.

Your function should include the following:

-  First, check that :math:`A` is invertible - return error message if
   it is not
-  Invert :math:`A` using SVD and solve
-  return :math:`x`

Test your function for correctness.

.. code:: python

    # Your code here
    
    def svdsolver(A,b):
        U, s, V = np.linalg.svd(A)
        if np.prod(s) == 0:
           print("Matrix is singular")
        else:
           return np.dot(np.dot((V.T).dot(np.diag(s**(-1))), U.T),b)
            

.. code:: python

    A = np.array([[1,1],[1,2]])
    b = np.array([3,1])
    print(np.linalg.solve(A,b))
    print(svdsolver(A,b))



.. parsed-literal::

    [ 5. -2.]
    [ 5. -2.]


