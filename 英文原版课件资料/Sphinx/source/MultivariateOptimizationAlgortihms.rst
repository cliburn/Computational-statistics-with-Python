
Algorithms for Optimization and Root Finding for Multivariate Problems
======================================================================

Optimizers
----------

Newton-Conjugate Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~

First a note about the interpretations of Newton's method in 1-D:

In the lecture on 1-D optimization, Newton's method was presented as a
method of finding zeros. That is what it is, but it may also be
interpreted as a method of optimization. In the latter case, we are
really looking for zeroes of the first derivative.

Let's compare the formulas for clarification:

.. math::

   \begin{array}{|c|c|c|c|}
   \hline
   \text{Finding roots of } f  & \text{Geometric Interpretation} & \text{Finding Extrema of } f & \text{Geometric Interpretation} \\
   \hline
   x_{n+1} = x_n -\frac{f(x_n)}{f'(x_n)} &\text{Invert linear approximation to }f & x_{n+1} = x_n -\frac{f'(x_n)}{f''(x_n)}& \text{Use quadratic approximation of } f \\
   \hline
   \end{array}

These are two ways of looking at exactly the same problem. For instance,
the linear approximation in the root finding problem is simply the
derivative function of the quadratic approximation in the optimization
problem.

Hessians, Gradients and Forms - Oh My!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's review the theory of optimization for multivariate functions.
Recall that in the single-variable case, extreme values (local extrema)
occur at points where the first derivative is zero, however, the
vanishing of the first derivative is not a sufficient condition for a
local max or min. Generally, we apply the second derivative test to
determine whether a candidate point is a max or min (sometimes it fails
- if the second derivative either does not exist or is zero). In the
multivariate case, the first and second derivatives are *matrices*. In
the case of a scalar-valued function on :math:`\mathbb{R}^n`, the first
derivative is an :math:`n\times 1` vector called the *gradient* (denoted
:math:`\nabla f`). The second derivative is an :math:`n\times n` matrix
called the *Hessian* (denoted :math:`H`)

Just to remind you, the gradient and Hessian are given by:

.. math:: \nabla f(x) = \left(\begin{matrix}\frac{\partial f}{\partial x_1}\\ \vdots \\\frac{\partial f}{\partial x_n}\end{matrix}\right)

.. math::

   H = \left(\begin{matrix}
     \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
     \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
     \vdots & \vdots & \ddots & \vdots \\[2.2ex]
     \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
   \end{matrix}\right)

One of the first things to note about the Hessian - it's symmetric. This
structure leads to some useful properties in terms of interpreting
critical points.

The multivariate analog of the test for a local max or min turns out to
be a statement about the gradient and the Hessian matrix. Specifically,
a function :math:`f:\mathbb{R}^n\rightarrow \mathbb{R}` has a critical
point at :math:`x` if :math:`\nabla f(x) = 0` (where zero is the zero
vector!). Furthermore, the second derivative test at a critical point is
as follows:

-  If :math:`H(x)` is positive-definite, :math:`f` has a local minimum
   at :math:`x`
-  If :math:`H(x)` is negative-definite, :math:`f` has a local maximum
   at :math:`x`
-  If :math:`H(x)` has both positive and negative eigenvalues, :math:`f`
   has a saddle point at :math:`x`.

Newton CG Algorithm
~~~~~~~~~~~~~~~~~~~

Features:

-  Minimizes a 'true' quadratic on :math:`\mathbb{R}^n` in :math:`n`
   steps
-  Does NOT require storage or inversion of an :math:`n \times n`
   matrix.

We begin with :math:`:\mathbb{R}^n\rightarrow \mathbb{R}`. Take a
quadratic approximation to :math:`f`:

.. math:: f(x) \approx \frac12 x^T H x + b^Tx + c

Note that in the neighborhood of a minimum, :math:`H` will be
positive-definite (and symmetric). (If we are maximizing, just consider
:math:`-H`).

This reduces the optimization problem to finding the zeros of

.. math:: Hx = -b

This is a linear problem, which is nice. The dimension :math:`n` may be
very large - which is not so nice. Also, *a priori* it looks like we
need to know :math:`H`. We actually don't but that will become clear
only after a bit of explanation.

General Inner Product
^^^^^^^^^^^^^^^^^^^^^

Recall the axiomatic definition of an inner product :math:`<,>_A`:

-  For any two vectors :math:`v,w` we have

   .. math:: <v,w>_A = <w,v>_A

-  For any vector :math:`v`

   .. math:: <v,v>_A \;\geq 0

   with equality :math:`\iff` :math:`v=0`.
-  For :math:`c\in\mathbb{R}` and :math:`u,v,w\in\mathbb{R}^n`, we have

   .. math:: <cv+w,u> = c<v,u> + <w,u>

These properties are known as symmetric, positive definite and bilinear,
respectively.

Fact: If we denote the standard inner product on :math:`\mathbb{R}^n` as
:math:`<,>` (this is the 'dot product'), any symmetric, positive
definite :math:`n\times n` matrix :math:`A` defines an inner product on
:math:`\mathbb{R}^n` via:

.. math:: <v,w>_A \; = <v,Aw> = v^TAw

Just as with the standard inner product, general inner products define
for us a notion of 'orthogonality'. Recall that with respect to the
standard product, 2 vectors are orthogonal if their product vanishes.
The same applies to :math:`<,>_A`:

.. math:: <v,w>_A = 0 

means that :math:`v` and :math:`w` are orthogonal under the inner
product induced by :math:`A`. Equivalently, if :math:`v,w` are
orthogonal under :math:`A`, we have:

.. math:: v^TAw = 0

This is also called *conjugate* (thus the name of the method).

Conjugate Vectors
^^^^^^^^^^^^^^^^^

Suppose we have a set of :math:`n` vectors :math:`p_1,...,p_n` that are
mutually conjugate. These vectors form a basis of :math:`\mathbb{R}^n`.
Getting back to the problem at hand, this means that our solution vector
:math:`x` to the linear problem may be written as follows:

.. math:: x = \sum\limits_{i=1}^n \alpha_i p_i

So, finding :math:`x` reduces to finding a conjugate basis and the
coefficients for :math:`x` in that basis.

Note that:

.. math:: {p}_k^{T} {b}={p}_k^{T} {A}{x}

and because :math:`x = \sum\limits_{i=1}^n \alpha_i p_i`, we have:

.. math:: p^TAx = \sum\limits_{i=1}^n \alpha_i p^TA p_i

we can solve for :math:`\alpha_k`:

.. math:: \alpha_k = \frac{{p}_k^{T}{b}}{{p}_k^{T} {A}{p}_k} = \frac{\langle {p}_k, {b}\rangle}{\,\,\,\langle {p}_k,  {p}_k\rangle_{A}} = \frac{\langle{p}_k, {b}\rangle}{\,\,\,\|{p}_k\|_{A}^2}.

Now, all we need are the :math:`p_k`'s.

A nice initial guess would be the gradient at some initial point
:math:`x_1`. So, we set :math:`p_1 = \nabla f(x_1)`. Then set:

.. math:: x_2 = x_1 + \alpha_1p_1

This should look familiar. In fact, it is gradient descent. For
:math:`p_2`, we want :math:`p_1` and :math:`p_2` to be conjugate (under
:math:`A`). That just means orthogonal under the inner product induced
by :math:`A`. We set

.. math:: p_2 = \nabla f(x_1) - \frac{p_1^TA\nabla f(x_1)}{{p}_1^{T}{A}{p}_1} {p}_1

I.e. We take the gradient at :math:`x_1` and subtract its projection
onto :math:`p_1`. This is the same as Gram-Schmidt orthogonalization.

The :math:`k^{th}` conjugate vector is:

.. math:: p_{k+1} = \nabla f(x_k) - \sum\limits_{i=1}^k\frac{p_i^T A \nabla f(x_k)}{p_i^TAp_i} p_i

The 'trick' is that in general, we do not need all :math:`n` conjugate
vectors.

Convergence rate is dependent on sparsity and condition number of
:math:`A`. Worst case is :math:`n^2`.

BFGS - Broyden–Fletcher–Goldfarb–Shanno
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BFGS is a 'quasi' Newton method of optimization. Such methods are
variants of the Newton method, where the Hessian :math:`H` is replaced
by some approximation. We we wish to solve the equation:

.. math:: B_k{p}_k = -\nabla f({x}_k)

for :math:`p_k`. This gives our search direction, and the next candidate
point is given by:

.. math:: x_{k+1} = x_k + \alpha_k p_k

.

where :math:`\alpha_k` is a step size.

At each step, we require that the new approximate :math:`H` meets the
secant condition:

.. math:: B_{k+1}(x_{k+1}-x_k) = \nabla f(x_{k+1}) -\nabla f(x_k)

There is a unique, rank one update that satisfies the above:

.. math:: B_{k+1} = B_k + c_k v_kv_k^T

where

.. math::  c_k = -\frac{1}{\left(B_k(x_{k+1}-x_k) - (\nabla f(x_{k+1})-\nabla f(x_k)\right)^T (x_{k+1}-x_k) }

and

.. math:: v_k = B_k(x_{k+1}-x_k) - (\nabla f(x_{k+1})-\nabla f(x_k))

Note that the update does NOT preserve positive definiteness if
:math:`c_k<0`. In this case, there are several options for the rank one
correction, but we will not address them here. Instead, we will describe
the BFGS method, which almost always guarantees a positive-definite
correction. Specifically:

.. math:: B_{k+1} = B_k + b_k g_k g_k^T + c_k B_k d_k d_k^TB_k

where we have introduced the shorthand:

.. math:: g_k = \nabla f(x_{k+1}) - \nabla f(x_k) \;\;\;\;\;\;\;\ \mathrm{ and }\;\;\;\;\;\;\; d_k = x_{k+1} - x_k

If we set:

.. math:: b_k = \frac{1}{g_k^Td_k} \;\;\;\;\; \mathrm{ and } \;\;\;\;\; c_k = \frac{1}{d_k^TB_kd_k}

we satisfy the secant condition.

Nelder-Mead Simplex
~~~~~~~~~~~~~~~~~~~

While Newton's method is considered a 'second order method' (requires
the second derivative), and quasi-Newton methods are first order
(require only first derivatives), Nelder-Mead is a zero-order method.
I.e. NM requires only the function itself - no derivatives.

For :math:`f:\mathbb{R}^n\rightarrow \mathbb{R}`, the algorithm computes
the values of the function on a simplex of dimension :math:`n`,
constructed from :math:`n+1` vertices. For a univariate function, the
simplex is a line segment. In two dimensions, the simplex is a triangle,
in 3D, a tetrahedral solid, and so on.

The algorithm begins with :math:`n+1` starting points and then the
follwing steps are repeated until convergence:

-  Compute the function at each of the points
-  Sort the function values so that

   .. math:: f(x_1)\leq ...\leq f(x_{n+1})

-  Compute the centroid :math:`x_c` of the n-dimensional region defined
   by :math:`x_1,...,x_n`
-  Reflect :math:`x_{n+1}` about the centroid to get :math:`x_r`

   .. math:: x_r = x_c + \alpha (x_c - x_{n+1})

-  Create a new simplex according to the following rules:

   -  If :math:`f(x_1)\leq f(x_r) < f(x_n)`, replace :math:`x_{n+1}`
      with :math:`x_r`
   -  If :math:`f(x_r)<f(x_1)`, expand the simplex through :math:`x_r`:

      .. math:: x_e = x_c + \gamma (x_c - x_{n+1})

      If :math:`f(x_e)<f(x_r)`, replace :math:`x_{n+1}` with
      :math:`x_e`, otherwise, replace :math:`x_{n+1}` with :math:`x_r`
   -  If :math:`f({x}_{r}) \geq f({x}_{n})`, compute
      :math:`x_p = x_c + \rho(x_c - x_{n+1})`. If
      :math:`f({x}_{p}) < f({x}_{n+1})`, replace :math:`x_{n+1}` with
      :math:`x_p`
   -  If all else fails, replace *all* points except :math:`x_1`
      according to

      .. math:: x_i = {x}_{1} + \sigma({x}_{i} - {x}_{1})

The default values of :math:`\alpha, \gamma,\rho` and :math:`\sigma` in
scipy are not listed in the documentation, nor are they inputs to the
function.

Powell's Method
~~~~~~~~~~~~~~~

Powell's method is another derivative-free optimization method that is
similar to conjugate-gradient. The algorithm steps are as follows:

Begin with a point :math:`p_0` (an initial guess) and a set of vectors
:math:`\xi_1,...,\xi_n`, initially the standard basis of
:math:`\mathbb{R}^n`.

-  Compute for :math:`i=1,...,n`, find :math:`\lambda_i` that minimizes
   :math:`f(p_{i-1} +\lambda_i \xi_i)` and set
   :math:`p_i = p_{i-1} + \lambda_i\xi_i`
-  For :math:`i=1,...,n-1`, replace :math:`\xi_{i}` with
   :math:`\xi_{i+1}` and then replace :math:`\xi_n` with
   :math:`p_n - p_0`
-  Choose :math:`\lambda` so that :math:`f(p_0 + \lambda(p_n-p_0)` is
   minimum and replace :math:`p_0` with :math:`p_0 + \lambda(p_n-p_0)`

Essentially, the algorithm performs line searches and tries to find
fruitful directions to search.

Solvers
-------

Levenberg-Marquardt (Damped Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the least squares problem:

Given a set of data points :math:`(x_i, y_i)` where :math:`x_i`'s are
independent variables (in :math:`\mathbb{R}^n` and the :math:`y_i`'s are
response variables (in :math:`\mathbb{R}`), find the parameter values of
:math:`\beta` for the model :math:`f(x;\beta)` so that

.. math:: S(\beta) = \sum\limits_{i=1}^m \left(y_i - f(x_i;\beta)\right)^2

is minimized.

If we were to use Newton's method, our update step would look like:

.. math:: \beta_{k+1} = \beta_k - H^{-1}\nabla S(\beta_k)

Gradient descent, on the other hand, would yield:

.. math:: \beta_{k+1} = \beta_k - \gamma\nabla S(\beta_k)

Levenberg-Marquardt adaptively switches between Newton's method and
gradient descent.

.. math:: \beta_{k+1} = \beta_k - (H + \lambda I)^{-1}\nabla S(\beta_k)

When :math:`\lambda` is small, the update is essentially Newton-Gauss,
while for :math:`\lambda` large, the update is gradient descent.

Newton-Krylov
~~~~~~~~~~~~~

The notion of a Krylov space comes from the Cayley-Hamilton theorem
(CH). CH states that a matrix :math:`A` satisfies its characteristic
polynomial. A direct corollary is that :math:`A^{-1}` may be written as
a linear combination of powers of the matrix (where the highest power is
:math:`n-1`).

The Krylov space of order :math:`r` generated by an :math:`n\times n`
matrix :math:`A` and an :math:`n`-dimensional vector :math:`b` is given
by:

.. math:: \mathcal{K}_r(A,b) = \operatorname{span} \, \{ b, Ab, A^2b, \ldots, A^{r-1}b \}

Thes are actually the subspaces spanned by the conjugate vectors we
mentioned in Newton-CG, so, technically speaking, Newton-CG is a Krylov
method.

Now, the scipy.optimize newton-krylov solver is what is known as a
'Jacobian Free Newton Krylov'. It is a very efficient algorithm for
solving *large* :math:`n\times n` non-linear systems. We won't go into
detail of the algorithm's steps, as this is really more applicable to
problems in physics and non-linear dynamics.

GLM Estimation and IRLS
-----------------------

Recall generalized linear models are models with the following
components:

-  A linear predictor :math:`\eta = X\beta`
-  A response variable with distribution in the exponential family
-  An invertible 'link' function :math:`g` such that

   .. math:: E(Y) = \mu = g^{-1}(\eta)

We may write the log-likelihood:

.. math:: \ell(\eta) = \sum\limits_{i=1}^m (y_i \log(\eta_i) + (\eta_i - y_i)\log(1-\eta_i) 

where :math:`\eta_i = \eta(x_i,\beta)`.

Differentiating, we obtain:

.. math:: \frac{\partial L}{\partial \beta} = \frac{\partial \eta}{\partial \beta}^T\frac{\partial L}{\partial \eta} = 0

Written slightly differently than we have in the previous sections, the
Newton update to find :math:`\beta` would be:

.. math:: -\frac{\partial^2 L}{\partial \beta \beta^T} \left(\beta_{k+1} -\beta_k\right) = \frac{\partial \eta}{\partial \beta}^T\frac{\partial L}{\partial \eta}

Now, if we compute:

.. math:: -\frac{\partial^2 L}{\partial \beta \beta^T} = \sum \frac{\partial L}{\partial \eta_i}\frac{\partial^2 \eta_i}{\partial \beta \beta^T} - \frac{\partial \eta}{\partial \beta}^T \frac{\partial^2 L}{\partial \eta \eta^T}  \frac{\partial \eta}{\partial \beta}

Taking expected values on the right hand side and noting:

.. math:: E\left(\frac{\partial L}{\partial \eta_i} \right) = 0

and

.. math:: E\left(-\frac{\partial^2 L}{\partial \eta \eta^T} \right) = E\left(\frac{\partial L}{\partial \eta}\frac{\partial L}{\partial \eta}^T\right) \equiv A

So if we replace the Hessian in Newton's method with its expected value,
we obtain:

.. math:: \frac{\partial \eta}{\partial \beta}^TA\frac{\partial \eta}{\partial \beta}\left(\beta_{k+1} -\beta_k\right) = \frac{\partial \eta}{\partial \beta}^T\frac{\partial L}{\partial \eta} 

Now, these actually have the form of the normal equations for a weighted
least squares problem.

.. math:: \min_{\beta_{k+1}}\left(A^{-1}\frac{\partial L}{\partial \eta} + \frac{\partial \eta}{\partial \beta}\left(\beta_{k+1} -\beta_k\right)\right)^T A \left(A^{-1}\frac{\partial L}{\partial \eta} + \frac{\partial \eta}{\partial \beta}\left(\beta_{k+1} -\beta_k\right)\right)

:math:`A` is a weight matrix, and changes with iteration - thus this
technique is *iteratively reweighted least squares*.

Constrained Optimization and Lagrange Multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, we want to optimize a function subject to a constraint or
multiple constraints. The most common analytical technique for this is
called 'Lagrange multipliers'. The theory is based on the following:

If we wish to optimize a function :math:`f(x,y)` subject to the
constraint :math:`g(x,y)=c`, we are really looking for points at which
the gradient of :math:`f` and the gradient of :math:`g` are in the same
direction. This amounts to:

.. math:: \nabla_{(x,y)}f = \lambda \nabla_{(x,y)}g

(often, this is written with a (-) sign in front of :math:`\lambda`).
The 2-d problem above defines two equations in three unknowns. The
original constraint, :math:`g(xy,)=c` yields a third equation.
Additional constraints are handled by finding:

.. math:: \nabla_{(x,y)}f = \lambda_1 \nabla_{(x,y)}g_1 + ... + \lambda_k \nabla_{(x,y)}g_k



The generalization to functions on :math:`\mathbb{R}^n` is also trivial:

.. math:: \nabla_{x}f = \lambda \nabla_{x}g

