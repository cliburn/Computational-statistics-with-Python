
.. code:: python

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy as scipy

Optimization and Non-linear Methods
===================================

It is sometimes necessary to solve equations or systems of equations
that are non-linear. Often, those non-linear equations arise as
optimization problems.

Example: Maximum Likelihood Estimation (MLE)
--------------------------------------------

Recall that in MLE, we are interested in estimating the value of a
parameter :math:`\theta` that maximizes a log-likelihood function
:math:`\ell(X;\theta)`. Let :math:`X_1,...,X_n` be an iid set of random
variables with pdf :math:`f(x;\theta)`, where
:math:`\theta \in \mathbb{R}^k` is a parameter. The likelihood function
is:

.. math:: L(X;\theta) = \prod_{i=1}^n f(X_i;\theta)

We want the value of :math:`\theta` that maximizes :math:`L`. We can
accomplish this by taking the first derivative (or gradient) of
:math:`L` with respect to :math:`\theta`, setting it to zero and solving
for :math:`\theta`. However, this is more easily accomplished if we
first take :math:`\log(L)`, as :math:`L` is a product of densities, and
taking the log of a product yields a sum. Because :math:`log` is a
monotonically increasing function, any value of :math:`\theta` that
maximizes :math:`\log(L)` also maximizes :math:`L`.

.. raw:: latex

   \begin{eqnarray}
   \ell(X;\theta) &=& \log(L(X;\theta)) \\\\
   &=& \log\left(\prod_{i=1}^n f(X_i;\theta)\right)\\\\
   &=&\sum_{i=1}^n \log(f(X_i;\theta)
   \end{eqnarray}

Optimization then amounts to finding the zeros of

.. raw:: latex

   \begin{eqnarray}
   \frac{\partial\ell}{\partial \theta} &=& \frac{\partial}{\partial \theta} \left(\sum_{i=1}^n\log(f(X_i;\theta)\right)\\\\
   &=& \sum_{i=1}^n \frac{\partial\log(f(X_i;\theta)}{\partial \theta}
   \end{eqnarray}

Bisection Method
----------------

The bisection method is one of the simplest methods for finding zeroes
of a non-linear function. It is guaranteed to find a root - but it can
be slow. The main idea comes from the intermediate value theorem: If
:math:`f(a)` and :math:`f(b)` have different signs and :math:`f` is
continous, then :math:`f` must have a zero between :math:`a` and
:math:`b`. We evaluate the function at the midpoint,
:math:`c = \frac12(a+b)`. :math:`f(c)` is either zero, has the same sign
as :math:`f(a)` or the same sign as :math:`f(b)`. Suppose :math:`f(c)`
has the same sign as :math:`f(a)` (as pictured below). We then repeat
the process on the interval :math:`[c,b]`.

.. code:: python

    def f(x):
        return x**3 + 4*x**2 -3
    
    x = np.linspace(-3.1, 0, 100)
    plt.plot(x, x**3 + 4*x**2 -3)
    
    a = -3.0
    b = -0.5
    c = 0.5*(a+b)
    
    plt.text(a,-1,"a")
    plt.text(b,-1,"b")
    plt.text(c,-1,"c")
    
    plt.scatter([a,b,c], [f(a), f(b),f(c)], s=50, facecolors='none')
    plt.scatter([a,b,c], [0,0,0], s=50, c='red')
    
    xaxis = plt.axhline(0);



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_11_0.png


.. code:: python

    
    x = np.linspace(-3.1, 0, 100)
    plt.plot(x, x**3 + 4*x**2 -3)
    
    d = 0.5*(b+c)
    
    plt.text(d,-1,"d")
    plt.text(b,-1,"b")
    plt.text(c,-1,"c")
    
    plt.scatter([d,b,c], [f(d), f(b),f(c)], s=50, facecolors='none')
    plt.scatter([d,b,c], [0,0,0], s=50, c='red')
    
    xaxis = plt.axhline(0);



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_12_0.png


We can terminate the process whenever the function evaluated at the new
midpoint is 'close enough' to zero.

Secant Method
-------------

The secant method also begins with two initial points, but without the
constraint that the function values are of opposite signs. We use the
secant line to extrapolate the next candidate point.

.. code:: python

    def f(x):
        return (x**3-2*x+7)/(x**4+2)
    
    x = np.arange(-3,5, 0.1);
    y = f(x)
    
    p1=plt.plot(x, y)
    plt.xlim(-3, 4)
    plt.ylim(-.5, 4)
    plt.xlabel('x')
    plt.axhline(0)
    t = np.arange(-10, 5., 0.1)
    
    x0=-1.2
    x1=-0.5
    xvals = []
    xvals.append(x0)
    xvals.append(x1)
    notconverge = 1
    count = 0
    cols=['r--','b--','g--','y--']
    while (notconverge==1 and count <  3):
        slope=(f(xvals[count+1])-f(xvals[count]))/(xvals[count+1]-xvals[count])
        intercept=-slope*xvals[count+1]+f(xvals[count+1])
        plt.plot(t, slope*t + intercept, cols[count])
        nextval = -intercept/slope
        if abs(f(nextval)) < 0.001:
            notconverge=0
        else:
            xvals.append(nextval)
        count = count+1
    
    plt.show()



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_16_0.png


The secant method has the advantage of fast convergence. While the
bisection method has a linear convergence rate (i.e. error goes to zero
at the rate that :math:`h(x) = x` goes to zero, the secant method has a
convergence rate that is faster than linear, but not quite quadratic
(i.e. :math:`\sim x^\alpha`, where
:math:`\alpha = \frac{1+\sqrt{5}}2 \approx 1.6`)

Newton-Rhapson Method
---------------------


We want to find the value :math:`\theta` so that some (differentiable)
function :math:`g(\theta)=0`. Idea: start with a guess,
:math:`\theta_0`. Let :math:`\tilde{\theta}` denote the value of
:math:`\theta` for which :math:`g(\theta) = 0` and define
:math:`h = \tilde{\theta} - \theta_0`. Then:

.. raw:: latex

   \begin{eqnarray}
   g(\tilde{\theta}) &=& 0 \\\\
   &=&g(\theta_0 + h) \\\\
   &\approx& g(\theta_0) + hg'(\theta_0)
   \end{eqnarray}

This implies that

.. math::  h\approx \frac{g(\theta_0)}{g'(\theta_0)}

So that

.. math:: \tilde{\theta}\approx \theta_0 - \frac{g(\theta_0)}{g'(\theta_0)}

Thus, we set our next approximation:

.. math:: \theta_1 = \theta_0 - \frac{g(\theta_0)}{g'(\theta_0)}

and we have developed an interative procedure with:

.. math:: \theta_n = \theta_{n-1} - \frac{g(\theta_{n-1})}{g'(\theta_{n-1})}

Example:
^^^^^^^^

Let

.. math:: g(x) = \frac{x^3-2x+7}{x^4+2}

The graph of this function is:

.. code:: python

    x = np.arange(-5,5, 0.1);
    y = (x**3-2*x+7)/(x**4+2)
    
    p1=plt.plot(x, y)
    plt.xlim(-4, 4)
    plt.ylim(-.5, 4)
    plt.xlabel('x')
    plt.axhline(0)
    plt.title('Example Function')
    plt.show()



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_24_0.png


.. code:: python

    
    x = np.arange(-5,5, 0.1);
    y = (x**3-2*x+7)/(x**4+2)
    
    p1=plt.plot(x, y)
    plt.xlim(-4, 4)
    plt.ylim(-.5, 4)
    plt.xlabel('x')
    plt.axhline(0)
    plt.title('Good Guess')
    t = np.arange(-5, 5., 0.1)
    
    x0=-1.5
    xvals = []
    xvals.append(x0)
    notconverge = 1
    count = 0
    cols=['r--','b--','g--','y--','c--','m--','k--','w--']
    while (notconverge==1 and count <  6):
        funval=(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
        slope=-((4*xvals[count]**3 *(7 - 2 *xvals[count] + xvals[count]**3))/(2 + xvals[count]**4)**2) + (-2 + 3 *xvals[count]**2)/(2 + xvals[count]**4)
       
        intercept=-slope*xvals[count]+(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    
        plt.plot(t, slope*t + intercept, cols[count])
        nextval = -intercept/slope
        if abs(funval) < 0.01:
            notconverge=0
        else:
            xvals.append(nextval)
        count = count+1
    
    plt.show()
    




.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_25_0.png


From the graph, we see the zero is near -2. We make an initial guess of

.. math:: x=-1.5

We have made an excellent choice for our first guess, and we can see
rapid convergence!

.. code:: python

    funval




.. parsed-literal::

    0.007591996330867034



In fact, the Newton-Rhapson method converges quadratically. However, NR
(and the secant method) have a fatal flaw:

.. code:: python

    
    x = np.arange(-5,5, 0.1);
    y = (x**3-2*x+7)/(x**4+2)
    
    p1=plt.plot(x, y)
    plt.xlim(-4, 4)
    plt.ylim(-.5, 4)
    plt.xlabel('x')
    plt.axhline(0)
    plt.title('Bad Guess')
    t = np.arange(-5, 5., 0.1)
    
    x0=-0.5
    xvals = []
    xvals.append(x0)
    notconverge = 1
    count = 0
    cols=['r--','b--','g--','y--','c--','m--','k--','w--']
    while (notconverge==1 and count <  6):
        funval=(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
        slope=-((4*xvals[count]**3 *(7 - 2 *xvals[count] + xvals[count]**3))/(2 + xvals[count]**4)**2) + (-2 + 3 *xvals[count]**2)/(2 + xvals[count]**4)
       
        intercept=-slope*xvals[count]+(xvals[count]**3-2*xvals[count]+7)/(xvals[count]**4+2)
    
        plt.plot(t, slope*t + intercept, cols[count])
        nextval = -intercept/slope
        if abs(funval) < 0.01:
            notconverge = 0
        else:
            xvals.append(nextval)
        count = count+1
    
    plt.show()




.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_30_0.png


We have stumbled on the horizontal asymptote. The algorithm fails to
converge.

Basins of Attraction Can Be 'Close'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def f(x):
        return x**3 - 2*x**2 - 11*x +12
    def s(x):
        return 3*x**2 - 4*x - 11
    
    x = np.arange(-5,5, 0.1);
    
    p1=plt.plot(x, f(x))
    plt.xlim(-4, 5)
    plt.ylim(-20, 22)
    plt.xlabel('x')
    plt.axhline(0)
    plt.title('Basin of Attraction')
    t = np.arange(-5, 5., 0.1)
    
    x0=2.43
    xvals = []
    xvals.append(x0)
    notconverge = 1
    count = 0
    cols=['r--','b--','g--','y--','c--','m--','k--','w--']
    while (notconverge==1 and count <  6):
        funval = f(xvals[count])
        slope = s(xvals[count])
       
        intercept=-slope*xvals[count]+funval
    
        plt.plot(t, slope*t + intercept, cols[count])
        nextval = -intercept/slope
        if abs(funval) < 0.01:
            notconverge = 0
        else:
            xvals.append(nextval)
        count = count+1
    
    plt.show()
    xvals[count-1]



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_33_0.png




.. parsed-literal::

    -3.1713324128480282



.. code:: python

    p1=plt.plot(x, f(x))
    plt.xlim(-4, 5)
    plt.ylim(-20, 22)
    plt.xlabel('x')
    plt.axhline(0)
    plt.title('Basin of Attraction')
    t = np.arange(-5, 5., 0.1)
    
    x0=2.349
    xvals = []
    xvals.append(x0)
    notconverge = 1
    count = 0
    cols=['r--','b--','g--','y--','c--','m--','k--','w--']
    while (notconverge==1 and count <  6):
        funval = f(xvals[count])
        slope = s(xvals[count])
       
        intercept=-slope*xvals[count]+funval
    
        plt.plot(t, slope*t + intercept, cols[count])
        nextval = -intercept/slope
        if abs(funval) < 0.01:
            notconverge = 0
        else:
            xvals.append(nextval)
        count = count+1
    
    plt.show()
    xvals[count-1]



.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_34_0.png




.. parsed-literal::

    0.9991912395651094



Convergence Rate
~~~~~~~~~~~~~~~~

The following is a derivation of the convergence rate of the NR method:

Suppose :math:`x_k \; \rightarrow \; x^*` and :math:`g'(x^*) \neq 0`.
Then we may write:

.. math:: x_k = x^* + \epsilon_k

.

Now expand :math:`g` at :math:`x^*`:

.. math:: g(x_k) = g(x^*) + g'(x^*)\epsilon_k + \frac12 g''(x^*)\epsilon_k^2 + ...

.. math:: g'(x_k)=g'(x^*) + g''(x^*)\epsilon_k

We have that

.. raw:: latex

   \begin{eqnarray}
   \epsilon_{k+1} &=& \epsilon_k + \left(x_{k-1}-x_k\right)\\
   &=& \epsilon_k -\frac{g(x_k)}{g'(x_k)}\\
   &\approx & \frac{g'(x^*)\epsilon_k + \frac12g''(x^*)\epsilon_k^2}{g'(x^*)+g''(x^*)\epsilon_k}\\
   &\approx & \frac{g''(x^*)}{2g'(x^*)}\epsilon_k^2
   \end{eqnarray}

Gauss-Newton
------------

For 1D, the Newton method is

.. math::


   x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}

We can generalize to :math:`k` dimensions by

.. math::


   x_{n+1} = x_n - J^{-1} f(x_n)

where :math:`x` and :math:`f(x)` are now vectors, and :math:`J^{-1}` is
the inverse Jacobian matrix. In general, the Jacobian is not a square
matrix, and we use the generalized inverse :math:`(J^TJ)^{-1}J^T`
instead, giving

.. math::


   x_{n+1} = x_n - (J^TJ)^{-1}J^T f(x_n)

In multivariate nonlinear estimation problems, we can find the vector of
parameters :math:`\beta` by minimizing the residuals :math:`r(\beta)`,

.. math::


   \beta_{n+1} = \beta_n - (J^TJ)^{-1}J^T r(\beta_n)

where the entries of the Jacobian matrix :math:`J` are

.. math::


   J_{ij} = \frac{\partial r_i(\beta)}{\partial \beta_j}

Inverse Quadratic Interpolation
-------------------------------

Inverse quadratic interpolation is a type of polynomial interpolation.
Polynomial interpolation simply means we find the polynomial of least
degree that fits a set of points. In quadratic interpolation, we use
three points, and find the quadratic polynomial that passes through
those three points.

Inverse quadratic interpolation means we do quadratic interpolation on
the inverse function. So, if we are looking for a root of :math:`f`, we
approximate :math:`f^{-1}(x)` using quadratic interpolation. Note that
the secant method can be viewed as a *linear* interpolation on the
inverse of :math:`f`. We can write:

.. math:: f^{-1}(y) = \frac{(y-f(x_n))(y-f(x_{n-1}))}{(f(x_{n-2})-f(x_{n-1}))(f(x_{n-2})-f(x_{n}))}x_{n-2} + \frac{(y-f(x_n))(y-f(x_{n-2}))}{(f(x_{n-1})-f(x_{n-2}))(f(x_{n-1})-f(x_{n}))}x_{n-1} + \frac{(y-f(x_{n-2}))(y-f(x_{n-1}))}{(f(x_{n})-f(x_{n-2}))(f(x_{n})-f(x_{n-1}))}x_{n-1}

We use the above formula to find the next guess :math:`x_{n+1}` for a
zero of :math:`f` (so :math:`y=0`):

.. math:: x_{n+1} = \frac{f(x_n)f(x_{n-1})}{(f(x_{n-2})-f(x_{n-1}))(f(x_{n-2})-f(x_{n}))}x_{n-2} + \frac{f(x_n)f(x_{n-2})}{(f(x_{n-1})-f(x_{n-2}))(f(x_{n-1})-f(x_{n}))}x_{n-1} + \frac{f(x_{n-2})f(x_{n-1})}{(f(x_{n})-f(x_{n-2}))(f(x_{n})-f(x_{n-1}))}x_{n}

Convergence rate is approximately :math:`1.8`.

Brent's Method
--------------

Brent's method is a combination of bisection, secant and inverse
quadratic interpolation. Like bisection, it is a 'bracketed' method
(starts with points :math:`(a,b)` such that :math:`f(a)f(b)<0`.

Roughly speaking, the method begins by using the secant method to obtain
a third point :math:`c`, then uses inverse quadratic interpolation to
generate the next possible root. Without going into too much detail, the
algorithm attempts to assess when interpolation will go awry, and if so,
performs a bisection step. Also, it has certain criteria to reject an
iterate. If that happens, the next step will be linear interpolation
(secant method).

The Brent method is the default method that scypy uses to minimize a
univariate function:

.. code:: python

    from scipy.optimize import minimize_scalar
    
    def f(x):
        return (x - 2) * x * (x + 2)**2
    
    res = minimize_scalar(f)
    res.x




.. parsed-literal::

    1.2807764040333458



.. code:: python

    x = np.arange(-5,5, 0.1);
    p1=plt.plot(x, f(x))
    plt.xlim(-4, 4)
    plt.ylim(-10, 20)
    plt.xlabel('x')
    plt.axhline(0)




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7f9c9b232bd0>




.. image:: OptimizationInOneDimension_files/OptimizationInOneDimension_46_1.png


To find zeroes, use

.. code:: python

    scipy.optimize.brentq(f,-1,.5)




.. parsed-literal::

    -7.864845203343107e-19



.. code:: python

    scipy.optimize.brentq(f,.5,3)




.. parsed-literal::

    2.0



.. code:: python

    scipy.optimize.newton(f,-3)




.. parsed-literal::

    -2.0000000172499592



