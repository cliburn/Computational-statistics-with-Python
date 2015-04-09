
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

    np.random.seed(1234)
    import pymc3 as pm
    import scipy.stats as stats

.. code:: python

    import logging
    _logger = logging.getLogger("theano.gof.compilelock")
    _logger.setLevel(logging.ERROR)

Using PyMC3
===========

Install PyMC3 directly from GitHub with

::

    pip install --process-dependency-links git+https://github.com/pymc-devs/pymc3

-  `Repository for PyMC3 <https://github.com/pymc-devs/pymc3>`__
-  `Getting
   started <http://pymc-devs.github.io/pymc3/getting_started/>`__

PyMC3 is alpha software that is intended to improve on PyMC2 in the
following ways (from GitHub page):

-  Intuitive model specification syntax, for example, x ~ N(0,1)
   translates to x = Normal(0,1)
-  Powerful sampling algorithms such as Hamiltonian Monte Carlo
-  Easy optimization for finding the maximum a posteriori point
-  Theano features

   -  Numpy broadcasting and advanced indexing
   -  Linear algebra operators
   -  Computation optimization and dynamic C compilation

-  Simple extensibility

It also comes with extensive
`examples <https://github.com/pymc-devs/pymc3/tree/master/pymc3/examples>`__
including ports of the R/JAGS code examples from `Doing Bayesian Data
Analysis <https://github.com/aloctavodia/Doing_bayesian_data_analysis>`__.

However, the API is different and it is not backwards compartible with
models specified in PyMC2.

Coin toss
---------

We'll repeat the example of determining the bias of a coin from observed
coin tosses. The likelihood is binomial, and we use a beta prior.

Note the different API from PyMC2.

.. code:: python

    n = 100
    h = 61
    alpha = 2
    beta = 2
    
    niter = 1000
    with pm.Model() as model: # context management
        # define priors
        p = pm.Beta('p', alpha=alpha, beta=beta)
        
        # define likelihood
        y = pm.Binomial('y', n=n, p=p, observed=h)
        
        # inference
        start = pm.find_MAP() # Use MAP estimate (optimization) as the initial state for MCMC
        step = pm.Metropolis() # Have a choice of samplers
        trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 0.2 sec

.. code:: python

    plt.hist(trace['p'], 15, histtype='step', normed=True, label='post');
    x = np.linspace(0, 1, 100)
    plt.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
    plt.legend(loc='best');



.. image:: PyMC3_files/PyMC3_6_0.png


Estimating mean and standard deviation of normal distribution
-------------------------------------------------------------

.. math::


   X \sim \mathcal{N}(\mu, \sigma^2)

.. code:: python

    # generate observed data
    N = 100
    _mu = np.array([10])
    _sigma = np.array([2])
    y = np.random.normal(_mu, _sigma, N)
    
    niter = 1000
    with pm.Model() as model:
        # define priors
        mu = pm.Uniform('mu', lower=0, upper=100, shape=_mu.shape)
        sigma = pm.Uniform('sigma', lower=0, upper=10, shape=_sigma.shape)
        
        # define likelihood
        y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)
        
        # inference
        start = pm.find_MAP()
        step = pm.Slice()
        trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 1.9 sec

.. code:: python

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); 
    plt.hist(trace['mu'][-niter/2:,0], 25, histtype='step');
    plt.subplot(1,2,2); 
    plt.hist(trace['sigma'][-niter/2:,0], 25, histtype='step');



.. image:: PyMC3_files/PyMC3_9_0.png


Estimating parameters of a linear regreession model
---------------------------------------------------

We will show how to estimate regression parameters using a simple linear
modesl

.. math::


   y \sim ax + b

We can restate the linear model

.. math:: y = ax + b + \epsilon

as sampling from a probability distribution

.. math::


   y \sim \mathcal{N}(ax + b, \sigma^2)

Now we can use pymc to estimate the paramters :math:`a`, :math:`b` and
:math:`\sigma` (pymc2 uses precision :math:`\tau` which is
:math:`1/\sigma^2` so we need to do a simple transformation). We will
assume the following priors

.. math::


   a \sim \mathcal{N}(0, 100) \\
   b \sim \mathcal{N}(0, 100) \\
   \tau \sim \text{Gamma}(0.1, 0.1)

.. code:: python

    # observed data
    n = 11
    _a = 6
    _b = 2
    x = np.linspace(0, 1, n)
    y = _a*x + _b + np.random.randn(n)
    
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sd=20)
        b = pm.Normal('b', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=20)
        
        y_est = a*x + b # simple auxiliary variables
        
        likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
        # inference
        start = pm.find_MAP()
        step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
        trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)
        pm.traceplot(trace);


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 8.9 sec


.. image:: PyMC3_files/PyMC3_11_1.png


Alternative fromulation using GLM formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    data = dict(x=x, y=y)
    
    with pm.Model() as model:
        pm.glm.glm('y ~ x', data)
        step = pm.NUTS() 
        trace = pm.sample(2000, step, progressbar=True) 


.. parsed-literal::

     [-----------------100%-----------------] 2000 of 2000 complete in 8.1 sec

.. code:: python

    pm.traceplot(trace);



.. image:: PyMC3_files/PyMC3_14_0.png


.. code:: python

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=30, label='data')
    pm.glm.plot_posterior_predictive(trace, samples=100, 
                                     label='posterior predictive regression lines', 
                                     c='blue', alpha=0.2)
    plt.plot(x, _a*x + _b, label='true regression line', lw=3., c='red')
    plt.legend(loc='best');



.. image:: PyMC3_files/PyMC3_15_0.png


Simple Logistic model
~~~~~~~~~~~~~~~~~~~~~

We have observations of height and weight and want to use a logistic
model to guess the sex.

.. code:: python

    # observed data
    df = pd.read_csv('HtWt.csv')
    df.head()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>male</th>
          <th>height</th>
          <th>weight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 0</td>
          <td> 63.2</td>
          <td> 168.7</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 0</td>
          <td> 68.7</td>
          <td> 169.8</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 0</td>
          <td> 64.8</td>
          <td> 176.6</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 0</td>
          <td> 67.9</td>
          <td> 246.8</td>
        </tr>
        <tr>
          <th>4</th>
          <td> 1</td>
          <td> 68.9</td>
          <td> 151.6</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    niter = 1000
    with pm.Model() as model:
        pm.glm.glm('male ~ height + weight', df, family=pm.glm.families.Binomial()) 
        trace = pm.sample(niter, step=pm.Slice(), random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 3.2 sec

.. code:: python

    # note that height and weigth in trace refer to the coefficients
    
    df_trace = pm.trace_to_dataframe(trace)
    pd.scatter_matrix(df_trace[-1000:], diagonal='kde');



.. image:: PyMC3_files/PyMC3_19_0.png


.. code:: python

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(df_trace.ix[-1000:, 'height'], linewidth=0.7)
    plt.subplot(122)
    plt.plot(df_trace.ix[-1000:, 'weight'], linewidth=0.7);



.. image:: PyMC3_files/PyMC3_20_0.png


There is no convergence!
~~~~~~~~~~~~~~~~~~~~~~~~

Becaue of ths strong correlation between height and weight, a
one-at-a-time sampler such as the slice or Gibbs sampler will take a
long time to converge. The HMC does much better.

.. code:: python

    niter = 1000
    with pm.Model() as model:
        pm.glm.glm('male ~ height + weight', df, family=pm.glm.families.Binomial()) 
        trace = pm.sample(niter, step=pm.NUTS(), random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 1001 of 1000 complete in 27.0 sec

.. code:: python

    df_trace = pm.trace_to_dataframe(trace)
    pd.scatter_matrix(df_trace[-1000:], diagonal='kde');



.. image:: PyMC3_files/PyMC3_23_0.png


.. code:: python

    pm.summary(trace);


.. parsed-literal::

    
    Intercept:
     
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      -51.393          11.299           0.828            [-73.102, -29.353]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      -76.964        -58.534        -50.383        -43.856        -30.630
    
    
    height:
     
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.747            0.170            0.012            [0.422, 1.096]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      0.453          0.630          0.732          0.853          1.139
    
    
    weight:
     
      Mean             SD               MC Error         95% HPD interval
      -------------------------------------------------------------------
      
      0.011            0.012            0.001            [-0.012, 0.034]
    
      Posterior quantiles:
      2.5            25             50             75             97.5
      |--------------|==============|==============|--------------|
      
      -0.012         0.002          0.010          0.019          0.034
    


.. code:: python

    import seaborn as sn
    sn.kdeplot(trace['weight'], trace['height'])
    plt.xlabel('Weight', fontsize=20)
    plt.ylabel('Height', fontsize=20)
    plt.style.use('ggplot')



.. image:: PyMC3_files/PyMC3_25_0.png


We can use the logistic regression results to classify subjects as male
or female based on their height and weight, using 0.5 as a cutoff, as
shown in the plot below. Green = true positive male, yellow = true
positive female, red halo = misclassification. Blue line shows the 0.5
cutoff.

.. code:: python

    intercept, height, weight = df_trace[-niter//2:].mean(0)
    
    def predict(w, h, height=height, weight=weight):
        """Predict gender given weight (w) and height (h) values."""
        v = intercept + height*h + weight*w
        return np.exp(v)/(1+np.exp(v))
    
    # calculate predictions on grid
    xs = np.linspace(df.weight.min(), df.weight.max(), 100)
    ys = np.linspace(df.height.min(), df.height.max(), 100)
    X, Y = np.meshgrid(xs, ys)
    Z = predict(X, Y)
    
    plt.figure(figsize=(6,6))
    
    # plot 0.5 contour line - classify as male if above this line
    plt.contour(X, Y, Z, levels=[0.5])
    
    # classify all subjects
    colors = ['lime' if i else 'yellow' for i in df.male]
    ps = predict(df.weight, df.height)
    errs = ((ps < 0.5) & df.male) |((ps >= 0.5) & (1-df.male))
    plt.scatter(df.weight[errs], df.height[errs], facecolors='red', s=150)
    plt.scatter(df.weight, df.height, facecolors=colors, edgecolors='k', s=50, alpha=1);
    plt.xlabel('Weight', fontsize=16)
    plt.ylabel('Height', fontsize=16)
    plt.title('Gender classification by weight and height', fontsize=16)
    plt.tight_layout();



.. image:: PyMC3_files/PyMC3_27_0.png


Estimating parameters of a logistic model
-----------------------------------------

Gelman's book has an example where the dose of a drug may be affected to
the number of rat deaths in an experiment.

+-------------------+----------+------------+
| Dose (log g/ml)   | # Rats   | # Deaths   |
+===================+==========+============+
| -0.896            | 5        | 0          |
+-------------------+----------+------------+
| -0.296            | 5        | 1          |
+-------------------+----------+------------+
| -0.053            | 5        | 3          |
+-------------------+----------+------------+
| 0.727             | 5        | 5          |
+-------------------+----------+------------+

We will model the number of deaths as a random sample from a binomial
distribution, where :math:`n` is the number of rats and :math:`p` the
probabbility of a rat dying. We are given :math:`n = 5`, but we believve
that :math:`p` may be related to the drug dose :math:`x`. As :math:`x`
increases the number of rats dying seems to increase, and since
:math:`p` is a probability, we use the following model:

.. math::


   y \sim \text{Bin}(n, p) \\
   \text{logit}(p) = \alpha + \beta x \\
   \alpha \sim \mathcal{N}(0, 5) \\
   \beta \sim \mathcal{N}(0, 10)

where we set vague priors for :math:`\alpha` and :math:`\beta`, the
parameters for the logistic model.

.. code:: python

    # observed data
    n = 5 * np.ones(4)
    x = np.array([-0.896, -0.296, -0.053, 0.727])
    y = np.array([0, 1, 3, 5])
    
    def invlogit(x):
        return pm.exp(x) / (1 + pm.exp(x))
    
    with pm.Model() as model:
        # define priors
        alpha = pm.Normal('alpha', mu=0, sd=5)
        beta = pm.Flat('beta')
        
        # define likelihood
        p = invlogit(alpha + beta*x)
        y_obs = pm.Binomial('y_obs', n=n, p=p, observed=y)
        
        # inference
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(niter, step, start, random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 1000 of 1000 complete in 2.5 sec

.. code:: python

    np.exp




.. parsed-literal::

    <ufunc 'exp'>



.. code:: python

    f = lambda a, b, xp: np.exp(a + b*xp)/(1 + np.exp(a + b*xp))
    
    xp = np.linspace(-1, 1, 100)
    a = trace.get_values('alpha').mean()
    b = trace.get_values('beta').mean()
    plt.plot(xp, f(a, b, xp))
    plt.scatter(x, y/5, s=50);
    plt.xlabel('Log does of drug')
    plt.ylabel('Risk of death');



.. image:: PyMC3_files/PyMC3_31_0.png


Using a hierarchcical model
---------------------------

This uses the Gelman radon data set and is based off this `IPython
notebook <http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/>`__.
Radon levels were measured in houses from all counties in several
states. Here we want to know if the preence of a basement affects the
level of radon, and if this is affected by which county the house is
located in.

The data set provided is just for the state of Minnesota, which has 85
counties with 2 to 116 measurements per county. We only need 3 columns
for this example ``county``, ``log_radon``, ``floor``, where ``floor=0``
indicates that there is a basement.

We will perfrom simple linear regression on log\_radon as a function of
county and floor.

.. code:: python

    radon = pd.read_csv('radon.csv')[['county', 'floor', 'log_radon']]
    radon.head()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>county</th>
          <th>floor</th>
          <th>log_radon</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> AITKIN</td>
          <td> 1</td>
          <td> 0.832909</td>
        </tr>
        <tr>
          <th>1</th>
          <td> AITKIN</td>
          <td> 0</td>
          <td> 0.832909</td>
        </tr>
        <tr>
          <th>2</th>
          <td> AITKIN</td>
          <td> 0</td>
          <td> 1.098612</td>
        </tr>
        <tr>
          <th>3</th>
          <td> AITKIN</td>
          <td> 0</td>
          <td> 0.095310</td>
        </tr>
        <tr>
          <th>4</th>
          <td>  ANOKA</td>
          <td> 0</td>
          <td> 1.163151</td>
        </tr>
      </tbody>
    </table>
    </div>



Hiearchical model
^^^^^^^^^^^^^^^^^

With a hierarchical model, there is an :math:`a_c` and a :math:`b_c` for
each county :math:`c` just as in the individual couty model, but they
are no longer indepnedent but assumed to come from a common group
distribution

.. math::


   a_c \sim \mathcal{N}(\mu_a, \sigma_a^2) \\
   b_c \sim \mathcal{N}(\mu_b, \sigma_b^2)

we furhter assume that the hyperparameters come from the following
distributions

.. math::


   \mu_a \sim \mathcal{N}(0, 100^2) \\
   \sigma_a \sim \mathcal{U}(0, 100) \\ 
   \mu_b \sim \mathcal{N}(0, 100^2) \\
   \sigma_b \sim \mathcal{U}(0, 100)

.. code:: python

    county = pd.Categorical(radon['county']).codes
    
    with pm.Model() as hm:
        # County hyperpriors
        mu_a = pm.Normal('mu_a', mu=0, tau=1.0/100**2)
        sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)
        mu_b = pm.Normal('mu_b', mu=0, tau=1.0/100**2)
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=100)
        
        # County slopes and intercepts
        a = pm.Normal('slope', mu=mu_a, sd=sigma_a, shape=len(set(county)))
        b = pm.Normal('intercept', mu=mu_b, tau=1.0/sigma_b**2, shape=len(set(county)))
        
        # Houseehold errors
        sigma = pm.Gamma("sigma", alpha=10, beta=1)
        
        # Model prediction of radon level
        mu = a[county] + b[county] * radon.floor.values
        
        # Data likelihood
        y = pm.Normal('y', mu=mu, sd=sigma, observed=radon.log_radon)

.. code:: python

    with hm:
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        hm_trace = pm.sample(2000, step, start=start, random_seed=123, progressbar=True)


.. parsed-literal::

     [-----------------100%-----------------] 2001 of 2000 complete in 1295.7 sec

.. code:: python

    plt.figure(figsize=(8, 60))
    pm.forestplot(hm_trace, vars=['slope', 'intercept']);




.. parsed-literal::

    <matplotlib.gridspec.GridSpec at 0x15d4808d0>




.. image:: PyMC3_files/PyMC3_37_1.png


