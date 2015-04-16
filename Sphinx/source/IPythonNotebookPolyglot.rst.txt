
Getting started with Python and the IPython notebook
====================================================

To make R users feel at ease, you can always use R from within the
IPython notebook if you install the rpy2 package

.. code:: bash

    pip install rpy2

.. code:: python

    %load_ext rpy2.ipython 


.. parsed-literal::

    The rpy2.ipython extension is already loaded. To reload it, use:
      %reload_ext rpy2.ipython


.. code:: python

    %matplotlib inline

.. code:: python

    %%R
    library(lattice) 
    attach(mtcars)
    
    # scatterplot matrix 
    splom(mtcars[c(1,3,4,5,6)], main="MTCARS Data")



.. parsed-literal::

    The following objects are masked from mtcars (pos = 3):
    
        am, carb, cyl, disp, drat, gear, hp, mpg, qsec, vs, wt
    




.. image:: IPythonNotebookPolyglot_files/IPythonNotebookPolyglot_4_1.png


Matlab users are also covered with

.. code:: bash

    pip install pymatbridge

.. code:: python

    import pymatbridge as pymat
    ip = get_ipython()
    pymat.load_ipython_extension(ip)

.. code:: python

    %%matlab
    
    xgv = -1.5:0.1:1.5;
    ygv = -3:0.1:3;
    [X,Y] = ndgrid(xgv,ygv);
    V = exp(-(X.^2 + Y.^2));
    surf(X,Y,V)
    title('Gridded Data Set', 'fontweight','b');


::


    ---------------------------------------------------------------------------
    RuntimeError                              Traceback (most recent call last)

    <ipython-input-14-8ef3de53fe4f> in <module>()
    ----> 1 get_ipython().run_cell_magic(u'matlab', u'', u"\nxgv = -1.5:0.1:1.5;\nygv = -3:0.1:3;\n[X,Y] = ndgrid(xgv,ygv);\nV = exp(-(X.^2 + Y.^2));\nsurf(X,Y,V)\ntitle('Gridded Data Set', 'fontweight','b');")
    

    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/core/interactiveshell.pyc in run_cell_magic(self, magic_name, line, cell)
       2160             magic_arg_s = self.var_expand(line, stack_depth)
       2161             with self.builtin_trap:
    -> 2162                 result = fn(magic_arg_s, cell)
       2163             return result
       2164 


    /home/bitnami/anaconda/lib/python2.7/site-packages/pymatbridge/matlab_magic.pyc in matlab(self, line, cell, local_ns)


    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/core/magic.pyc in <lambda>(f, *a, **k)
        191     # but it's overkill for just that one bit of state.
        192     def magic_deco(arg):
    --> 193         call = lambda f, *a, **k: f(*a, **k)
        194 
        195         if callable(arg):


    /home/bitnami/anaconda/lib/python2.7/site-packages/pymatbridge/matlab_magic.pyc in matlab(self, line, cell, local_ns)
        215             e_s += "\n-----------------------"
        216             e_s += "\nAre you sure Matlab is started?"
    --> 217             raise RuntimeError(e_s)
        218 
        219 


    RuntimeError: There was an error running the code:
     
    xgv = -1.5:0.1:1.5;
    ygv = -3:0.1:3;
    [X,Y] = ndgrid(xgv,ygv);
    V = exp(-(X.^2 + Y.^2));
    surf(X,Y,V)
    title('Gridded Data Set', 'fontweight','b');
    -----------------------
    Are you sure Matlab is started?


And it is also OK if you prefer Octave. Just type

.. code:: bash

    pip install oct2py

.. code:: python

    %load_ext octavemagic

.. code:: python

    %%octave
    
    A = reshape(1:4,2,2)'; 
    b = [36; 88];
    A\b
    [L,U,P] = lu(A)
    [Q,R] = qr(A)
    [V,D] = eig(A)


::


    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)

    <ipython-input-16-290bbde86e1b> in <module>()
    ----> 1 get_ipython().run_cell_magic(u'octave', u'', u"\nA = reshape(1:4,2,2)'; \nb = [36; 88];\nA\\b\n[L,U,P] = lu(A)\n[Q,R] = qr(A)\n[V,D] = eig(A)")
    

    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/core/interactiveshell.pyc in run_cell_magic(self, magic_name, line, cell)
       2160             magic_arg_s = self.var_expand(line, stack_depth)
       2161             with self.builtin_trap:
    -> 2162                 result = fn(magic_arg_s, cell)
       2163             return result
       2164 


    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/extensions/octavemagic.pyc in octave(self, line, cell, local_ns)


    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/core/magic.pyc in <lambda>(f, *a, **k)
        191     # but it's overkill for just that one bit of state.
        192     def magic_deco(arg):
    --> 193         call = lambda f, *a, **k: f(*a, **k)
        194 
        195         if callable(arg):


    /home/bitnami/anaconda/lib/python2.7/site-packages/IPython/extensions/octavemagic.pyc in octave(self, line, cell, local_ns)
        327         except (oct2py.Oct2PyError) as exception:
        328             msg = exception.message
    --> 329             msg = msg.split('# ___<end_pre_call>___ #')[1]
        330             msg = msg.split('# ___<start_post_call>___ #')[0]
        331             raise OctaveMagicError('Octave could not complete execution.  '


    IndexError: list index out of range


We will redo these examples in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import pandas as pd
    import numpy as np
    import statsmodels.api as sm 
    from pandas.tools.plotting import scatter_matrix

.. code:: python

    # First we will load the mtcars dataset and do a scatterplot matrix
    
    mtcars = sm.datasets.get_rdataset('mtcars')
    df = pd.DataFrame(mtcars.data)
    scatter_matrix(df[[0,2,3,4,5]], alpha=0.3, figsize=(8, 8), diagonal='kde', marker='o');

.. code:: python

    # Next we will do the 3D mesh
    
    xgv = np.arange(-1.5, 1.5, 0.1)
    ygv = np.arange(-3, 3, 0.1)
    [X,Y] = np.meshgrid(xgv, ygv)
    V = np.exp(-(X**2 + Y**2))
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0.25)
    plt.title('Gridded Data Set');

.. code:: python

    # And finally, the matrix manipulations
    
    import scipy
    
    A = np.reshape(np.arange(1, 5), (2,2))
    b = np.array([36, 88])
    ans = scipy.linalg.solve(A, b)
    P, L, U = scipy.linalg.lu(A)
    Q, R = scipy.linalg.qr(A)
    D, V = scipy.linalg.eig(A)
    print 'ans =\n', ans, '\n'
    print 'L =\n', L, '\n'
    print "U =\n", U, '\n'
    print "P = \nPermutation Matrix\n", P, '\n'
    print 'Q =\n', Q, '\n'
    print "R =\n", R, '\n'
    print 'V =\n', V, '\n'
    print "D =\nDiagonal matrix\n", np.diag(abs(D)), '\n'

Julia
-----

