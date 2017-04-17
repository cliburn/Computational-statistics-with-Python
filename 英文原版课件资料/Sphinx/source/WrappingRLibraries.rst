
Wrapping R libraries with Rpy
=============================

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

    from IPython.core.display import Image
    import uuid 

.. code:: python

    import rpy2.robjects as robjects

.. code:: python

    from rpy2.robjects.packages import importr

.. code:: python

    fastclime = importr('fastclime')
    grdevices = importr('grDevices')

.. code:: python

    def fastclime_plot(data):
        fn = '{uuid}.png'.format(uuid = uuid.uuid4())
        grdevices.png(fn, width = 800, height = 600)
        fastclime.fastclime_plot(data)
        grdevices.dev_off()
        return Image(filename=fn)

.. code:: python

    L = fastclime.fastclime_generator(n = 100, d = 20)


.. parsed-literal::

    Generating data from the multivariate normal distribution with the random graph structure....done.


.. code:: python

    out1 = fastclime.fastclime(L.rx2('data'),0.1)
    O = fastclime.fastclime_lambda(out1.rx2('lambdamtx'), out1.rx2('icovlist'),0.2)
    fastclime_plot(O.rx2('path'))




.. image:: WrappingRLibraries_files/WrappingRLibraries_8_0.png



.. code:: python

    out1 = fastclime.fastclime(cor(L.rx2('data')),0.1)
    O = fastclime.fastclime_lambda(out1.rx2('lambdamtx'), out1.rx2('icovlist'),0.2)
    fastclime_plot(O.rx2('path'))


.. parsed-literal::

    Allocating memory 
    start recovering 
    preparing precision and path matrix list 
    Done! 




.. image:: WrappingRLibraries_files/WrappingRLibraries_9_1.png



.. code:: python

    #generate an LP problem and solve it
    r_matrix = robjects.r['matrix']
    
    A = r_matrix(robjects.FloatVector([-1,-1,0,1,-2,1]), nrow = 3)
    b = robjects.FloatVector([-1,-2,1])
    c = robjects.FloatVector ([-2,3])
    v = fastclime.fastlp(c,A,b)


.. parsed-literal::

    optimal solution found! 


.. code:: python

    v




.. parsed-literal::

    <FloatVector - Python:0x11dec5290 / R:0x1223ecdc8>
    [2.000000, 1.000000]



.. code:: python

    np.array(v)




.. parsed-literal::

    array([ 2.,  1.])



.. code:: python

    #generate an LP problem and solve it
    
    b_bar = robjects.FloatVector([1,1,1])
    c_bar = robjects.FloatVector([1,1])
    fastclime.paralp(c,A,b,c_bar,b_bar)


.. parsed-literal::

    optimal solution found! 




.. parsed-literal::

    <FloatVector - Python:0x11df20e60 / R:0x1223ed030>
    [1.333333, 0.333333]



