
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

Julia and Python
================

You will need to install Julia from http://julialang.org/downloads/ and
make sure it is on your path. On Ubuntu you can install via ``apt-get``
with

::

    sudo add-apt-repository ppa:staticfloat/juliareleases
    sudo add-apt-repository ppa:staticfloat/julia-deps
    sudo apt-get update
    sudo apt-get install julia

Within an instance of Julia, run the following commands:

::

    Pkg.update()
    Pkg.add("PyCall")
    Pkg.add("IJulia")

Then finally install ``pyjulia`` from
https://github.com/JuliaLang/pyjulia

Make sure that you can start ``julia`` line - if not, add a symlink to
it.

.. code:: python

    %install_ext https://raw.githubusercontent.com/JuliaLang/pyjulia/master/julia/magic.py


.. parsed-literal::

    Installed magic.py. To use it, type:
      %load_ext magic


.. code:: python

    %load_ext magic
    %julia @pyimport matplotlib.pyplot as plt
    %julia @pyimport numpy as np
    %julia @pyimport numpy.random as npr


.. parsed-literal::

    Initializing Julia interpreter. This may take some time...


Defining a function in Julia
----------------------------

.. code:: python

    %%julia
    
    function fib(n)
        a, b = 0.0,  1.0
        for i = 1:n
            a, b = a+b, a
        end
        return a
    end




.. parsed-literal::

    <PyCall.jlwrap fib>



Using it in Python
------------------

.. code:: python

    jfib = %julia fib
    
    jfib(100)




.. parsed-literal::

    354224848179261997056.0000



Using Python libraries in Julia
-------------------------------

.. code:: python

    %%julia
    
    xs = npr.multivariate_normal([0,0], np.eye(2), 100)
    plt.scatter(xs[:,1], xs[:, 2], s=30);




.. parsed-literal::

    <matplotlib.collections.PathCollection at 0x1220f3a50>




.. image:: FromJuliaToPython_files/FromJuliaToPython_10_1.png


Benchmarking
~~~~~~~~~~~~

.. code:: python

    %timeit jfib(100)


.. parsed-literal::

    10000 loops, best of 3: 22.9 µs per loop


