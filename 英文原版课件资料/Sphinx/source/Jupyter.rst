
Tour of the Jupyter (IPython3) notebook
=======================================

Installing Jupyter
------------------

If you have not already done so, run

::

    ! pip install -U ipython

Installing other kernels
------------------------

A list of kernels is maintained
`here <https://github.com/ipython/ipython/wiki/IPython-kernels-for-other-languages>`__

Bash
~~~~

.. code:: python

    ! pip install bash_kernel 

R
~

Run this in your shell

.. code:: bash

    sudo apt-get install libzmq3-dev libcurl4-openssl-dev

Run this in R

.. code:: r

    install.packages("devtools")
    install.packages('RCurl')
    library(devtools)
    install_github('armstrtw/rzmq')
    install_github('IRkernel/repr')
    install_github('IRkernel/IRdisplay')
    install_github('IRkernel/IRkernel')
    IRkernel::installspec()

Julia
~~~~~

Download and install `Julia <http://julialang.org/downloads/>`__. Then
run this in Julia

::

    Pkg.add("IJulia")

Octave
~~~~~~

Install octave in your shell

.. code:: bash

    sudo apt-get install octave

.. code:: python

    ! pip install octave_kernel

Matlab
~~~~~~

You must have Matlab on your system - Duke has a site licesne so you
should be able to get it.

.. code:: python

    ! pip install pymatbridge 

.. code:: python

    ! pip install matlab_kernel

Scala
~~~~~

Install `Scala <http://www.scala-lang.org/download/>`__. Add these lines
to ~/.bashrc

::

    export SCALA_HOME=/usr/local/share/scala
    export PATH=$PATH:$SCALA_HOME/bin:$PATH

Follow these instructions from the GitHub site:

Download and unpack pre-packaged binaries `Scala
2.11 <https://oss.sonatype.org/content/repositories/snapshots/sh/jove/jove-scala-cli_2.11/0.1.1-1-SNAPSHOT/jove-scala-cli_2.11-0.1.1-1-SNAPSHOT.tar.gz>`__.
Unpack each downloaded archive(s), and, from a console, go to the bin
sub-directory of the directory it contains. Then run the following to
set-up the corresponding Scala kernel:

.. code:: bash

    ./jove-scala --kernel-spec

Installing extensions
---------------------

See description of extensions
`here <http://jupyter.cs.brynmawr.edu/hub/dblank/public/Jupyter%20Help.ipynb#1.4.2-Enable-Python-3-kernel>`__

And also see the tutorial on `bibliographic
support <http://jupyter.cs.brynmawr.edu/hub/dblank/public/Jupyter%20Notebook%20Users%20Manual.ipynb#5.-Bibliographic-Support>`__
in Jupyter.

Spell-checking
~~~~~~~~~~~~~~

.. code:: python

    ! ipython install-nbextension \
        https://bitbucket.org/ipre/calico/downloads/calico-spell-check-1.0.zip

Notebook sections
~~~~~~~~~~~~~~~~~

.. code:: python

    ! !ipython install-nbextension \
        https://bitbucket.org/ipre/calico/downloads/calico-document-tools-1.0.zip

Adding to configuration
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %%file ~/.ipython/profile_default/static/custom/custom.js
    
    require(["base/js/events"], function (events) {
        events.on("app_initialized.NotebookApp", function () {
            IPython.load_extensions('calico-spell-check', 'calico-document-tools');
            // To turn off automatically creating closing parenthesis and bracket:
            IPython.CodeCell.options_default.cm_config["autoCloseBrackets"] = "";
        });
    });

Installing Python3 while keeping Python2
----------------------------------------

.. code:: python

    %%bash
    
    conda create -n python3 python=3.4 anaconda
    source activate python3
    pip install -U ipython
    ipython3 kernelspec install-self 

Now, restart your notebook server
---------------------------------

If you were successful, you should now see a large number of kernnel
options in the New drop dwon menu.

Note that you can also change the kernel used *for each individual
cell*!

