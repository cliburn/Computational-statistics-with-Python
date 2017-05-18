
Modules and Packaging
=====================

At some point, you will want to organize and distribute your code
library for the whole world to share, preferably on PyPI so that it is
pip installable.

Modules
-------

In Pythoh, any ``.py`` file is a module in that it can be imported.
Because the interpreter runs the entrie file when a moudle is imported,
it is traditional to use a guard to ignore code that should only run
when the file is executed as a script.

.. code:: python

    %%file foo.py
    """
    When this file is imported with `import foo`,
    only `useful_func1()` and `useful_func()` are loaded, 
    and the test code `assert ...` is ignored. However,
    when we run foo.py as a script `python foo.py`, then
    the two assert statements are run.
    Most commonly, the code under `if __naem__ == '__main__':`
    consists of simple examples or test cases for the functions
    defined in the moule.
    """
    
    def useful_func1():
        pass
    
    def useful_fucn2():
        pass
    
    if __name__ == '__main__':
        assert(useful_func1() is None)
        assert(useful_fucn2() is None)


.. parsed-literal::

    Overwriting foo.py


Organization of files in a module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the number of files you write grow large, you will probably want to
orgnize them into their own directory structure. To make a folder a
module, you just need to include a file named ``__init__.py`` in the
folder. This file can be empty. For example, here is a module named
``pkg`` with sub-modules ``sub1`` and ``sub2``.

::

    ./pkg:
    __init__.py foo.py      sub1        sub2

    ./pkg/sub1:
    __init__.py     more_sub1_stuff.py  sub1_stuff.py

    ./pkg/sub2:
    __init__.py sub2_stuff.py

.. code:: python

    import pkg.foo as foo

.. code:: python

    foo.f1()




.. parsed-literal::

    1



.. code:: python

    pkg.foo.f1()




.. parsed-literal::

    1



How to import a module at the same level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within a package, we need to use absolute path names for importing other
modules in the same directory. This prevents confusion as to whether you
want to import a system moudle with the same name. For example,
``foo.sub1.more_sub1_stuff.py`` imports functions from
``foo.sub1.sub1_stuff.py``

.. code:: python

    ! cat pkg/sub1/more_sub1_stuff.py


.. parsed-literal::

    from pkg.sub1.sub1_stuff import g1, g2
    
    def g3():
        return 'g3 uses %s, %s' % (g1(), g2())
    


.. code:: python

    from pkg.sub1.more_sub1_stuff import g3
    
    g3()




.. parsed-literal::

    'g3 uses g1, g2'



How to import a moudle at a different level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again, just use absolute paths. For example, ``sub2_stuff.py`` in the
``sub2`` directory uses functions from ``sub1_stuff.py`` in the ``sub1``
directory:

.. code:: python

    ! cat pkg/sub2/sub2_stuff.py


.. parsed-literal::

    from pkg.sub1.sub1_stuff import g1, g2
    
    def h1():
        return g1()
    
    def h2():
        return g1() + g2()


.. code:: python

    from pkg.sub2.sub2_stuff import h2
    
    h2()




.. parsed-literal::

    'g1g2'



Distributing your package
-------------------------

Suppose we want to distribute our code as a library (for example, on
PyPI so that it cnn be installed with ``pip``). Let's create an
``sta663`` library containing the ``pkg`` package and some other files:

-  ``README.md``: some information about the library
-  ``sta663.py``: a standalone module
-  ``run_sta663.py``: a script (intended for use as
   ``python run_sta.py``)

.. code:: python

    ! ls -R sta663


.. parsed-literal::

    README.txt    run_sta663.py sta663.py
    [34mpkg//           setup.py      [34mtests//
    
    sta663/pkg:
    __init__.py foo.py      [34msub1//        [34msub2//
    
    sta663/pkg/sub1:
    __init__.py        more_sub1_stuff.py sub1_stuff.py
    
    sta663/pkg/sub2:
    __init__.py   sub2_stuff.py
    
    sta663/tests:


.. code:: python

    ! cat sta663/run_sta663.py


.. parsed-literal::

    import pkg.foo as foo
    from pkg.sub1.more_sub1_stuff import g3
    from pkg.sub2.sub2_stuff import h2
    
    print foo.f1()
    print g3()
    print h2()


Using distutils
~~~~~~~~~~~~~~~

All we need to do is to write a ``setup.py`` file.

.. code:: python

    ! cat sta663/setup.py


.. parsed-literal::

    from distutils.core import setup
    
    setup(name = "sta663",
          version = "1.0",
          author='Cliburn Chan',
          author_email='cliburn.chan@duke.edu',
          url='http://people.duke.edu/~ccc14/sta-663/',
          py_modules = ['sta663'],
          packages = ['pkg', 'pkg/sub1', 'pkg/sub2'],
          scripts = ['run_sta663.py']
          )


Build a source archive for distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    %%bash
    
    cd sta663
    python setup.py sdist
    cd -


.. parsed-literal::

    running sdist
    running check
    writing manifest file 'MANIFEST'
    creating sta663-1.0
    creating sta663-1.0/pkg
    creating sta663-1.0/pkg/sub1
    creating sta663-1.0/pkg/sub2
    making hard links in sta663-1.0...
    hard linking README.txt -> sta663-1.0
    hard linking run_sta663.py -> sta663-1.0
    hard linking setup.py -> sta663-1.0
    hard linking sta663.py -> sta663-1.0
    hard linking pkg/__init__.py -> sta663-1.0/pkg
    hard linking pkg/foo.py -> sta663-1.0/pkg
    hard linking pkg/sub1/__init__.py -> sta663-1.0/pkg/sub1
    hard linking pkg/sub1/more_sub1_stuff.py -> sta663-1.0/pkg/sub1
    hard linking pkg/sub1/sub1_stuff.py -> sta663-1.0/pkg/sub1
    hard linking pkg/sub2/__init__.py -> sta663-1.0/pkg/sub2
    hard linking pkg/sub2/sub2_stuff.py -> sta663-1.0/pkg/sub2
    creating dist
    Creating tar archive
    removing 'sta663-1.0' (and everything under it)
    /Users/cliburn/git/STA663-2015/Lectures/Topic23_Packaging


.. parsed-literal::

    warning: sdist: manifest template 'MANIFEST.in' does not exist (using default file list)
    


.. code:: python

    ! ls -R sta663


.. parsed-literal::

    MANIFEST      [34mdist//          run_sta663.py sta663.py
    README.txt    [34mpkg//           setup.py      [34mtests//
    
    sta663/dist:
    sta663-1.0.tar.gz
    
    sta663/pkg:
    __init__.py foo.py      [34msub1//        [34msub2//
    
    sta663/pkg/sub1:
    __init__.py        more_sub1_stuff.py sub1_stuff.py
    
    sta663/pkg/sub2:
    __init__.py   sub2_stuff.py
    
    sta663/tests:


Distribution
~~~~~~~~~~~~

You can now distribute ``sta663-1.0.tar.gz`` to somebody else for
installation in the usual way.

.. code:: python

    %%bash
    
    cp sta663/dist/sta663-1.0.tar.gz /tmp
    cd /tmp
    tar xzf sta663-1.0.tar.gz
    cd sta663-1.0
    python setup.py install


.. parsed-literal::

    running install
    running build
    running build_py
    running build_scripts
    running install_lib
    running install_scripts
    changing mode of /Users/cliburn/anaconda/bin/run_sta663.py to 755
    running install_egg_info
    Writing /Users/cliburn/anaconda/lib/python2.7/site-packages/sta663-1.0-py2.7.egg-info


Distributing to PyPI
^^^^^^^^^^^^^^^^^^^^

Just enter ``python setup.py register`` and respond to the prompts to
register as a new user.

References
~~~~~~~~~~

-  `Python Packaging User
   Guide <https://packaging.python.org/en/latest/index.html>`__
-  `Distributing Python
   Modules <https://docs.python.org/2/distutils/>`__
-  `A more detailed blog post
   tutoiral <https://gehrcke.de/2014/02/distributing-a-python-command-line-application/>`__

