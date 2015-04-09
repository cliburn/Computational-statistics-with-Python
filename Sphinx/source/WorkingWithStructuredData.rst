
.. code:: python

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    %precision 4
    import os, sys, glob

Working with structured data
============================

Using SQLite3
-------------

Working example dataset
~~~~~~~~~~~~~~~~~~~~~~~

This data contains the survival time after receiving a heart transplant,
the age of the patient and whether or not the survival time was censored

-  Number of Observations - 69
-  Number of Variables - 3

Variable name definitions:: \* death - Days after surgery until death \*
age - age at the time of surgery \* censored - indicates if an
observation is censored. 1 is uncensored

.. code:: python

    import statsmodels.api as sm
    heart = sm.datasets.heart.load_pandas().data
    heart.take(np.random.choice(len(heart), 6))




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>survival</th>
          <th>censors</th>
          <th>age</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>66</th>
          <td>  110</td>
          <td> 0</td>
          <td> 23.7</td>
        </tr>
        <tr>
          <th>24</th>
          <td> 1367</td>
          <td> 0</td>
          <td> 48.6</td>
        </tr>
        <tr>
          <th>30</th>
          <td>  897</td>
          <td> 1</td>
          <td> 46.1</td>
        </tr>
        <tr>
          <th>67</th>
          <td>   13</td>
          <td> 0</td>
          <td> 28.9</td>
        </tr>
        <tr>
          <th>49</th>
          <td>  499</td>
          <td> 0</td>
          <td> 52.2</td>
        </tr>
        <tr>
          <th>35</th>
          <td>  322</td>
          <td> 1</td>
          <td> 48.1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    import sqlite3
    conn = sqlite3.connect('heart.db')

Creating and populating a table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS transplant
                 (survival integer, censors integer, age real)''')
    
    c.executemany("insert into transplant(survival, censors, age) values (?, ?, ?)", heart.values);

SQL queries
~~~~~~~~~~~

SQL Queries take the form

.. code:: sql

    select (distinct) ... from ... (limit ...)
    where ...
    groupby ..
    order by ...

where most of the query apart from the ``select ... from ...`` are
optional.

Selecting all columns, first 10 rows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    for row in c.execute('''select * from transplant limit 5;'''):
        print row


.. parsed-literal::

    (15, 1, 54.3)
    (3, 1, 40.4)
    (624, 1, 51.0)
    (46, 1, 42.5)
    (127, 1, 48.0)


Using where to filter rows
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # only find censored data for subjects < 40 years old
    for row in c.execute('''
    select * from transplant 
    where censors=0 and age < 40 limit 5;'''):
        print row


.. parsed-literal::

    (1775, 0, 33.3)
    (1106, 0, 36.8)
    (875, 0, 38.9)
    (815, 0, 32.7)
    (592, 0, 26.7)


Using SQL functions
^^^^^^^^^^^^^^^^^^^

.. code:: python

    for row in c.execute('''select count(*), avg(age) from transplant where censors=0 and age < 40;'''):
        print row


.. parsed-literal::

    (9, 31.43333333333333)


Using groupby to find number of cnesored and uncensored subjects and thier average age
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    query = '''
    select censors, count(*), avg(age) from transplant 
    group by censors;
    '''
    for row in c.execute(query):
        print row


.. parsed-literal::

    (0, 24, 41.729166666666664)
    (1, 45, 48.484444444444456)


Using having to filter grouped results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    query = '''
    select censors, count(*), avg(age) from transplant 
    group by censors
    having avg(age) < 45;
    '''
    for row in c.execute(query):
        print row


.. parsed-literal::

    (0, 24, 41.729166666666664)


Using order by to sort results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    query = '''
    select * from transplant 
    where age < 40
    order by age desc;
    '''
    for row in c.execute(query):
        print row


.. parsed-literal::

    (875, 0, 38.9)
    (1106, 0, 36.8)
    (44, 1, 36.2)
    (1, 0, 35.2)
    (1775, 0, 33.3)
    (815, 0, 32.7)
    (12, 1, 29.2)
    (13, 0, 28.9)
    (592, 0, 26.7)
    (167, 0, 26.7)
    (110, 0, 23.7)
    (228, 1, 19.7)


Reading into a numpy structured array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    result = c.execute(query).fetchall()
    arr = np.fromiter(result, dtype='i4,i4,f4')
    arr.dtype.names = ['survival', 'censors', 'age']
    print '\n'.join(map(str, arr))


.. parsed-literal::

    (875, 0, 38.900001525878906)
    (1106, 0, 36.79999923706055)
    (44, 1, 36.20000076293945)
    (1, 0, 35.20000076293945)
    (1775, 0, 33.29999923706055)
    (815, 0, 32.70000076293945)
    (12, 1, 29.200000762939453)
    (13, 0, 28.899999618530273)
    (592, 0, 26.700000762939453)
    (167, 0, 26.700000762939453)
    (110, 0, 23.700000762939453)
    (228, 1, 19.700000762939453)


Reading into a numpy regular array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from itertools import chain
    result = c.execute(query).fetchall()
    arr = np.fromiter(chain.from_iterable(result), dtype=np.float)
    print arr.reshape(-1,3)


.. parsed-literal::

    [[  8.7500e+02   0.0000e+00   3.8900e+01]
     [  1.1060e+03   0.0000e+00   3.6800e+01]
     [  4.4000e+01   1.0000e+00   3.6200e+01]
     [  1.0000e+00   0.0000e+00   3.5200e+01]
     [  1.7750e+03   0.0000e+00   3.3300e+01]
     [  8.1500e+02   0.0000e+00   3.2700e+01]
     [  1.2000e+01   1.0000e+00   2.9200e+01]
     [  1.3000e+01   0.0000e+00   2.8900e+01]
     [  5.9200e+02   0.0000e+00   2.6700e+01]
     [  1.6700e+02   0.0000e+00   2.6700e+01]
     [  1.1000e+02   0.0000e+00   2.3700e+01]
     [  2.2800e+02   1.0000e+00   1.9700e+01]]


Working wiht multiple tables in SQL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will consturct a new database with 2 tables to illustrate the concept
of joins.

.. code:: python

    conn1 = sqlite3.connect('samples.db')
    c1 = conn1.cursor()
    
    c1.execute(
    '''
    CREATE TABLE IF NOT EXISTS t1(
      ID TEXT,
      Name TEXT,
      Value Real);
    ''')
    
    c1.execute('''
    CREATE TABLE IF NOT EXISTS t2(
      ID TEXT,
      Name TEXT,
      Value Real,
      Age INTEGER);
    ''');
    
    from string import ascii_lowercase
    for i in range(5):
        c1.execute('''insert into t1(ID, Name, Value) values (%d, '%s', %.2f)''' % (i, ascii_lowercase[i], i*i));
        c1.execute('''insert into t2(ID, Name, Value, Age) values (%d, '%s', %.2f, %d)''' % (i*2, ascii_lowercase[i*2], i*i+5, 10*i));

Cartesian product
^^^^^^^^^^^^^^^^^

.. code:: python

    # Without specifiying a join, the result is all possible combinations
    query = '''
    select t1.ID, t2.ID from t1, t2; 
    '''
    for row in c1.execute(query):
        print row


.. parsed-literal::

    (u'0', u'0')
    (u'0', u'2')
    (u'0', u'4')
    (u'0', u'6')
    (u'0', u'8')
    (u'1', u'0')
    (u'1', u'2')
    (u'1', u'4')
    (u'1', u'6')
    (u'1', u'8')
    (u'2', u'0')
    (u'2', u'2')
    (u'2', u'4')
    (u'2', u'6')
    (u'2', u'8')
    (u'3', u'0')
    (u'3', u'2')
    (u'3', u'4')
    (u'3', u'6')
    (u'3', u'8')
    (u'4', u'0')
    (u'4', u'2')
    (u'4', u'4')
    (u'4', u'6')
    (u'4', u'8')


Inner joins
^^^^^^^^^^^

.. code:: python

    # Inner join (intersection)
    query = '''
    select t1.ID, t2.ID, t1.value, t2.value, t1.value * t2.value from t1, t2
    where t1.ID = t2.ID;
    '''
    for row in c1.execute(query):
        print row


.. parsed-literal::

    (u'0', u'0', 0.0, 5.0, 0.0)
    (u'2', u'2', 4.0, 6.0, 24.0)
    (u'4', u'4', 16.0, 9.0, 144.0)


.. code:: python

    # left join keeps all values from the left table (t2) 
    # and values from the right (t1) where there is a match
    query = '''
    select t1.id, t2.ID, t1.value, t2.value from t2 left join t1 on t1.ID = t2.ID
    '''
    for row in c1.execute(query):
        print row


.. parsed-literal::

    (u'0', u'0', 0.0, 5.0)
    (u'2', u'2', 4.0, 6.0)
    (u'4', u'4', 16.0, 9.0)
    (None, u'6', None, 14.0)
    (None, u'8', None, 21.0)


.. code:: python

    # same join but we swtich left and right tables
    query = '''
    select t1.ID, t2.ID, t1.value, t2.value from t1 left join t2 on t1.ID = t2.ID
    '''
    for row in c1.execute(query):
        print row


.. parsed-literal::

    (u'0', u'0', 0.0, 5.0)
    (u'1', None, 1.0, None)
    (u'2', u'2', 4.0, 6.0)
    (u'3', None, 9.0, None)
    (u'4', u'4', 16.0, 9.0)


Self-joins
^^^^^^^^^^

.. code:: python

    # we can join a table to itself by using aliases 
    # lets add a few more rows to t1 which may have the same id and name but different values
    
    for i in range(5):
        c1.execute('''insert into t1(ID, Name, Value) values (%d, '%s', %.2f)''' % (i, ascii_lowercase[i], i*i*i));
    
    for row in c1.execute('select * from t1;'):
        print row


.. parsed-literal::

    (u'0', u'a', 0.0)
    (u'1', u'b', 1.0)
    (u'2', u'c', 4.0)
    (u'3', u'd', 9.0)
    (u'4', u'e', 16.0)
    (u'0', u'a', 0.0)
    (u'1', u'b', 1.0)
    (u'2', u'c', 8.0)
    (u'3', u'd', 27.0)
    (u'4', u'e', 64.0)


.. code:: python

    # Now use a self-join to find paired values for the same ID and name
    
    query = '''
    select t1a.ID, t1a.Name, t1a.value, t1b.value from t1 as t1a, t1 as t1b
    where t1a.Name = t1b.Name and t1a.Value < t1b.Value
    order by t1a.ID ASC;
    '''
    for row in c1.execute(query):
        print row


.. parsed-literal::

    (u'2', u'c', 4.0, 8.0)
    (u'3', u'd', 9.0, 27.0)
    (u'4', u'e', 16.0, 64.0)


Basic concepts of database normalization
----------------------------------------

In which we convert a dataframe into a normalized database.

.. code:: python

    names = ['ann', 'bob', 'ann', 'bob', 'carl', 'delia', 'ann']
    tests = ['wbc', 'wbc', 'rbc', 'rbc', 'wbc', 'rbc', 'platelets']
    values1 = [10, 11.2, 300, 204, 9.8, 340, 125]
    values2 = [10.6, 13.2, 322, 214, 10.3, 343, 145]
    df = pd.DataFrame([names, tests, values1, values2]).T
    df.columns = ['names', 'tests', 'values1', 'values2']
    df




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>names</th>
          <th>tests</th>
          <th>values1</th>
          <th>values2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>   ann</td>
          <td>       wbc</td>
          <td>   10</td>
          <td> 10.6</td>
        </tr>
        <tr>
          <th>1</th>
          <td>   bob</td>
          <td>       wbc</td>
          <td> 11.2</td>
          <td> 13.2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>   ann</td>
          <td>       rbc</td>
          <td>  300</td>
          <td>  322</td>
        </tr>
        <tr>
          <th>3</th>
          <td>   bob</td>
          <td>       rbc</td>
          <td>  204</td>
          <td>  214</td>
        </tr>
        <tr>
          <th>4</th>
          <td>  carl</td>
          <td>       wbc</td>
          <td>  9.8</td>
          <td> 10.3</td>
        </tr>
        <tr>
          <th>5</th>
          <td> delia</td>
          <td>       rbc</td>
          <td>  340</td>
          <td>  343</td>
        </tr>
        <tr>
          <th>6</th>
          <td>   ann</td>
          <td> platelets</td>
          <td>  125</td>
          <td>  145</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # names are put into their own table so there is no dubplication
    
    name_table = pd.DataFrame(df['names'].unique(), columns=['name'])
    name_table['name_id'] = name_table.index
    columns = ['name_id', 'name']
    name_table[columns]




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name_id</th>
          <th>name</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 0</td>
          <td>   ann</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 1</td>
          <td>   bob</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 2</td>
          <td>  carl</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 3</td>
          <td> delia</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # tests are put inot their own table so there is no duplication
    
    test_table = pd.DataFrame(df['tests'].unique(), columns=['test'])
    test_table['test_id'] = test_table.index
    columns = ['test_id', 'test']
    test_table[columns]




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>test_id</th>
          <th>test</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 0</td>
          <td>       wbc</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 1</td>
          <td>       rbc</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 2</td>
          <td> platelets</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # the values1 and values2 correspond to visit 1 and 2, so
    # we create a visits table
    
    visit_table = pd.DataFrame([1,2], columns=['visit'])
    visit_table['visit_id'] = visit_table.index
    columns = ['visit_id', 'visit']
    visit_table[columns]




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>visit_id</th>
          <th>visit</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 0</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 1</td>
          <td> 2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # finally, we link each value to a triple(name_id, test_id, visit_id)
    
    value_table = pd.DataFrame([
        [0,0,0,10], [1,0,0,11.2], [0,1,0,300], [1,1,0,204], [2,0,0,9.8], [3,1,0,340], [0,2,0,125],
       [0,0,1,10.6], [1,0,1,13.2], [0,1,1,322], [1,1,1,214], [2,0,1,10.3], [3,1,1,343], [0,2,1,145]
    ], columns=['name_id', 'test_id', 'visit_id', 'value'])
    value_table




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>name_id</th>
          <th>test_id</th>
          <th>visit_id</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0 </th>
          <td> 0</td>
          <td> 0</td>
          <td> 0</td>
          <td>  10.0</td>
        </tr>
        <tr>
          <th>1 </th>
          <td> 1</td>
          <td> 0</td>
          <td> 0</td>
          <td>  11.2</td>
        </tr>
        <tr>
          <th>2 </th>
          <td> 0</td>
          <td> 1</td>
          <td> 0</td>
          <td> 300.0</td>
        </tr>
        <tr>
          <th>3 </th>
          <td> 1</td>
          <td> 1</td>
          <td> 0</td>
          <td> 204.0</td>
        </tr>
        <tr>
          <th>4 </th>
          <td> 2</td>
          <td> 0</td>
          <td> 0</td>
          <td>   9.8</td>
        </tr>
        <tr>
          <th>5 </th>
          <td> 3</td>
          <td> 1</td>
          <td> 0</td>
          <td> 340.0</td>
        </tr>
        <tr>
          <th>6 </th>
          <td> 0</td>
          <td> 2</td>
          <td> 0</td>
          <td> 125.0</td>
        </tr>
        <tr>
          <th>7 </th>
          <td> 0</td>
          <td> 0</td>
          <td> 1</td>
          <td>  10.6</td>
        </tr>
        <tr>
          <th>8 </th>
          <td> 1</td>
          <td> 0</td>
          <td> 1</td>
          <td>  13.2</td>
        </tr>
        <tr>
          <th>9 </th>
          <td> 0</td>
          <td> 1</td>
          <td> 1</td>
          <td> 322.0</td>
        </tr>
        <tr>
          <th>10</th>
          <td> 1</td>
          <td> 1</td>
          <td> 1</td>
          <td> 214.0</td>
        </tr>
        <tr>
          <th>11</th>
          <td> 2</td>
          <td> 0</td>
          <td> 1</td>
          <td>  10.3</td>
        </tr>
        <tr>
          <th>12</th>
          <td> 3</td>
          <td> 1</td>
          <td> 1</td>
          <td> 343.0</td>
        </tr>
        <tr>
          <th>13</th>
          <td> 0</td>
          <td> 2</td>
          <td> 1</td>
          <td> 145.0</td>
        </tr>
      </tbody>
    </table>
    </div>



At the end of the normalizaiton, we have gone from 1 dataframe with
multiple redundancies to 4 tables with unique entries in each row. This
organization helps maintain data integrity and is necesssary for
effficeincy as the number of test values grows, possibly into millions
of rows. As we have seen, we can use SQL queries to recreate the
origianl dataformat if that is more convenient for analysis.

Using HDF5
----------

When your data consists of many numerical and matrices, each of which is
relatively independent, relational databases offer little benefit, and
it is more efficient to use HDF5 (Hierarchical Data Format) for storage.
For example, your data may come from a simulation which generates a 3D
matrix and a list of count data at every iteration.

.. code:: python

    import h5py
    
    f = h5py.File('simulation.h5')

.. code:: python

    for i in range(10): # iterations in simulation
        xs = np.random.random((100,100,100))
        ys = np.random.randint(0,100,(i+1)*10)
        group = f.create_group('Iteration%03d' % i)
        group.create_dataset('xs', data=xs)
        group.create_dataset('ys', data=ys)

.. code:: python

    f.keys()




.. parsed-literal::

    [u'Iteration000',
     u'Iteration001',
     u'Iteration002',
     u'Iteration003',
     u'Iteration004',
     u'Iteration005',
     u'Iteration006',
     u'Iteration007',
     u'Iteration008',
     u'Iteration009']



.. code:: python

    f['Iteration008'].keys()




.. parsed-literal::

    [u'xs', u'ys']



.. code:: python

    g8 = f['Iteration008']
    print g8['xs'][2:5,2:5,2:5]
    print g8['ys'][-10:]


.. parsed-literal::

    [[[ 0.0367  0.2883  0.5562]
      [ 0.9494  0.5614  0.1159]
      [ 0.8887  0.7396  0.891 ]]
    
     [[ 0.7552  0.1539  0.216 ]
      [ 0.6671  0.4682  0.9107]
      [ 0.5565  0.5443  0.1665]]
    
     [[ 0.3972  0.1205  0.9487]
      [ 0.7874  0.3466  0.2818]
      [ 0.1248  0.0161  0.6898]]]
    [37 69  5 15 10 44 20 73 74 24]


Interfacing withPandas
----------------------

.. code:: python

    import pandas as pd

.. code:: python

    df = pd.read_sql('select * from transplant;', conn)

.. code:: python

    df.take(np.random.randint(0, len(df), 6))




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>survival</th>
          <th>censors</th>
          <th>age</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>8 </th>
          <td>  23</td>
          <td> 1</td>
          <td> 56.9</td>
        </tr>
        <tr>
          <th>38</th>
          <td> 815</td>
          <td> 0</td>
          <td> 32.7</td>
        </tr>
        <tr>
          <th>12</th>
          <td> 730</td>
          <td> 1</td>
          <td> 58.4</td>
        </tr>
        <tr>
          <th>58</th>
          <td> 339</td>
          <td> 0</td>
          <td> 54.4</td>
        </tr>
        <tr>
          <th>53</th>
          <td> 439</td>
          <td> 0</td>
          <td> 52.9</td>
        </tr>
        <tr>
          <th>27</th>
          <td> 994</td>
          <td> 1</td>
          <td> 48.6</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    df1 = pd.read_sql('select t1.name, t2.value, t2.age from t1, t2 where t1.name = t2.name;', conn1)

.. code:: python

    df1




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Name</th>
          <th>Value</th>
          <th>Age</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> a</td>
          <td> 5</td>
          <td>  0</td>
        </tr>
        <tr>
          <th>1</th>
          <td> c</td>
          <td> 6</td>
          <td> 10</td>
        </tr>
        <tr>
          <th>2</th>
          <td> e</td>
          <td> 9</td>
          <td> 20</td>
        </tr>
        <tr>
          <th>3</th>
          <td> a</td>
          <td> 5</td>
          <td>  0</td>
        </tr>
        <tr>
          <th>4</th>
          <td> c</td>
          <td> 6</td>
          <td> 10</td>
        </tr>
        <tr>
          <th>5</th>
          <td> e</td>
          <td> 9</td>
          <td> 20</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    c.close()
    c1.close()
    conn.close()
    conn1.close()

.. code:: python

    store = pd.HDFStore('dump.h5')
    store['transplant'] = df
    store['tables'] = df1
    store.close()


.. parsed-literal::

    /Users/cliburn/anaconda/lib/python2.7/site-packages/pandas/io/pytables.py:2453: PerformanceWarning: 
    your performance may suffer as PyTables will pickle object types that it cannot
    map directly to c-types [inferred_type->unicode,key->block2_values] [items->['Name']]
    
      warnings.warn(ws, PerformanceWarning)


.. code:: python

    transplant_df = pd.read_hdf('dump.h5', 'transplant')
    transplant_df.take(np.random.randint(0, len(df), 6))




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>survival</th>
          <th>censors</th>
          <th>age</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>50</th>
          <td>  305</td>
          <td> 0</td>
          <td> 49.3</td>
        </tr>
        <tr>
          <th>3 </th>
          <td>   46</td>
          <td> 1</td>
          <td> 42.5</td>
        </tr>
        <tr>
          <th>0 </th>
          <td>   15</td>
          <td> 1</td>
          <td> 54.3</td>
        </tr>
        <tr>
          <th>22</th>
          <td>    1</td>
          <td> 1</td>
          <td> 41.5</td>
        </tr>
        <tr>
          <th>47</th>
          <td>   63</td>
          <td> 1</td>
          <td> 56.4</td>
        </tr>
        <tr>
          <th>19</th>
          <td> 1549</td>
          <td> 0</td>
          <td> 40.6</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    table_df = pd.read_hdf('dump.h5', 'tables')
    table_df




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Name</th>
          <th>Value</th>
          <th>Age</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> a</td>
          <td> 5</td>
          <td>  0</td>
        </tr>
        <tr>
          <th>1</th>
          <td> c</td>
          <td> 6</td>
          <td> 10</td>
        </tr>
        <tr>
          <th>2</th>
          <td> e</td>
          <td> 9</td>
          <td> 20</td>
        </tr>
        <tr>
          <th>3</th>
          <td> a</td>
          <td> 5</td>
          <td>  0</td>
        </tr>
        <tr>
          <th>4</th>
          <td> c</td>
          <td> 6</td>
          <td> 10</td>
        </tr>
        <tr>
          <th>5</th>
          <td> e</td>
          <td> 9</td>
          <td> 20</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    store 




.. parsed-literal::

    <class 'pandas.io.pytables.HDFStore'>
    File path: dump.h5
    File is CLOSED



.. code:: python

    store = pd.HDFStore('dump.h5')

.. code:: python

    store.keys()




.. parsed-literal::

    ['/tables', '/transplant']



.. code:: python

    store.close()

