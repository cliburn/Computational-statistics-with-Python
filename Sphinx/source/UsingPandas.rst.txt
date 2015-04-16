
.. code:: python

    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt

.. code:: python

    import warnings
    warnings.filterwarnings("ignore")

Using Pandas
------------

The ``numpy`` module is excellent for numerical computations, but to
handle missing data or arrays with mixed types takes more work. The
``pandas`` module provides objects similar to R's data frames, and these
are more convenient for most statistical analysis. The ``pandas`` module
also provides many mehtods for data import and manipulaiton that we will
explore in this section.

`Pandas for R
Users <http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html>`__

.. code:: python

    import pandas as pd
    import statsmodels.api as sm
    from pandas import Series, DataFrame, Panel
    from string import ascii_lowercase as letters
    from scipy.stats import chisqprob

Series
~~~~~~

Series is a 1D array with axis labels.

.. code:: python

    # Creating a series and extracting elements.
    
    xs = Series(np.arange(10), index=tuple(letters[:10]))
    print xs[:3],'\n'
    print xs[7:], '\n'
    print xs[::3], '\n'
    print xs[['d', 'f', 'h']], '\n'
    print xs.d, xs.f, xs.h


.. parsed-literal::

    a    0
    b    1
    c    2
    dtype: int64 
    
    h    7
    i    8
    j    9
    dtype: int64 
    
    a    0
    d    3
    g    6
    j    9
    dtype: int64 
    
    d    3
    f    5
    h    7
    dtype: int64 
    
    3 5 7


.. code:: python

    # All the numpy functions wiill work with Series objects, and return another Series
    
    y1, y2 = np.mean(xs), np.var(xs)
    y1, y2




.. parsed-literal::

    (4.5, 8.25)



.. code:: python

    # Matplotlib will work on Series objects too
    plt.plot(xs, np.sin(xs), 'r-o', xs, np.cos(xs), 'b-x');



.. image:: UsingPandas_files/UsingPandas_7_0.png


.. code:: python

    # Convert to numpy arrays with values
    
    print xs.values


.. parsed-literal::

    [0 1 2 3 4 5 6 7 8 9]


.. code:: python

    # The Series datatype can also be used to represent time series
    
    import datetime as dt
    from pandas import date_range
    
    # today = dt.date.today()
    today = dt.datetime.strptime('Jan 21 2015', '%b %d %Y') 
    print today, '\n'
    days = date_range(today, periods=35, freq='D')
    ts = Series(np.random.normal(10, 1, len(days)), index=days)
                
    # Extracting elements
    print ts[0:4], '\n'
    print ts['2015-01-21':'2015-01-28'], '\n' # Note - includes end time


.. parsed-literal::

    2015-01-21 00:00:00 
    
    2015-01-21     9.719261
    2015-01-22     8.894461
    2015-01-23    10.074521
    2015-01-24    10.769334
    Freq: D, dtype: float64 
    
    2015-01-21     9.719261
    2015-01-22     8.894461
    2015-01-23    10.074521
    2015-01-24    10.769334
    2015-01-25    10.159401
    2015-01-26     8.992754
    2015-01-27     9.681121
    2015-01-28     9.908445
    Freq: D, dtype: float64 
    


.. code:: python

    # We can geenerate statistics for time ranges with the resample method
    # For example, suppose we are interested in weekly means, standard deviations and sum-of-squares
    
    df = ts.resample(rule='W', how=('mean', 'std', lambda x: sum(x*x)))
    df




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>mean</th>
          <th>std</th>
          <th>&lt;lambda&gt;</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2015-01-25</th>
          <td>  9.923396</td>
          <td> 0.688209</td>
          <td> 494.263430</td>
        </tr>
        <tr>
          <th>2015-02-01</th>
          <td> 10.357088</td>
          <td> 0.848930</td>
          <td> 755.208973</td>
        </tr>
        <tr>
          <th>2015-02-08</th>
          <td> 10.224806</td>
          <td> 0.869441</td>
          <td> 736.362134</td>
        </tr>
        <tr>
          <th>2015-02-15</th>
          <td> 10.672230</td>
          <td> 0.942680</td>
          <td> 802.607338</td>
        </tr>
        <tr>
          <th>2015-02-22</th>
          <td>  9.785174</td>
          <td> 1.012906</td>
          <td> 676.403270</td>
        </tr>
        <tr>
          <th>2015-03-01</th>
          <td>  9.495084</td>
          <td> 1.472653</td>
          <td> 182.481942</td>
        </tr>
      </tbody>
    </table>
    </div>



DataFrame
~~~~~~~~~

For statisticians, a DataFrame is similar to the R dataframe object. For
everyone else, it is like a simple tabular spreadsheet. Each column is a
Series object.

.. code:: python

    # Note that the df object in the previous cell is a DataFrame
    print type(df)


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>


.. code:: python

    # Renaming columns
    # The use of mean and std are problmeatic because there are also methods in dataframe with those names
    # Also, <lambda> is unifnormative
    # So we would like to give better names to the columns of df
    
    df.columns = ('mu', 'sigma', 'sum_of_sq')
    print df


.. parsed-literal::

                       mu     sigma   sum_of_sq
    2015-01-25   9.923396  0.688209  494.263430
    2015-02-01  10.357088  0.848930  755.208973
    2015-02-08  10.224806  0.869441  736.362134
    2015-02-15  10.672230  0.942680  802.607338
    2015-02-22   9.785174  1.012906  676.403270
    2015-03-01   9.495084  1.472653  182.481942


.. code:: python

    # Extracitng columns from a DataFrame
    
    print df.mu, '\n' # by attribute
    print df['sigma'], '\n' # by column name


.. parsed-literal::

    2015-01-25     9.923396
    2015-02-01    10.357088
    2015-02-08    10.224806
    2015-02-15    10.672230
    2015-02-22     9.785174
    2015-03-01     9.495084
    Freq: W-SUN, Name: mu, dtype: float64 
    
    2015-01-25    0.688209
    2015-02-01    0.848930
    2015-02-08    0.869441
    2015-02-15    0.942680
    2015-02-22    1.012906
    2015-03-01    1.472653
    Freq: W-SUN, Name: sigma, dtype: float64 
    


.. code:: python

    # Extracting rows from a DataFrame
    
    print df[1:3], '\n'
    print df['2015-01-21'::2]


.. parsed-literal::

                       mu     sigma   sum_of_sq
    2015-02-01  10.357088  0.848930  755.208973
    2015-02-08  10.224806  0.869441  736.362134 
    
                       mu     sigma   sum_of_sq
    2015-01-25   9.923396  0.688209  494.263430
    2015-02-08  10.224806  0.869441  736.362134
    2015-02-22   9.785174  1.012906  676.403270


.. code:: python

    # Extracting blocks and scalars
    
    print df.iat[2, 2], '\n' # extract an element with iat()
    print df.loc['2015-01-25':'2015-03-01', 'sum_of_sq'], '\n' # indexing by label
    print df.iloc[:3, 2], '\n'  # indexing by position
    print df.ix[:3, 'sum_of_sq'], '\n' # by label OR position


.. parsed-literal::

    736.362134378 
    
    2015-01-25    494.263430
    2015-02-01    755.208973
    2015-02-08    736.362134
    2015-02-15    802.607338
    2015-02-22    676.403270
    2015-03-01    182.481942
    Freq: W-SUN, Name: sum_of_sq, dtype: float64 
    
    2015-01-25    494.263430
    2015-02-01    755.208973
    2015-02-08    736.362134
    Freq: W-SUN, Name: sum_of_sq, dtype: float64 
    
    2015-01-25    494.263430
    2015-02-01    755.208973
    2015-02-08    736.362134
    Freq: W-SUN, Name: sum_of_sq, dtype: float64 
    


.. code:: python

    # Using Boolean conditions for selecting eleements
    
    print df[(df.sigma < 1) & (df.sum_of_sq < 700)], '\n' # need parenthesis because of operator precedence
    print df.query('sigma < 1 and sum_of_sq < 700') # the query() method allows more readable query strings


.. parsed-literal::

                      mu     sigma  sum_of_sq
    2015-01-25  9.923396  0.688209  494.26343 
    
                      mu     sigma  sum_of_sq
    2015-01-25  9.923396  0.688209  494.26343


Panels
~~~~~~

Panels are 3D arrays - they can be thought of as dictionaries of
DataFrames.

.. code:: python

    df= np.random.binomial(100, 0.95, (9,2))
    dm = np.random.binomial(100, 0.9, [12,2])
    dff = DataFrame(df, columns = ['Physics', 'Math'])
    dfm = DataFrame(dm, columns = ['Physics', 'Math'])
    score_panel = Panel({'Girls': dff, 'Boys': dfm})
    print score_panel, '\n'


.. parsed-literal::

    <class 'pandas.core.panel.Panel'>
    Dimensions: 2 (items) x 12 (major_axis) x 2 (minor_axis)
    Items axis: Boys to Girls
    Major_axis axis: 0 to 11
    Minor_axis axis: Physics to Math 
    


.. code:: python

    score_panel['Girls'].transpose()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
          <th>11</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Physics</th>
          <td> 95</td>
          <td> 95</td>
          <td> 96</td>
          <td> 95</td>
          <td> 93</td>
          <td> 95</td>
          <td> 96</td>
          <td> 94</td>
          <td> 96</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>Math</th>
          <td> 95</td>
          <td> 95</td>
          <td> 94</td>
          <td> 92</td>
          <td> 91</td>
          <td> 92</td>
          <td> 96</td>
          <td> 95</td>
          <td> 97</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # find physics and math scores of girls who scored >= 93 in math
    # a DataFrame is returned
    score_panel.ix['Girls', score_panel.Girls.Math >= 93, :]




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Physics</th>
          <th>Math</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 95</td>
          <td> 95</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 95</td>
          <td> 95</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 96</td>
          <td> 94</td>
        </tr>
        <tr>
          <th>6</th>
          <td> 96</td>
          <td> 96</td>
        </tr>
        <tr>
          <th>7</th>
          <td> 94</td>
          <td> 95</td>
        </tr>
        <tr>
          <th>8</th>
          <td> 96</td>
          <td> 97</td>
        </tr>
      </tbody>
    </table>
    </div>



Split-Apply-Combine
~~~~~~~~~~~~~~~~~~~

Many statistical summaries are in the form of split along some property,
then apply a funciton to each subgroup and finally combine the results
into some object. This is known as the 'split-apply-combine' pattern and
implemnented in Pandas via groupby() and a function that can be applied
to each subgroup.

.. code:: python

    # import a DataFrame to play with
    try:
        tips = pd.read_pickle('tips.pic')
    except:
        tips = pd.read_csv('https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/reshape2/tips.csv', )
        tips.to_pickle('tips.pic')

.. code:: python

    tips.head(n=4)




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Unnamed: 0</th>
          <th>total_bill</th>
          <th>tip</th>
          <th>sex</th>
          <th>smoker</th>
          <th>day</th>
          <th>time</th>
          <th>size</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 1</td>
          <td> 16.99</td>
          <td> 1.01</td>
          <td> Female</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 2</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 2</td>
          <td> 10.34</td>
          <td> 1.66</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 3</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 3</td>
          <td> 21.01</td>
          <td> 3.50</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 3</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 4</td>
          <td> 23.68</td>
          <td> 3.31</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # We have an extra set of indices in the first column
    # Let's get rid of it
    
    tips = tips.ix[:, 1:]
    tips.head(n=4)




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>total_bill</th>
          <th>tip</th>
          <th>sex</th>
          <th>smoker</th>
          <th>day</th>
          <th>time</th>
          <th>size</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 16.99</td>
          <td> 1.01</td>
          <td> Female</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 2</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 10.34</td>
          <td> 1.66</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 3</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 21.01</td>
          <td> 3.50</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 3</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 23.68</td>
          <td> 3.31</td>
          <td>   Male</td>
          <td> No</td>
          <td> Sun</td>
          <td> Dinner</td>
          <td> 2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # For an example of the split-apply-combine pattern, we want to see counts by sex and smoker status.
    # In other words, we split by sex and smoker status to get 2x2 groups,
    # then apply the size function to count the number of entries per group
    # and finally combine the results into a new multi-index Series
    
    grouped = tips.groupby(['sex', 'smoker'])
    grouped.size()




.. parsed-literal::

    sex     smoker
    Female  No        54
            Yes       33
    Male    No        97
            Yes       60
    dtype: int64



.. code:: python

    # If you need the margins, use the crosstab function
    
    pd.crosstab(tips.sex, tips.smoker, margins=True)




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>smoker</th>
          <th>No</th>
          <th>Yes</th>
          <th>All</th>
        </tr>
        <tr>
          <th>sex</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Female</th>
          <td>  54</td>
          <td> 33</td>
          <td>  87</td>
        </tr>
        <tr>
          <th>Male</th>
          <td>  97</td>
          <td> 60</td>
          <td> 157</td>
        </tr>
        <tr>
          <th>All</th>
          <td> 151</td>
          <td> 93</td>
          <td> 244</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # If more than 1 column of resutls is generated, a DataFrame is returned
    
    grouped.mean()




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>total_bill</th>
          <th>tip</th>
          <th>size</th>
        </tr>
        <tr>
          <th>sex</th>
          <th>smoker</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">Female</th>
          <th>No</th>
          <td> 18.105185</td>
          <td> 2.773519</td>
          <td> 2.592593</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 17.977879</td>
          <td> 2.931515</td>
          <td> 2.242424</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">Male</th>
          <th>No</th>
          <td> 19.791237</td>
          <td> 3.113402</td>
          <td> 2.711340</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 22.284500</td>
          <td> 3.051167</td>
          <td> 2.500000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # The returned results can be further manipulated via apply()
    # For example, suppose the bill and tips are in USD but we want EUR
    
    import json
    import urllib
    
    # get current conversion rate
    converter = json.loads(urllib.urlopen('http://rate-exchange.appspot.com/currency?from=USD&to=EUR').read())
    print converter
    grouped['total_bill', 'tip'].mean().apply(lambda x: x*converter['rate'])


.. parsed-literal::

    {u'to': u'EUR', u'rate': 0.879191, u'from': u'USD'}




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>total_bill</th>
          <th>tip</th>
        </tr>
        <tr>
          <th>sex</th>
          <th>smoker</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">Female</th>
          <th>No</th>
          <td> 15.917916</td>
          <td> 2.438453</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 15.805989</td>
          <td> 2.577362</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">Male</th>
          <th>No</th>
          <td> 17.400278</td>
          <td> 2.737275</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 19.592332</td>
          <td> 2.682558</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # We can also transform the original data for more convenient analysis
    # For example, suppose we want standardized units for total bill and tips
    
    zscore = lambda x: (x - x.mean())/x.std()
    
    std_grouped = grouped['total_bill', 'tip'].transform(zscore)
    std_grouped.head(n=4)




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>total_bill</th>
          <th>tip</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.153049</td>
          <td>-1.562813</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-1.083042</td>
          <td>-0.975727</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 0.139661</td>
          <td> 0.259539</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 0.445623</td>
          <td> 0.131984</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # Suppose we want to apply a set of functions to only some columns
    grouped['total_bill', 'tip'].agg(['mean', 'min', 'max'])




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th></th>
          <th colspan="3" halign="left">total_bill</th>
          <th colspan="3" halign="left">tip</th>
        </tr>
        <tr>
          <th></th>
          <th></th>
          <th>mean</th>
          <th>min</th>
          <th>max</th>
          <th>mean</th>
          <th>min</th>
          <th>max</th>
        </tr>
        <tr>
          <th>sex</th>
          <th>smoker</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">Female</th>
          <th>No</th>
          <td> 18.105185</td>
          <td> 7.25</td>
          <td> 35.83</td>
          <td> 2.773519</td>
          <td> 1.00</td>
          <td>  5.2</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 17.977879</td>
          <td> 3.07</td>
          <td> 44.30</td>
          <td> 2.931515</td>
          <td> 1.00</td>
          <td>  6.5</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">Male</th>
          <th>No</th>
          <td> 19.791237</td>
          <td> 7.51</td>
          <td> 48.33</td>
          <td> 3.113402</td>
          <td> 1.25</td>
          <td>  9.0</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 22.284500</td>
          <td> 7.25</td>
          <td> 50.81</td>
          <td> 3.051167</td>
          <td> 1.00</td>
          <td> 10.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # We can also apply specific functions to specific columns
    df = grouped.agg({'total_bill': (min, max), 'tip': sum})
    df




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th></th>
          <th>tip</th>
          <th colspan="2" halign="left">total_bill</th>
        </tr>
        <tr>
          <th></th>
          <th></th>
          <th>sum</th>
          <th>min</th>
          <th>max</th>
        </tr>
        <tr>
          <th>sex</th>
          <th>smoker</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="2" valign="top">Female</th>
          <th>No</th>
          <td> 149.77</td>
          <td> 7.25</td>
          <td> 35.83</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td>  96.74</td>
          <td> 3.07</td>
          <td> 44.30</td>
        </tr>
        <tr>
          <th rowspan="2" valign="top">Male</th>
          <th>No</th>
          <td> 302.00</td>
          <td> 7.51</td>
          <td> 48.33</td>
        </tr>
        <tr>
          <th>Yes</th>
          <td> 183.07</td>
          <td> 7.25</td>
          <td> 50.81</td>
        </tr>
      </tbody>
    </table>
    </div>



Using statsmodels
~~~~~~~~~~~~~~~~~

Many of the basic statistical tools available in R are replicted in the
``statsmodels`` package. We will only show one example.

.. code:: python

    # Simulate the genotype for 4 SNPs in a case-control study using an additive genetic model
    
    n = 1000
    status = np.random.choice([0,1], n )
    genotype = np.random.choice([0,1,2], (n,4))
    genotype[status==0] = np.random.choice([0,1,2], (sum(status==0), 4), p=[0.33, 0.33, 0.34])
    genotype[status==1] = np.random.choice([0,1,2], (sum(status==1), 4), p=[0.2, 0.3, 0.5])
    df = DataFrame(np.hstack([status[:, np.newaxis], genotype]), columns=['status', 'SNP1', 'SNP2', 'SNP3', 'SNP4'])
    df.head(6)




.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>status</th>
          <th>SNP1</th>
          <th>SNP2</th>
          <th>SNP3</th>
          <th>SNP4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td> 0</td>
          <td> 2</td>
          <td> 1</td>
          <td> 2</td>
          <td> 0</td>
        </tr>
        <tr>
          <th>1</th>
          <td> 1</td>
          <td> 1</td>
          <td> 0</td>
          <td> 2</td>
          <td> 2</td>
        </tr>
        <tr>
          <th>2</th>
          <td> 1</td>
          <td> 0</td>
          <td> 1</td>
          <td> 2</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>3</th>
          <td> 1</td>
          <td> 2</td>
          <td> 2</td>
          <td> 1</td>
          <td> 2</td>
        </tr>
        <tr>
          <th>4</th>
          <td> 1</td>
          <td> 1</td>
          <td> 2</td>
          <td> 0</td>
          <td> 1</td>
        </tr>
        <tr>
          <th>5</th>
          <td> 1</td>
          <td> 0</td>
          <td> 0</td>
          <td> 1</td>
          <td> 2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # Use statsmodels to fit a logistic regression to  the data
    fit1 = sm.Logit.from_formula('status ~ %s' % '+'.join(df.columns[1:]), data=df).fit()
    fit1.summary()


.. parsed-literal::

    Optimization terminated successfully.
             Current function value: 0.642824
             Iterations 5




.. raw:: html

    <table class="simpletable">
    <caption>Logit Regression Results</caption>
    <tr>
      <th>Dep. Variable:</th>      <td>status</td>      <th>  No. Observations:  </th>  <td>  1000</td>  
    </tr>
    <tr>
      <th>Model:</th>               <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   995</td>  
    </tr>
    <tr>
      <th>Method:</th>               <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
    </tr>
    <tr>
      <th>Date:</th>          <td>Thu, 22 Jan 2015</td> <th>  Pseudo R-squ.:     </th>  <td>0.07259</td> 
    </tr>
    <tr>
      <th>Time:</th>              <td>15:34:43</td>     <th>  Log-Likelihood:    </th> <td> -642.82</td> 
    </tr>
    <tr>
      <th>converged:</th>           <td>True</td>       <th>  LL-Null:           </th> <td> -693.14</td> 
    </tr>
    <tr>
      <th> </th>                      <td> </td>        <th>  LLR p-value:       </th> <td>7.222e-21</td>
    </tr>
    </table>
    <table class="simpletable">
    <tr>
          <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
    </tr>
    <tr>
      <th>Intercept</th> <td>   -1.7409</td> <td>    0.203</td> <td>   -8.560</td> <td> 0.000</td> <td>   -2.140    -1.342</td>
    </tr>
    <tr>
      <th>SNP1</th>      <td>    0.4306</td> <td>    0.083</td> <td>    5.173</td> <td> 0.000</td> <td>    0.267     0.594</td>
    </tr>
    <tr>
      <th>SNP2</th>      <td>    0.3155</td> <td>    0.081</td> <td>    3.882</td> <td> 0.000</td> <td>    0.156     0.475</td>
    </tr>
    <tr>
      <th>SNP3</th>      <td>    0.2255</td> <td>    0.082</td> <td>    2.750</td> <td> 0.006</td> <td>    0.065     0.386</td>
    </tr>
    <tr>
      <th>SNP4</th>      <td>    0.5341</td> <td>    0.083</td> <td>    6.404</td> <td> 0.000</td> <td>    0.371     0.698</td>
    </tr>
    </table>



.. code:: python

    # Alternative using GLM - similar to R
    fit2 = sm.GLM.from_formula('status ~ SNP1 + SNP2 + SNP3 + SNP4', data=df, family=sm.families.Binomial()).fit()
    print fit2.summary()
    print chisqprob(fit2.null_deviance - fit2.deviance, fit2.df_model)
    print(fit2.null_deviance - fit2.deviance, fit2.df_model)


.. parsed-literal::

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                 status   No. Observations:                 1000
    Model:                            GLM   Df Residuals:                      995
    Model Family:                Binomial   Df Model:                            4
    Link Function:                  logit   Scale:                             1.0
    Method:                          IRLS   Log-Likelihood:                -642.82
    Date:                Thu, 22 Jan 2015   Deviance:                       1285.6
    Time:                        15:34:43   Pearson chi2:                 1.01e+03
    No. Iterations:                     5                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------
    Intercept     -1.7409      0.203     -8.560      0.000        -2.140    -1.342
    SNP1           0.4306      0.083      5.173      0.000         0.267     0.594
    SNP2           0.3155      0.081      3.882      0.000         0.156     0.475
    SNP3           0.2255      0.082      2.750      0.006         0.065     0.386
    SNP4           0.5341      0.083      6.404      0.000         0.371     0.698
    ==============================================================================
    7.22229516479e-21
    (100.63019840179481, 4)


Using R from IPython
--------------------

While Python support for statstical computing is rapidly improving
(especially with the pandas, statsmodels and scikit-learn modules), the
R ecosystem is staill vastly larger. However, we can have our cake and
eat it too, since IPyhton allows us to run R (almost) seamlessly with
the Rmagic (rpy2.ipython) extension.

There are two ways to use Rmagic - using %R (appleis to single line) and
%%R (applies to entire cell). Python objects can be passed into R with
the -i flag and R objects pased out with the -o flag.

.. code:: python

    ! pip install ggplot &> /dev/null

Using Rmagic
~~~~~~~~~~~~

.. code:: python

    %load_ext rpy2.ipython

.. code:: python

    %%R -i df,status -o fit
    
    fit <- glm(status ~ ., data=df)
    print(summary(fit))
    print(fit$null.deviance - fit$deviance)
    print(fit$df.null - fit$df.residual)
    with(fit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))



.. parsed-literal::

    
    Call:
    glm(formula = status ~ ., data = df)
    
    Deviance Residuals: 
        Min       1Q   Median       3Q      Max  
    -0.7927  -0.4464   0.2073   0.4301   0.8999  
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  0.10014    0.04323   2.316  0.02075 *  
    SNP1         0.09904    0.01874   5.285 1.55e-07 ***
    SNP2         0.07217    0.01836   3.932 9.01e-05 ***
    SNP3         0.05135    0.01856   2.767  0.00576 ** 
    SNP4         0.12372    0.01869   6.620 5.86e-11 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for gaussian family taken to be 0.2269642)
    
        Null deviance: 250.00  on 999  degrees of freedom
    Residual deviance: 225.83  on 995  degrees of freedom
    AIC: 1361.9
    
    Number of Fisher Scoring iterations: 2
    
    [1] 24.16657
    [1] 4
    [1] 7.396261e-05



Using rpy2 directly
^^^^^^^^^^^^^^^^^^^

.. code:: python

    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    
    base = importr('base')
    
    fit_full = ro.r("lm('mpg ~ wt + cyl', data=mtcars)")
    print(base.summary(fit_full))


.. parsed-literal::

    
    Call:
    lm(formula = "mpg ~ wt + cyl", data = mtcars)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -4.2893 -1.5512 -0.4684  1.5743  6.1004 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  39.6863     1.7150  23.141  < 2e-16 ***
    wt           -3.1910     0.7569  -4.216 0.000222 ***
    cyl          -1.5078     0.4147  -3.636 0.001064 ** 
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 2.568 on 29 degrees of freedom
    Multiple R-squared:  0.8302,	Adjusted R-squared:  0.8185 
    F-statistic: 70.91 on 2 and 29 DF,  p-value: 6.809e-12
    
    


Using R from pandas
~~~~~~~~~~~~~~~~~~~

Reading R dataset into Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    import pandas.rpy.common as com
    
    df = com.load_data('mtcars')
    print df.head(n=6)


.. parsed-literal::

        mpg  cyl  disp   hp  drat     wt   qsec  vs  am  gear  carb
    0  21.0    6   160  110  3.90  2.620  16.46   0   1     4     4
    1  21.0    6   160  110  3.90  2.875  17.02   0   1     4     4
    2  22.8    4   108   93  3.85  2.320  18.61   1   1     4     1
    3  21.4    6   258  110  3.08  3.215  19.44   1   0     3     1
    4  18.7    8   360  175  3.15  3.440  17.02   0   0     3     2
    5  18.1    6   225  105  2.76  3.460  20.22   1   0     3     1


.. code:: python

    %load_ext version_information
    
    %version_information numpy, matplotlib, pandas, statsmodels




.. raw:: html

    <table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.9 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>2.3.1</td></tr><tr><td>OS</td><td>Darwin 13.4.0 x86_64 i386 64bit</td></tr><tr><td>numpy</td><td>1.9.1</td></tr><tr><td>matplotlib</td><td>1.4.2</td></tr><tr><td>pandas</td><td>0.15.1</td></tr><tr><td>statsmodels</td><td>0.5.0</td></tr><tr><td colspan='2'>Thu Jan 22 15:34:45 2015 EST</td></tr></table>


