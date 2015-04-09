
Introduction to Python
======================

This lecture is based loosely on the online tutorial :
http://www.afterhoursprogramming.com/tutorial/Python/Introduction/

We will be using Python a fair amount in this class. Python is a
high-level scripting language that offers an interactive programming
environment. We assume programming experience, so this lecture will
focus on the unique properties of Python.

Programming languages generally have the following common ingredients:
variables, operators, iterators, conditional statements, functions
(built-in and user defined) and higher-order data structures. We will
look at these in Python and highlight qualities unique to this language.

Variables
---------

Variables in Python are defined and typed for you when you set a value
to them.

.. code:: python

    my_variable = 2 
    print(my_variable)
    type(my_variable)



.. parsed-literal::

    2




.. parsed-literal::

    int



This makes variable definition easy for the programmer. As usual,
though, great power comes with great responsibility. For example:

.. code:: python

    my_varible = my_variable+1
    print (my_variable)


.. parsed-literal::

    2


"If you leave out word, spell-check will not put the word in you" --
Taylor Mali, The the impotence of proofreading

If you accidentally mistype a variable name, Python will not catch it
for you. This can lead to bugs that can be hard to track - so beware.

Types and Typecasting
~~~~~~~~~~~~~~~~~~~~~

The usual typecasting is available in Python, so it is easy to convert
strings to ints or floats, floats to ints, etc. The syntax is slightly
different than C:

.. code:: python

    a = "1"
    b = 5 
    print(a+b)



::


    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)

    <ipython-input-3-6463279979e9> in <module>()
          1 a = "1"
          2 b = 5
    ----> 3 print(a+b)
    

    TypeError: cannot concatenate 'str' and 'int' objects


.. code:: python

    a = "1"
    b = 5
    print(int(a)+b)

Note that the typing is dynamic. I.e. a variable that was initally say
an integer can become another type (float, string, etc.) via
reassignment.

.. code:: python

    a = "1"
    type(a)
    print(type(a))
    
    a = 1.0
    print(type(a))

Python has some other special data types such as lists, tuples and
dictionaries that we will address later.

Operators
---------

| Python offers the usual operators such as +,-,/,\*,=,>,<,==,!=,&,\|,
(sum, difference, divide, product, assignment, greater than, less than,
equal - comparison,not equal, and, or, respectively).
| Additionally, there are %,// and \*\* (modulo, floor division and 'to
the power'). Note a few specifics:

.. code:: python

    print(3/4)
    print(3.0 / 4.0)
    print(3%4)
    print(3//4)
    print(3**4)

Note the behavior of / when applied to integers! This is similar to the
behavior of other strongly typed languages such as C/C++. The result of
the integer division is the same as the floor division //. If you want
the floating point result, the arguments to / must be floats as well (or
appropriately typecast).

.. code:: python

    a = 3
    b = 4
    print(a/b)
    print(float(a)/float(b))

Iterators
---------

Python has the usual iterators, while, for, and some other constructions
that will be addressed later. Here are examples of each:

.. code:: python

    for i in range(1,10):
         print(i)

The most important thing to note above is that the range function gives
us values up to, but not including, the upper limit.

.. code:: python

    i = 1
    while i < 10:
        print(i)
        i+=1

This is unremarkable, so we proceeed without further comment.

Conditional Statements
----------------------

.. code:: python

    a = 20
    if a >= 22:
       print("if")
    elif a >= 21:
        print("elif")
    else:
        print("else")

Again, nothing remarkable here, just need to learn the syntax. Here, we
should also mention spacing. Python is picky about indentation - you
must start a newline after each conditional statemen (it is the same for
the iterators above) and indent the same number of spaces for every
statement within the scope of that condition.

.. code:: python

    a = 23
    if a >= 22:
       print("if")
        print("greater than or equal 22")
    elif a >= 21:
        print("elif")
    else:
        print("else")

.. code:: python

    a = 23
    if a >= 22:
       print("if")
       print("greater than or equal 22")
    elif a >= 21:
        print("elif")
    else:
        print("else")

Four spaces are customary, but you can use whatever you like.
Consistency is necessary.

Exceptions
^^^^^^^^^^

Python has another type of conditional expression that is very useful.
Suppose your program is processing user input or data from a file. You
don't always know for sure what you are getting in that case, and this
can lead to problems. The 'try/except' conditional can solve them!

.. code:: python

    a = "1"
    
    try:
      b = a + 2 
    except:
      print(a, " is not a number") 


Here, we have tried to add a number and a string. That generates an
exception - but we have trapped the exception and informed the user of
the problem. This is much preferable to the programming crashing with
some cryptic error like:

.. code:: python

    a = "1"
    b = a + 2 


Functions
---------

.. code:: python

    def Division(a, b):
        print(a/b)
    Division(3,4)
    Division(3.0,4.0)
    Division(3,4.0)
    Division(3.0,4)

Notice that the function does not specify the types of the arguments,
like you would see in statically typed languages. This is both useful
and dangerous. For example:

.. code:: python

    def Division(a, b):
        print(a/b)
    Division(2,"2")

In a statically typed language, the programmer would have specified the
type of a and b (float, int, etc.) and the compiler would have
complained about the function being passed a variable of the wrong type.
This does not happen here, but we can use the try/except construction.

.. code:: python

    def Division(a, b):
        try:
            print(a/b)
        except:
            if b == 0:
               print("cannot divide by zero")
            else:
               print(float(a)/float(b))
    Division(2,"2")
    Division(2,0)

Strings and String Handling
---------------------------

One of the most important features of Python is its powerful and easy
handling of strings. Defining strings is simple enough in most
languages. But in Python, it is easy to search and replace, convert
cases, concatenate, or access elements. We'll discuss a few of these
here. For a complete list, see:
http://www.tutorialspoint.com/python/python\_strings.htm

.. code:: python

    a = "A string of characters, with newline \n CAPITALS, etc."
    print(a)
    b=5.0
    newstring = a + "\n We can format strings for printing %.2f"
    print(newstring %b)


Now let's try some other string operations:

.. code:: python

    a = "ABC DEFG"
    print(a[1:3])
    print(a[0:5])

There are several things to learn from the above. First, Python has
associated an index to the string. Second the indexing starts at 0, and
lastly, the upper limit again means 'up to but not including' (a[0:5]
prints elements 0,1,2,3,4).

.. code:: python

    a = "ABC defg"
    print(a.lower())
    print(a.upper())
    print(a.find('d'))
    print(a.replace('de','a'))
    print(a)
    b = a.replace('def','aaa')
    print(b)
    b = b.replace('a','c')
    print(b)
    b.count('c')


This is fun! What else can you do with strings in Python? Pretty much
anything you can think of!

Lists, Tuples, Dictionaries
---------------------------

Lists
~~~~~

Lists are exactly as the name implies. They are lists of objects. The
objects can be any data type (including lists), and it is allowed to mix
data types. In this way they are much more flexible than arrays. It is
possible to append, delete, insert and count elements and to sort,
reverse, etc. the list.

.. code:: python

    a_list = [1,2,3,"this is a string",5.3]
    b_list = ["A","B","F","G","d","x","c",a_list,3]
    print(b_list)


.. code:: python

    print(b_list[7:9])

.. code:: python

    a = [1,2,3,4,5,6,7]
    a.insert(0,0)
    print(a)
    a.append(8)
    print(a)
    a.reverse()
    print(a)
    a.sort()
    print(a)
    a.pop()
    print(a)
    a.remove(3)
    print(a)
    a.remove(a[4])
    print(a)

Just like with strings, elements are indexed beginning with 0.

Lists can be constructed using 'for' and some conditional statements.
These are called, 'list comprehensions'. For example:

.. code:: python

    even_numbers = [x for x in range(100) if x % 2 == 0]
    print(even_numbers)

List comprehensions can work on strings as well:

.. code:: python

    first_sentence = "It was a dark and stormy night."
    characters = [x for x in first_sentence]
    print(characters)

For more on comprehensions see:
https://docs.python.org/2/tutorial/datastructures.html?highlight=comprehensions

Another similar feature is called 'map'. Map applies a function to a
list. The syntax is

map(aFunction, aSequence). Consider the following examples:

.. code:: python

    def sqr(x): return x ** 2
    a = [2,3,4]
    b = [10,5,3]
    c = map(sqr,a)
    print(c)
    d = map(pow,a,b)
    print(d)

Note that map is usually more efficient than the equivalent list
comprehension or looping contruct.

Tuples
~~~~~~

Tuples are like lists with one very important difference. Tuples are not
changeable.

.. code:: python

    a = (1,2,3,4)
    print(a)
    a[1] = 2

.. code:: python

    a = (1,"string in a tuple",5.3)
    b = (a,1,2,3)
    print(a)
    print(b)


As you can see, all of the other flexibility remains - so use tuples
when you have a list that you do not want to modify.

One other handy feature of tuples is known as 'tuple unpacking'.
Essentially, this means we can assign the values of a tuple to a list of
variable names, like so:

.. code:: python

    my_pets = ("Chestnut", "Tibbs", "Dash", "Bast")
    (aussie,b_collie,indoor_cat,outdoor_cat) = my_pets
    print(aussie)
    cats=(indoor_cat,outdoor_cat)
    print(cats)


Dictionaries
~~~~~~~~~~~~

Dictionaries are unordered, keyed lists. Lists are ordered, and the
index may be viewed as a key.

.. code:: python

    a = ["A","B","C","D"] #list example
    print(a[1])


.. code:: python

    a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"} # dictionary example
    print(a[1])

.. code:: python

    a = {'anItem': "A", 'anotherItem': "B",'athirdItem':"C",'afourthItem':"D"} # dictionary example
    print(a['anItem'])


.. code:: python

    print(a)

The dictionary does not order the items, and you cannot access them assuming an order (as an index does).  You access elements using the keys.

Sets
~~~~

Sets are unordered collections of *unique* elements. Intersections,
unions and set differences are supported operations. They can be used to
remove duplicates from a collection or to test for membership. For
example:

.. code:: python

    from sets import Set
    fruits = Set(["apples","oranges","grapes","bananas"])
    citrus = Set(["lemons","oranges","limes","grapefruits","clementines"])
    citrus_in_fruits = fruits & citrus   #intersection
    print(citrus_in_fruits)
    diff_fruits = fruits - citrus        # set difference
    print(diff_fruits)
    diff_fruits_reverse = citrus - fruits  # set difference
    print(diff_fruits_reverse)
    citrus_or_fruits = citrus | fruits     # set union
    print(citrus_or_fruits)

.. code:: python

    a_list = ["a", "a","a", "b",1,2,3,"d",1]
    print(a_list)
    a_set = Set(a_list)  # Convert list to set
    print(a_set)         # Creates a set with unique elements
    new_list = list(a_set) # Convert set to list
    print(new_list)        # Obtain a list with unique elements 

More examples and details regarding sets can be found at:
https://docs.python.org/2/library/sets.html

Classes
-------

A class (or object) bundles data (known as attributes) and functions
(known as methods) together. We access the attributes and methods of a
class using the '.' notation. Since everything in Python is an object,
we have already been using this attribute acccess - e.g. when we call
``'hello'.upper()``, we are using the ``upper`` method of the instance
``'hello'`` of the ``string`` class.

The creation of custom classes will not be covered in this course.

Modules
-------

As the code base gets larger, it is convenient to organize them as
*modules* or packages. At the simplest level, modules can just be
regular python files. We import functions in modules using one of the
following ``import`` variants:

.. code:: python

    import numpy
    import numpy as np # using an alias
    import numpy.linalg as la # modules can have submodules
    from numpy import sin, cos, tan # bring trig functions into global namespace
    from numpy import * # frowned upon because it pollutes the namespace

The standard library
--------------------

Python comes with "batteries included", with a diverse collection of
functionality available in standard library modules and functions.

**References**

-  `Standard library docs <https://docs.python.org/2/library/>`__
-  `Python Module of the Week <http://pymotw.com/2/contents.html>`__
   gives examples of usage.

Installing additional modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the time, we can use the ``pip`` package manager to install and
uninstall modules for us. In general, all that is needed is to issue the
command

.. code:: bash

    pip install <packagename>

at the command line or

.. code:: python

    ! pip install <packagename>

from within an IPython notebook.

Packages that can be installed using ``pip`` are listed in the `Python
Package Index (PyPI) <https://pypi.python.org/pypi>`__.

Pip documentation is at https://pip.pypa.io/en/latest/.

Keeping the Anaconda distribution up-to-date
--------------------------------------------

Just issue

.. code:: bash

    conda update conda
    conda update anaconda

at the command line.

Note that ``conda`` can do `much, much,
more <http://conda.pydata.org/docs/index.html>`__.

Exercises
---------

**1**. Solve the FizzBuzz probelm

"Write a program that prints the numbers from 1 to 100. But for
multiples of three print “Fizz” instead of the number and for the
multiples of five print “Buzz”. For numbers which are multiples of both
three and five print “FizzBuzz”.

.. code:: python

    # YOUR CODE HERE
    
    # range(start, stop, step)
    # for loop
    # print function
    # % operator
    # check for equality
    # if-elif-else control flow
    
    for i in range(1, 101):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


.. parsed-literal::

    1
    2
    Fizz
    4
    Buzz
    Fizz
    7
    8
    Fizz
    Buzz
    11
    Fizz
    13
    14
    FizzBuzz
    16
    17
    Fizz
    19
    Buzz
    Fizz
    22
    23
    Fizz
    Buzz
    26
    Fizz
    28
    29
    FizzBuzz
    31
    32
    Fizz
    34
    Buzz
    Fizz
    37
    38
    Fizz
    Buzz
    41
    Fizz
    43
    44
    FizzBuzz
    46
    47
    Fizz
    49
    Buzz
    Fizz
    52
    53
    Fizz
    Buzz
    56
    Fizz
    58
    59
    FizzBuzz
    61
    62
    Fizz
    64
    Buzz
    Fizz
    67
    68
    Fizz
    Buzz
    71
    Fizz
    73
    74
    FizzBuzz
    76
    77
    Fizz
    79
    Buzz
    Fizz
    82
    83
    Fizz
    Buzz
    86
    Fizz
    88
    89
    FizzBuzz
    91
    92
    Fizz
    94
    Buzz
    Fizz
    97
    98
    Fizz
    Buzz


**2**. Given x=3 and y=4, swap the values of x and y so that x=4 and
y=3.

.. code:: python

    x = 3
    y = 4
    # YOUR CODE HERE
    
    # use of temporary variable
    # tuple unpacking
    
    tmp = x
    x = y
    y = x
    print x, y
    
    x = 3
    y = 4
    x, y = y, x
    print x, y


.. parsed-literal::

    4 4
    4 3


**3**. Write a function that calculates and returns the euclidean
distance between two points :math:`u` and :math:`v`, where :math:`u` and
:math:`v` are both 2-tuples :math:`(x, y)`. For example, if
:math:`u = (3,0)` and :math:`v = (0,4)`, the function should return
:math:`5`.

.. code:: python

    # YOUR CODE HERE
    
    # euclidean distance formula
    # operators **
    # square root function
    # anatomy of a function
    
    u = (3, 0)
    v = (0, 4)
    
    ((v[0] - u[0])**2 + (v[1] - u[1])**2)**0.5
    
    def euclidean(u, v):
        """Returns the Euclidean distance between points u and v."""
        return ((v[0] - u[0])**2 + (v[1] - u[1])**2)**0.5
    
    euclidean(u, v)




.. parsed-literal::

    5.0



**4**. Using a dictionary, write a program to calculate the number times
each character occurs in the given string s. Ignore differneces in
capitalization - i.e 'a' and 'A' should be treated as a single key. For
example, we should get a count of 7 for 'a'.

.. code:: python

    s = """
    Write a program that prints the numbers from 1 to 100. 
    But for multiples of three print 'Fizz' instead of the number and f
    or the multiples of five print 'Buzz'. For numbers which are 
    multiples of both three and five print 'FizzBuzz'
    """
    
    # YOUR CODE HERE
    
    # string methods
    # dictionary
    # for loop
    # collections.Counter
    
    # Version 1
    print s.lower().count('a')
    
    # Version 2
    counter1 = {}
    for _ in s.lower():
        counter1[_] = counter1.get(_, 0) + 1
    print counter1['a']
    
    # Version 3
    from collections import defaultdict
    counter2 = defaultdict(int)
    for _ in s.lower():
        counter2[_] += 1
    print counter2['a']
    
    # Version 4
    from collections import Counter
    counter3 = Counter(s.lower())
    print counter3['a']


.. parsed-literal::

    7
    7
    7
    7


**5**. Write a program that finds the percentage of sliding windows of
length 5 for the sentence s that contain at least one 'a'. Ignore case,
spaces and punctuation. For example, the first sliding window is 'write'
which contains 0 'a's, and the second is 'ritea' which contains 1 'a'.

.. code:: python

    s = """
    Write a program that prints the numbers from 1 to 100. 
    But for multiples of three print 'Fizz' instead of the number and f
    or the multiples of five print 'Buzz'. For numbers which are 
    multiples of both three and five print 'FizzBuzz'
    """
    
    # YOUR CODE HERE
    
    # string constants
    # translate method
    # replace method
    # slicing iterables
    # len function
    
    import string
    s1 = s.lower().translate(None, string.punctuation).replace(' ', '').replace('\n', '')
    
    count = 0
    start = 0
    stop = 5
    
    while (stop <= len(s1)):
        # print s1[start:stop]
        if 'a' in s1[start:stop]:
            count += 1
        start += 1
        stop += 1
    
    print count


.. parsed-literal::

    34


**6**. Find the unique numbers in the following list.

.. code:: python

    x = [36, 45, 58, 3, 74, 96, 64, 45, 31, 10, 24, 19, 33, 86, 99, 18, 63, 70, 85,
     85, 63, 47, 56, 42, 70, 84, 88, 55, 20, 54, 8, 56, 51, 79, 81, 57, 37, 91,
     1, 84, 84, 36, 66, 9, 89, 50, 42, 91, 50, 95, 90, 98, 39, 16, 82, 31, 92, 41,
     45, 30, 66, 70, 34, 85, 94, 5, 3, 36, 72, 91, 84, 34, 87, 75, 53, 51, 20, 89, 51, 20]
    
    # YOUR CODE HERE
    
    # sort and remove duplicates
    # negative indexing
    
    # version 1
    sorted_x = sorted(x)
    unique_x = [sx[0]]
    for _ in sorted_x[1:]:
        if _ != unique_x[-1]:
            unique_x.append(_)
    
    print unique_x
    print len(x)
    print len(unique_x)
    
    # using set
    print list(set(x))
    print len(x)
    print len(set(x))


.. parsed-literal::

    [1, 3, 5, 8, 9, 10, 16, 18, 19, 20, 24, 30, 31, 33, 34, 36, 37, 39, 41, 42, 45, 47, 50, 51, 53, 54, 55, 56, 57, 58, 63, 64, 66, 70, 72, 74, 75, 79, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 98, 99]
    80
    54
    [1, 3, 5, 8, 9, 10, 16, 18, 19, 20, 24, 30, 31, 33, 34, 36, 37, 39, 41, 42, 45, 47, 50, 51, 53, 54, 55, 56, 57, 58, 63, 64, 66, 70, 72, 74, 75, 79, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 98, 99]
    80
    54


**7**. Write two functions - one that returns the square of a number,
and one that returns the cube. Now write a third function that returns
the number raised to the :math:`6^{th}` power using the two previous
functions.

.. code:: python

    # YOUR CODE HERE
    
    # getting comforatble with functions
    # unit tests
    
    def square(x):
        """Returns x^2."""
        return x**2
    
    def cube(x):
        """Returns x^3."""
        return x**3
    
    def pow6(x):
        """Returns x^6."""
        return cube(square(x))
    
    # use of assert for testing
    def test_pow6(x):
        assert(abs(pow6(x) - x**6) < 1e-6)
    
    xs = [-2, 0, 1.5]
    for x in xs:
        test_pow6(x)

**8**. Create a list of the cubes of x for x in [0, 10] using

-  a for loop
-  a list comprehension
-  the map function

.. code:: python

    # YOUR CODE HERE
    
    # list comprehensions
    # map
    # lambda functions
    
    cubes1 = []
    for i in range(1, 11):
        cubes1.append(i**3)
    print cubes1
    
    cubes2 = [i**3 for i in range(1, 11)]
    print cubes2
    
    print map(lambda x: x**3, range(1, 11))


.. parsed-literal::

    [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
    [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
    [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]


**9**. A Pythagorean triple is an integer solution to the Pythagorean
theorem :math:`a^2 + b^2 = c^2`. The first Pythagorean triple is
(3,4,5). Find all unique Pythagorean triples for the positive integers
a, b and c less than 100.

.. code:: python

    # YOUR CODE HERE
    
    # nested list comprehsnions
    # inner and outer loops
    
    print([(i, j) for i in range(1,4) for j in range(10, 14)])
    print 
    
    pythagorean_triples = [(a, b, c) for a in range(1, 100) 
                                     for b in range(1, 100) 
                                     for c in range(1, 100)
                                     if a**2 + b**2 == c**2]
    print pythagorean_triples
    print
    
    pythagorean_triples = [(a, b, c) for a in range(1, 100) 
                                     for b in range(a, 100) 
                                     for c in range(b, 100)
                                     if a**2 + b**2 == c**2]
    print pythagorean_triples


.. parsed-literal::

    [(1, 10), (1, 11), (1, 12), (1, 13), (2, 10), (2, 11), (2, 12), (2, 13), (3, 10), (3, 11), (3, 12), (3, 13)]
    
    [(3, 4, 5), (4, 3, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25), (8, 6, 10), (8, 15, 17), (9, 12, 15), (9, 40, 41), (10, 24, 26), (11, 60, 61), (12, 5, 13), (12, 9, 15), (12, 16, 20), (12, 35, 37), (13, 84, 85), (14, 48, 50), (15, 8, 17), (15, 20, 25), (15, 36, 39), (16, 12, 20), (16, 30, 34), (16, 63, 65), (18, 24, 30), (18, 80, 82), (20, 15, 25), (20, 21, 29), (20, 48, 52), (21, 20, 29), (21, 28, 35), (21, 72, 75), (24, 7, 25), (24, 10, 26), (24, 18, 30), (24, 32, 40), (24, 45, 51), (24, 70, 74), (25, 60, 65), (27, 36, 45), (28, 21, 35), (28, 45, 53), (30, 16, 34), (30, 40, 50), (30, 72, 78), (32, 24, 40), (32, 60, 68), (33, 44, 55), (33, 56, 65), (35, 12, 37), (35, 84, 91), (36, 15, 39), (36, 27, 45), (36, 48, 60), (36, 77, 85), (39, 52, 65), (39, 80, 89), (40, 9, 41), (40, 30, 50), (40, 42, 58), (40, 75, 85), (42, 40, 58), (42, 56, 70), (44, 33, 55), (45, 24, 51), (45, 28, 53), (45, 60, 75), (48, 14, 50), (48, 20, 52), (48, 36, 60), (48, 55, 73), (48, 64, 80), (51, 68, 85), (52, 39, 65), (54, 72, 90), (55, 48, 73), (56, 33, 65), (56, 42, 70), (57, 76, 95), (60, 11, 61), (60, 25, 65), (60, 32, 68), (60, 45, 75), (60, 63, 87), (63, 16, 65), (63, 60, 87), (64, 48, 80), (65, 72, 97), (68, 51, 85), (70, 24, 74), (72, 21, 75), (72, 30, 78), (72, 54, 90), (72, 65, 97), (75, 40, 85), (76, 57, 95), (77, 36, 85), (80, 18, 82), (80, 39, 89), (84, 13, 85), (84, 35, 91)]
    
    [(3, 4, 5), (5, 12, 13), (6, 8, 10), (7, 24, 25), (8, 15, 17), (9, 12, 15), (9, 40, 41), (10, 24, 26), (11, 60, 61), (12, 16, 20), (12, 35, 37), (13, 84, 85), (14, 48, 50), (15, 20, 25), (15, 36, 39), (16, 30, 34), (16, 63, 65), (18, 24, 30), (18, 80, 82), (20, 21, 29), (20, 48, 52), (21, 28, 35), (21, 72, 75), (24, 32, 40), (24, 45, 51), (24, 70, 74), (25, 60, 65), (27, 36, 45), (28, 45, 53), (30, 40, 50), (30, 72, 78), (32, 60, 68), (33, 44, 55), (33, 56, 65), (35, 84, 91), (36, 48, 60), (36, 77, 85), (39, 52, 65), (39, 80, 89), (40, 42, 58), (40, 75, 85), (42, 56, 70), (45, 60, 75), (48, 55, 73), (48, 64, 80), (51, 68, 85), (54, 72, 90), (57, 76, 95), (60, 63, 87), (65, 72, 97)]


**10**. Fix the bug in this function that is intended to take a list of
numbers and return a list of normalized numbers.

.. code:: python

    def f(xs):
        """Return normalized list summing to 1."""
        s = 0
        for x in xs:
            s += x
        return [x/s for x in xs]

.. code:: python

    # YOUR CODE HERE
    
    # elementary debugging
    
    def f(xs):
        """Return normalized list summing to 1."""
        s = 0
        for x in xs:
            s += x
        return [x/s for x in xs]
    
    xs = [1.1,2.2,3.3,4.4]
    print f(xs)
    
    xs = [1,2,3,4]
    print f(xs)
    
    
    def f(xs):
        """Return normalized list summing to 1."""
        s = 0.0
        for x in xs:
            s += x
        return [x/s for x in xs]
    
    
    xs = [1.1,2.2,3.3,4.4]
    print f(xs)
    
    xs = [1,2,3,4]
    print f(xs)


.. parsed-literal::

    [0.1, 0.2, 0.3, 0.4]
    [0, 0, 0, 0]
    [0.1, 0.2, 0.3, 0.4]
    [0.1, 0.2, 0.3, 0.4]


