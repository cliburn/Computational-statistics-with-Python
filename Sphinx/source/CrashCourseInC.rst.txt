
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

    from IPython.display import Image

C Crash Course
==============

C functions are typically split into header files (``.h``) where things
are declared but not defined, and implementation files (``.c``) where
they are defined. When we run the C compiler, a complex sequence of
events is triggered with the usual successful outcome begin an
executable file as illuatrated at http://www.codingunit.com/

.. figure:: http://www.codingunit.com/images/preprocessor-compiler-linker.jpg
   :alt: Compilation process

   Compilation process

The preprocessor merges the contents of the header and implementation
files, and also expands any macros. The compiler then translates these
into low level object code (``.o``) for each file, and the linker then
joins together the newly generated object code with pre-compiled object
code from libraries to form an executable. Sometimes we just want to
generate object code and save it as a library (e.g. so that we can use
it in Python).

Hello world
-----------

.. code:: python

    %%file hello.c
    #include <stdio.h>
    
    int main() {
        printf("Hello, world!");
    }

.. code:: python

    ! gcc hello.c -o hello

.. code:: python

    ! ./hello

A tutorial example - coding a Fibonacci function in C
-----------------------------------------------------

Python version
^^^^^^^^^^^^^^

.. code:: python

    def fib(n):
        a, b = 0,  1
        for i in range(n):
            a, b = a+b, a
        return a

.. code:: python

    fib(100)

C version
^^^^^^^^^

Header file
'''''''''''

.. code:: python

    %%file fib.h
    
    double fib(int n);

Implemetnation file
'''''''''''''''''''

.. code:: python

    %%file fib.c
    
    double fib(int n) {
        double a = 0, b = 1;
        for (int i=0; i<n; i++) {
            double tmp = b;
            b = a;
            a += tmp;
         }
        return a;
    }

Driver program
''''''''''''''

.. code:: python

    %%file main.c
    #include <stdio.h> // for printf()
    #include <stdlib.h> // for atoi())
    #include "fib.h" // for fib()
    
    int main(int argc, char* argv[]) {
        int n = atoi(argv[1]);
        printf("%f", fib(n));
    }


Makefile
''''''''

.. code:: python

    %%file Makefile
    
    CC=clang
    CFLAGS=-Wall
    
    fib: main.o fib.o
    	 $(CC) $(CFLAGS) -o fib main.o fib.o
    
    main.o: main.c fib.h
    	 $(CC) $(CFAGS) -c main.c
    
    fib.o: fib.c
    	 $(CC) $(CFLAGS) -c fib.c
    
    clean:
    	 rm -f *.o

Compile
'''''''

.. code:: python

    ! make 

Run executable file
^^^^^^^^^^^^^^^^^^^

.. code:: python

    ! ./fib 100

Types in C
----------

The basic types are very simple - use int, float and double for numbers.
In genneral, avoid float for plain C code as its lack of precision may
bite you unless you are writing CUDA code. Strings are quite nasty to
use in C - I would suggest doing all your string processing in Python
...

Structs are sort of like classes in Python

.. code:: c

    struct point {
        double x;
        double y;
        double z;
    };

    struct point p1 = {.x = 1, .y = 2, .z = 3};
    struct point p2 = {1, 2, 3};
    struct point p3;
    p3.x = 1;
    p3.y = 2;
    p3.z = 3;

You can define your own types using ``typedef`` -.e.g.

.. code:: c

    #include <stdio.h>
    struct point {
        double x;
        double y;
        double z;
    };

    typedef struct point point;

    int main() {
        point p = {1, 2, 3};
        printf("%.2f, %.2f, %.2f", p.x, p.y, p.z);
    };

Operators
---------

Most of the operators in C are the same in Python, but an important
difference is the increment/decrement operator. That is

.. code:: c

    int c = 10;
    c++; // same as c = c + 1, i.e., c is now 11
    c--; // same as c = c - 1, i.e.. c is now 10 again

There are two forms of the incremanet operator - postfix ``c++`` and
prefix ``++c``. Both increemnt the varible, but in an expressino, the
postfix veersion returns the value before the increment and the prefix
returns the value after the increment.

.. code:: python

    %%file increment.c
    #include <stdio.h>
    #include <stdlib.h>
    
    int main()
    {
        int x = 3, y;
        y = x++; // x is incremented and y takes the value of x before incrementation
        printf("x = %d, y = %d\n", x, y); 
        y = ++x; // x is incremented and y takes the value of x after incrementation
        printf("x = %d, y = %d\n", x, y); 
    }

.. code:: python

    %%bash
    
    clang -Wall increment.c -o increment
    ./increment

Ternary operator
^^^^^^^^^^^^^^^^

The ternary operator ``expr = condition ? expr1 : expr2`` allows an
if-else statement to be put in a single line. In English, this says that
if condition is True, expr1 is assigned to expr, otherwise expr2 is
assigned to expr. We used it in the tutorial code to print a comma
between elements in a list unless the elememnt was the last one, in
which case we printed a new line ''.

Note: There is a similar ternary construct in Python
``expr = expr1 if condition else epxr2``.

Control of program flow
-----------------------

Very similar to Python or R. The examples below should be
self-explanatory.

if-else
^^^^^^^

.. code:: c

    // Interpretation of grades by Asian parent
    if (grade == 'A') {
        printf("Acceptable\n");
    } else if (grade == 'B') {
        printf("Bad\n");
    } else if (grade == 'C') {
        printf("Catastrophe\n");
    } else if (grade == 'D') {
        printf("Disowned\n");
    } else {
        printf("Missing child report filed with local police\n")
    }

for, while, do
^^^^^^^^^^^^^^

.. code:: c

    // Looping variants

    // the for loop in C consists of the keyword for followed by
    // (initializing statement; loop condition statement; loop update statement)
    // followed by the body of the loop in curly braces
    int arr[3] = {1, 2, 3};
    for (int i=0; i<sizeof(arr)/sizeof(arr[0]); i++) {
        printf("%d\n", i);
    }

    // the while loop
    int i = 3;
    while (i > 0) {
        i--;
    }

    // the do loop is similar to the while loop but will execute the body at least once
    int i = 3;
    do {
        i==;
    } while (i > 0);

The C standard does not require braces if the body is a singel line, but
I think it is safer to always include them. Note that whitespace is not
significant in C (unlike Python), so

.. code:: c

    int i = 10;
    while (i > 0)
        i--;
        i++;

actually means

::

    int i = 10;
    while (i > 0) {
        i--;
    }
    i++;

and the use of braces even for single statement bodies prevnets such
errors.

Arrays and pointers
-------------------

Automatic arrays
^^^^^^^^^^^^^^^^

If you know the size of the arrays at initialization (i.e. when the
program is first run), you can usually get away with the use of fixed
size arrays for which C will automatically manage memory for you.

.. code:: c

    int len = 3;

    // Giving an explicit size
    double xs[len];
    for (int i=0; i<len; i++) {
        xs[i] = 0.0;
    }

    // C can infer size if initializer is given
    double ys[] = {1, 2, 3};

Pointers and dynamic memory management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Otherwise, we have to manage memory ourselves using pointers. Bascially,
memory in C can be auotmatic, static or dynamic. Variables in automatic
memory are managed by the computer on the *stack*, when it goes out of
*scope*, the varible disappears. Static variables essentially live
forever. Dynamic memory is allocated in the *heap*, and you manage its
lifetime.

Mini-glossary: \* **scope**: Where a variable is visible - basically C
variables have *block* scope - variables either live within a pair of
curly braces (inlucdes variables in parentheses just before block such
as function arguments and the counter in a for loop), or they are
visible thorughout the file. \* **stack**: Computer memory is divided
into a stack (small) and a heap (big). Automatic varianbles are put on
the stack; dynamcic variables are put in the heap. Hence if you have a
very large array, you would use dynamic memory allocation even if you
knwe its size at initialization.

Any variable in memory has an address represented as a 64-bit integer in
most operating systems. A pointer is basically an integer containing the
address of a block of memory. This is what is returned by functions such
as ``malloc``. In C, a pointer is dentoed by ``*``. However, the ``*``
notation is confusing because its interpreation depends on whehter you
are using it in a declaraiton or not. In a declaration

.. code:: c

    int *p = malloc(sizeof(int)); // p is a pointer to an integer
    *p = 5; // *p is an integer

To get the actual address value, we can use the ``&`` address opertor.
This is often used so that a function can alter the value of an argument
passed in (e.g. see address.c below).

.. code:: python

    %%file pointers.c
    #include <stdio.h>
    
    int main()
    {
        int i = 2;
        int j = 3;
        int *p;
        int *q;
        *p = i;
        q = &j;
        printf("p  = %p\n", p);
        printf("*p = %d\n", *p);
        printf("&p = %p\n", &p);
        printf("q  = %p\n", q);
        printf("*q = %d\n", *q);
        printf("&q = %p\n", &q);
    }

.. code:: python

    %%bash
    
    clang -Wall -Wno-uninitialized pointers.c -o pointers
    ./pointers

Passing by value and passing by reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    %%file by_val.c
    #include <stdio.h>
    
    void change_arg(int p) {
        p *= 2;
    }
    
    int main()
    {
        int x = 5;
        change_arg(x);
        printf("%d\n", x);
    }


.. code:: python

    %%bash
    
    clang -Wall by_val.c -o by_val
    ./by_val

.. code:: python

    %%file by_ref.c
    #include <stdio.h>
    
    void change_arg(int *p) {
        *p *= 2;
    }
    int main()
    {
        int x = 5;
        change_arg(&x);
        printf("%d\n", x);
    }

.. code:: python

    %%bash
    
    clang -Wall by_ref.c -o by_ref
    ./by_ref

Pointers to pointers to pointers - just remember that a pointer is simply a name for an integer that represents an address; since it is an integer, it also has an address ...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    %%file ptr.c
    #include <stdio.h>
    
    int main() {
        int x = 2;
        int *p = &x;
        int **q = &p;
        int ***r = &q;
    
        printf("%d, %p, %p, %p, %p, %p, %p, %d", x, &x, p, &p, q, &q, r, ***r);
    }

.. code:: python

    %%bash
    gcc ptr.c -o ptr
    ./ptr

Pointer arithmetic
^^^^^^^^^^^^^^^^^^

If we want to store a whole sequence of ints, we can do so by simply
allocating more memory:

.. code:: c

    int *ps = malloc(5 * sizeof(int)); // ps is a pointer to an integer
    for (int i=0; i<5; i++) {
        ps[i] = i;
    }

The computer will find enough space in the heap to store 5 consecutive
integers in a **contiguour** way. Since C arrays are all fo the same
type, this allows us to do **pointer arithmetic** - i.e. the pointer
``ps`` is the same as ``&ps[0]`` and ``ps + 2`` is the same as
``&ps[2]``. An example at this point is helpful.

.. code:: python

    %%file pointers2.c
    #include <stdio.h>
    #include <stdlib.h>
    
    int main()
    {
        int *ps = malloc(5 * sizeof(int));
        for (int i =0; i < 5; i++) {
            ps[i] = i + 10;
        }
    
        printf("%d, %d\n", *ps, ps[0]); // remmeber that *ptr is just a regular variable outside of a declaration, in this case, an int
        printf("%d, %d\n", *(ps+2), ps[2]); 
        printf("%d, %d\n", *(ps+4), *(&ps[4])); // * and & are inverses
    
        free(ps); // avoid memory leak
    }

.. code:: python

    %%bash
    
    clang -Wall pointers2.c -o pointers2
    ./pointers2

Pointers and arrays
^^^^^^^^^^^^^^^^^^^

An array name is actualy just a constant pointer to the address of the
beginning of the array. Hence, we can derferecne an array name just like
a pointer. We can also do pointer arithmetic with array names - this
leads to the following legal but weird syntax:

.. code:: c

    arr[i] = *(arr + i) = i[arr]

.. code:: python

    %%file array_pointer.c
    #include <stdio.h>
    
    int main()
    {
        int arr[] = {1, 2, 3};
        printf("%d\t%d\t%d\t%d\t%d\t%d\n", *arr, arr[0], 0[arr], *(arr + 2), arr[2], 2[arr]);
    }

.. code:: python

    %%bash
    
    clang -Wall array_pointer.c -o array_pointer
    ./array_pointer

Allocating memory for 2D arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    %%file array_2d.c
    #include <stdio.h>
    #include <stdlib.h>
     
    int main()
    {
        int r = 3, c = 4;
     
        // first allocate space for the pointers to all rows
        int **arr = malloc(r * sizeof(int *));
        // then allocate space for the number of columns in each row
        for (int i=0; i<r; i++) {
            arr[i] = malloc(c * sizeof(int));
        }
     
        // fill array with integer values
        for (int i = 0; i <  r; i++) {
            for (int j = 0; j < c; j++) {
                 arr[i][j] =i*r+j; 
            }
        }
     
        for (int i = 0; i <  r; i++) {
          for (int j = 0; j < c; j++) {
             printf("%d ", arr[i][j]);
            }
        }
        
        // every malloc should have a free to avoid memory leaks
        for (int i=0; i<r; i++) {
            free(arr[i]);
        }
        free(arr);
    }

.. code:: python

    %%bash
    
    gcc -Wall array_2d.c -o array_2d
    ./array_2d

More on pointers
^^^^^^^^^^^^^^^^

**Differnt kinds of nothing**: There is a special null pointer indicated
by the keyword NULL that points to nothing. It is typically used for
pointer comparisons, since NULL pointers are guaranteed to compare as
not equal to any other pointer (including another NULL). In paticular,
it is often used as a sentinel value to mark the end of a list. In
contrast a void pointer (void \*) points to a memory location whose type
is not decalred. It is used in C for generic operations - for example,
``malloc`` returns a void pointer. To totally confuse the beginning C
student, there is also the NUL keyword, which refers to the ``'\0'``
character used to terminate C strings. NUL and NULL are totally
differnet beasts.

**Deciphering pointer idioms**: A common C idiom that you should get
used to is ``*q++ = *p++`` where p and q are both pointers. In English,
this says

-  \*q = \*p (copy the variable pointed to by p into the variable
   pointed to by q)
-  increment q
-  increment p

.. code:: python

    %%file pointers3.c
    #include <stdio.h>
    #include <stdlib.h>
    
    int main()
    {
        // example 1
        typedef char* string;
        char *s[] = {"mary ", "had ", "a ", "little ", "lamb", NULL};
        for (char **sp = s; *sp != NULL; sp++) {
            printf("%s", *sp);
        }
        printf("\n");
    
        // example 2
        char *src = "abcde";
        char *dest = malloc(5); // char is always 1 byte by C99 definition
        
        char *p = src + 4;
        char *q = dest;
        while ((*q++ = *p--)); // put the string in src into dest in reverse order
    
        for (int i = 0; i < 5; i++) {
            printf("i = %d, src[i] = %c, dest[i] = %c\n", i, src[i], dest[i]);
        }
    }

.. code:: python

    %%bash
    
    clang -Wall pointers3.c -o pointers3
    ./pointers3

Functions
---------

.. code:: python

    %%file square.c
    #include <stdio.h>
    
    double square(double x)
    {
        return x * x;
    }
    
    int main()
    {
        double a = 3;
        printf("%f\n", square(a));
    }

.. code:: python

    %%bash
    
    clang -Wall square.c -o square
    ./square

Function pointers
-----------------

How to make a nice function pointer: Start with a regular function
declaration func, for example, here func is a function that takes a pair
of ints and returns an int

::

    int func(int, int);

To turn it to a function pointer, just add a ``*`` and wrap the funtion
name in parenthesis like so

::

    int (*func)(int, int);

Now ``func`` is a pointer to a funciton that takes a pair of ints and
returns an int. Finally, add a typedef so that we can use ``func`` as a
new type

::

    typedef int (*func)(int, int);

which allows us to create arrays of function pointers, higher order
functions etc as shown in the following example.

.. code:: python

    %%file square2.c
    #include <stdio.h>
    #include <math.h>
    
    // Create a function pointer type that takes a double and returns a double
    typedef double (*func)(double x);
    
    // A higher order function that takes just such a function pointer
    double apply(func f, double x)
    {
        return f(x);
    }
    
    double square(double x)
    {
        return x * x;
    }
    
    double cube(double x)
    {
        return pow(x, 3);
    }
    
    int main()
    {
        double a = 3;
        func fs[] = {square, cube, NULL};
    
        for (func *f=fs; *f; f++) {
            printf("%.1f\n", apply(*f, a));
        }   
    }

.. code:: python

    %%bash
    
    clang -Wall -lm square2.c -o square2
    ./square2

Using make to compile C programs
--------------------------------

As you have seen, the processs of C program compilation can be quite
messy, with all sorts of different compiler and linker flags to specify,
libraries to add and so on. For this reason, most C programs are
compiled using the ``make`` build tool that you are already familiar
with. Here is a simple generic makefile that you can customize to
compile your own programs adapted from the book 21st Centur C by Ben
Kelmens (O'Reilly Media).

-  **TARGET**: Typically the name of the execuatble
-  **OBJECTS**: The intemediate object files - typically there is one
   file.o for every file.c
-  **CFLAGS**: Compiler flags, e.g. -Wall (show all warnings), -g (add
   debug information), -O3 (use level 3 optimization). Also used to
   indicate paths to headers in non-standard locations, e.g.
   -I/opt/include
-  **LDFLAGS**: Linker flags, e.g. -lm (link agains the libmath
   library). Alos used to indicate pahts to libaries in non-standard
   locations, e.g. -L/opt/lib
-  **CC**: Compiler, e.g. gcc or clang or icc

In addition, there are traiditonal dummy flags \* **all**: Builds all
targets (for example, you may also have html and pdf targets that are
optional) \* **clean**: Remove intermediate and final products generated
by the makefile

.. code:: python

    %%file makefile
    TARGET = 
    OBJECTS = 
    CFLAGS = -g -Wall -O3 
    LDLIBS = 
    CC = c99 
    
    all: TARGET
        
    clean:
    	 rm $(TARGET) $(OBJECTS)
    
    $(TARGET): $(OBJECTS)

Just fill in the blanks with whatever is appropriate for your program.
Here is a simple example where the main file ``test_main.c`` uses a
function from ``stuff.c`` with declarations in ``stuff.h`` and also
depends on the libm C math library.

.. code:: python

    %%file stuff.h
    #include <stdio.h>
    #include <math.h>
    
    void do_stuff();

.. code:: python

    %%file stuff.c
    #include "stuff.h"
    
    void do_stuff() {
        printf("The square root of 2 is %.2f\n", sqrt(2));
    }

.. code:: python

    %%file test_make.c
    #include "stuff.h"
    
    int main()
    {
        do_stuff();
    }

.. code:: python

    %%file makefile
    TARGET = test_make
    OBJECTS = stuff.o
    CFLAGS = -g -Wall -O3 
    LDLIBS = -lm
    CC = clang
    
    all: $(TARGET)
        
    clean:
    	 rm $(TARGET) $(OBJECTS)
    
    $(TARGET): $(OBJECTS)

.. code:: python

    ! make

.. code:: python

    ! ./test_make

.. code:: python

    # Make is clever enough to recompile only what has been changed since the last time it was called
    ! make

.. code:: python

    ! make clean

.. code:: python

    ! make

Exercise
--------

Debugging programs (understanding compiler warnings and errors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try to fix the following buggy program.

.. code:: python

    %%file buggy.c
    
    # Create a function pointer type that takes a double and returns a double
    double *func(double x);
    
    # A higher order function that takes just such a function pointer
    double apply(func f, double x)
    {
        return f(x);
    }
    
    double square(double x)
    {
        return x * x;
    }
    
    double cube(double x)
    {
        return pow(3, x);
    }
    
    double mystery(double x)
    {
        double y = 10;
        if (x < 10)
            x = square(x);
        else
            x += y;
            x = cube(x);
        return x;
    }
    
    int main()
    {
        double a = 3;
        func fs[] = {square, cube, mystery, NULL}
    
        for (func *f=fs, f != NULL, f++) {
            printf("%d\n", apply(f, a));
        }   
    }

.. code:: python

    ! clang -g -Wall buggy.c -o buggy


Why not C?
~~~~~~~~~~

What other language has an annual Obfuscated Code Contest
http://www.ioccc.org/? In particular, the following features of C are
very conducive to writing unreadable code:

-  lax rules for identifiers (e.g. \_o, \_0, \_O, O are all valide
   identifiers)
-  chars are bytes and pointers are integers
-  pointer arithmetic means that ``array[index]`` is the same as
   ``*(array+index)`` whihc is the same as ``index[array]``!
-  lax formatting rules especially with respect to whitespace (or lack
   of it)
-  Use of the comma operator to combine multiple expressions together
   with the ?: operator
-  Recursive function calls - e.g. main calling main repeatedly is legal
   C

Here is one winning entry from the 2013 IOCCC
`entry <http://www.ioccc.org/2013/dlowe/hint.html>`__ that should warm
the heart of statisticians - it displays sparklines (invented by Tufte).

.. code:: c

    main(a,b)char**b;{int c=1,d=c,e=a-d;for(;e;e--)_(e)<_(c)?c=e:_(e)>_(d)?d=e:7;
    while(++e<a)printf("\xe2\x96%c",129+(**b=8*(_(e)-_(c))/(_(d)-_(c))));}

.. code:: python

    %%file sparkl.c
    main(a,b)char**b;{int c=1,d=c,e=a-d;for(;e;e--)_(e)<_(c)?c=e:_(e)>_(d)?d=e:7;
    while(++e<a)printf("\xe2\x96%c",129+(**b=8*(_(e)-_(c))/(_(d)-_(c))));}

.. code:: python

    ! gcc -Wno-implicit-int -include stdio.h -include stdlib.h -D'_(x)=strtof(b[x],0)' sparkl.c -o sparkl

.. code:: python

    import numpy as np
    np.set_printoptions(linewidth=np.infty)
    print ' '.join(map(str, (100*np.sin(np.linspace(0, 8*np.pi, 30))).astype('int')))

.. code:: python

    %%bash
    
    ./sparkl 0 76 98 51 -31 -92 -88 -21 60 99 68 -10 -82 -96 -41 41 96 82 10 -68 -99 -60 21 88 92 31 -51 -98 -76 0

Learning Obfuscated C
~~~~~~~~~~~~~~~~~~~~~

If you have too much time on your hands and really want to know how
**not** to write C code (unless you are crafting an entry for the
IOCCC), I recommend this tutorial
http://www.dreamincode.net/forums/topic/38102-obfuscated-code-a-simple-introduction/

