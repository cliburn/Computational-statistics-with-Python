
.. code:: python

    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    %precision 4
    import os, sys, glob
    import regex as re
    import string

Preprocessing text data
=======================

Common applciations where there is a need to process text include:

1. Where the data *is* text - for example, if you are performing
   statistical analysis on the content of a billion web pages (perhaps
   you work for Google), or your research is in statistical natural
   language processing.
2. Where you have to preprocess messy real world dataa - e.g. column
   titles that are inconsistent in order to construct a DataFrame for
   analysis.

You may need to refer to the following:

-  For string constatns and some utilitiels, see the ``string`` module -
   e.g ``string.punctuation``, ``string.ascii_lowercase()``
-  For basic text processing, see string methods - e.g. ``lower()``,
   ``upper()``, ``split()``, ``replace()``, ``find()``, ``count()``
-  For regulear expression use, see the ``re`` module functions,
   especially ``compile()``, ``match()``, ``search()``, ``sub()``

As usual, make liberal use of IPython help (e.g ``string.punctuation?``)
to get information on a specific function or classs.

We will illustrate the use of string methods, regular expressions and
natural langauge parsing, as well as some Python built-in data
structures (e.g. Multiset (counter) and set) that can be used to clean
or analyze text data. This is meant only as an walk-thourgh of some of
the tools available; refer to the documentation for detals:

-  `The string
   module <https://docs.python.org/2/library/string.html>>`__
-  `String
   methods <https://docs.python.org/2/library/stdtypes.html#string-methods>`__
-  `The re module <https://docs.python.org/2/library/re.html>`__
-  `The regex module - new version of regular expression
   module <https://pypi.python.org/pypi/regex>`__
-  `Tutorial on regular
   expression <http://www.diveintopython.net/regular_expressions/>`__
-  `NLTK <http://www.nltk.org/>`__

Example: Counting words in a document
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perhaps the most basic thing we can do with textual data is to first
tokenize (spilt into words) the document, then count the number of times
each word (or pair of words, or ...) occurs. We will use the book (How
to be Happy Though Married) as an example (from Project Gutenberg).

.. code:: python

    import requests
    url = "http://www.gutenberg.org/cache/epub/35534/pg35534.txt"
    raw = requests.get(url).text

.. code:: python

    # peek at the first 1000 characters of the downloaded text
    raw[:1000]




.. parsed-literal::

    u'\ufeffProject Gutenberg\'s How to be Happy Though Married, by Edward John  Hardy\r\n\r\nThis eBook is for the use of anyone anywhere at no cost and with\r\nalmost no restrictions whatsoever.  You may copy it, give it away or\r\nre-use it under the terms of the Project Gutenberg License included\r\nwith this eBook or online at www.gutenberg.org\r\n\r\n\r\nTitle: How to be Happy Though Married\r\n       Being a Handbook to Marriage\r\n\r\nAuthor: Edward John  Hardy\r\n\r\nRelease Date: March 9, 2011 [EBook #35534]\r\n\r\nLanguage: English\r\n\r\n\r\n*** START OF THIS PROJECT GUTENBERG EBOOK HOW TO BE HAPPY THOUGH MARRIED ***\r\n\r\n\r\n\r\n\r\nProduced by Colin Bell, Christine P. Travers and the Online\r\nDistributed Proofreading Team at http://www.pgdp.net (This\r\nfile was produced from images generously made available\r\nby The Internet Archive)\r\n\r\n\r\n\r\n\r\n\r\n\r\n[Transcriber\'s note: The author\'s spelling has been maintained.\r\n\r\n+ signs around words indicate the use of a different font in the book.\r\n\r\nIn the word "Puranic", the "a" is overlined i'



Getting real cnntent
^^^^^^^^^^^^^^^^^^^^

The actual content of Project Guternberg books are delimited by the
phrases
``"*** START OF THIS PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE ***``
and ``End of the Project Gutenberg EBook`` respectively. Since the
actual book title will vary from book to book, we will use a regular
expression to search for
``*** START OF THIS PROJECT GUTENBERG EBOOK <STUFF> ***``. For the end
of the book, we can use a simple string search, but will use a regular
expression too for consistency. Note that we need the index of the last
character and the index of the first character respectively as limits to
extract only the text of the downloaded book.

.. code:: python

    start = re.search(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*", raw).end()
    stop = re.search(r"End of the Project Gutenberg EBook", raw).start()
    text = raw[start:stop]
    text[:1000]




.. parsed-literal::

    u'\r\n\r\n\r\n\r\n\r\nProduced by Colin Bell, Christine P. Travers and the Online\r\nDistributed Proofreading Team at http://www.pgdp.net (This\r\nfile was produced from images generously made available\r\nby The Internet Archive)\r\n\r\n\r\n\r\n\r\n\r\n\r\n[Transcriber\'s note: The author\'s spelling has been maintained.\r\n\r\n+ signs around words indicate the use of a different font in the book.\r\n\r\nIn the word "Puranic", the "a" is overlined in the book.]\r\n\r\n\r\n\r\n\r\n_HOW TO BE HAPPY THOUGH MARRIED._\r\n\r\n\r\n\r\n\r\nPRESS NOTICES ON THE FIRST EDITION.\r\n\r\n  "_If wholesome advice you can brook,\r\n    When single too long you have tarried;\r\n  If comfort you\'d gain from a book,\r\n    When very much wedded and harried;\r\n  No doubt you should speedily look,\r\n    In \'How to be Happy though Married!\'_"--PUNCH.\r\n\r\n\r\n"We strongly recommend this book as one of the best of wedding presents.\r\nIt is a complete handbook to an earthly Paradise, and its author may be\r\nregarded as the Murray of Matrimony and the Baedeker of Bliss."--_Pall\r\nMall Gaze'



Splitting into words - version using standard string methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # A naive but workable approach would be to first strip all punctuation, 
    # convert to lower case, then split on white space
    words1 = re.sub(ur"\p{P}+", "", text.lower()).split()
    print words1[:100]
    len(words1)


.. parsed-literal::

    [u'produced', u'by', u'colin', u'bell', u'christine', u'p', u'travers', u'and', u'the', u'online', u'distributed', u'proofreading', u'team', u'at', u'httpwwwpgdpnet', u'this', u'file', u'was', u'produced', u'from', u'images', u'generously', u'made', u'available', u'by', u'the', u'internet', u'archive', u'transcribers', u'note', u'the', u'authors', u'spelling', u'has', u'been', u'maintained', u'+', u'signs', u'around', u'words', u'indicate', u'the', u'use', u'of', u'a', u'different', u'font', u'in', u'the', u'book', u'in', u'the', u'word', u'puranic', u'the', u'a', u'is', u'overlined', u'in', u'the', u'book', u'how', u'to', u'be', u'happy', u'though', u'married', u'press', u'notices', u'on', u'the', u'first', u'edition', u'if', u'wholesome', u'advice', u'you', u'can', u'brook', u'when', u'single', u'too', u'long', u'you', u'have', u'tarried', u'if', u'comfort', u'youd', u'gain', u'from', u'a', u'book', u'when', u'very', u'much', u'wedded', u'and', u'harried', u'no']




.. parsed-literal::

    86545



Splitting into words - version using the NLTK (Natural Langauge Tool Kit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # If you need to be more careful, use the nltk tokenizer.
    import nltk
    from multiprocessing import Pool
    from itertools import chain
    punkt = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = punkt.tokenize(text.lower())
    # since the tokenizer works on a per sentence level, we can parallelize
    p = Pool()
    words2 = list(chain.from_iterable(p.map(nltk.tokenize.word_tokenize, sentences)))
    p.close()
    # Now remove words that consist of only punctuation characters
    words2 = [word for word in words2 if not all(char in string.punctuation for char in word)]
    # Remove contractions - wods that begin with '
    words2 = [word for word in words2 if not (word.startswith("'") and len(word) <=2)]
    print words2[:100]
    len(words2)


.. parsed-literal::

    [u'produced', u'by', u'colin', u'bell', u'christine', u'p.', u'travers', u'and', u'the', u'online', u'distributed', u'proofreading', u'team', u'at', u'http', u'//www.pgdp.net', u'this', u'file', u'was', u'produced', u'from', u'images', u'generously', u'made', u'available', u'by', u'the', u'internet', u'archive', u'transcriber', u'note', u'the', u'author', u'spelling', u'has', u'been', u'maintained', u'signs', u'around', u'words', u'indicate', u'the', u'use', u'of', u'a', u'different', u'font', u'in', u'the', u'book', u'in', u'the', u'word', u'puranic', u'the', u'a', u'is', u'overlined', u'in', u'the', u'book', u'_how', u'to', u'be', u'happy', u'though', u'married._', u'press', u'notices', u'on', u'the', u'first', u'edition', u'_if', u'wholesome', u'advice', u'you', u'can', u'brook', u'when', u'single', u'too', u'long', u'you', u'have', u'tarried', u'if', u'comfort', u'you', u'gain', u'from', u'a', u'book', u'when', u'very', u'much', u'wedded', u'and', u'harried', u'no']




.. parsed-literal::

    87158



Counting words
^^^^^^^^^^^^^^

.. code:: python

    from collections import Counter
    c = Counter(words2)
    c.most_common(n=10)




.. parsed-literal::

    [(u'the', 4356),
     (u'of', 3322),
     (u'and', 2699),
     (u'to', 2601),
     (u'a', 2335),
     (u'in', 1524),
     (u'is', 1209),
     (u'that', 1059),
     (u'it', 848),
     (u'be', 819)]



Ignoring stopwords
^^^^^^^^^^^^^^^^^^

.. code:: python

    # this isn't very helpful since there are many "stop" words that don't man much
    # now just the top 10 wordss give a good idea of what the book is about!
    stopwords = nltk.corpus.stopwords.words('english')
    new_c = c.copy()
    for key in c:
        if key in stopwords:
            del new_c[key]
    new_c.most_common(n=10)




.. parsed-literal::

    [(u'wife', 353),
     (u'one', 352),
     (u'life', 271),
     (u'man', 241),
     (u'would', 237),
     (u'said', 227),
     (u'may', 219),
     (u'husband', 208),
     (u'good', 205),
     (u'children', 194)]



What is the difference between words1 and words2?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    # words in words1 but not in words2
    w12 = list(set(words1) - set(words2))
    w12[:10]




.. parsed-literal::

    [u'wedmore',
     u'servantgirl',
     u'childs',
     u'folklore',
     u'mores',
     u'loveletters',
     u'itliterary',
     u'motheror',
     u'modium',
     u'worldthen']



.. code:: python

    # words in word2 but not in word1
    w21 = list(set(words2) - set(words1))
    w21[:10]




.. parsed-literal::

    [u'_john',
     u"daughter's",
     u'_illustrated',
     u'party.',
     u'seventy-seven',
     u'34.',
     u'co-operation',
     u'mercury._',
     u'proudie',
     u'_publishers']



.. code:: python

    %load_ext version_information
    
    %version_information requests, regex, nltk


.. parsed-literal::

    The version_information extension is already loaded. To reload it, use:
      %reload_ext version_information




.. raw:: html

    <table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.5 (default, Mar  9 2014, 22:15:05) [GCC 4.2.1 Compatible Apple LLVM 5.0 (clang-500.0.68)]</td></tr><tr><td>IPython</td><td>2.1.0</td></tr><tr><td>OS</td><td>posix [darwin]</td></tr><tr><td>requests</td><td>2.3.0</td></tr><tr><td>regex</td><td>2.4.46</td></tr><tr><td>nltk</td><td>2.0.4</td></tr><tr><td colspan='2'>Sat Aug 02 13:20:24 2014 EDT</td></tr></table>



