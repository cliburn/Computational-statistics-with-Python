TARGET=_apop.so
OBJECTS=ftest.o ftest_wrap.o
CFLAGS=-I/usr/include -I/usr/local/Cellar/gsl/1.16/include -I/Users/cliburn/anaconda/include/python2.7 -g -Wall -Wno-self-assign -O3
LDLIBS=-bundle -lgsl -lapophenia -lpython2.7
CC=clang

SWIGTARGET=apop.py ftest_wrap.c
SWIGOBJECTS=ftest.i 
SWIG=swig
SWIGFLAGS=-python

$(TARGET): $(OBJECTS) $(SWGTARGET)
	 $(CC) $(CFLAGS) $(LDLIBS) -o $(TARGET) $(OBJECTS)
        
$(SWIGTARGET): $(SWIGOBJECTS)
	 $(SWIG) $(SWIGFLAGS) $(SWIGOBJECTS)
        
clean:
	 rm $(SWIGTARGET) $(OBJECTS) $(TARGET)