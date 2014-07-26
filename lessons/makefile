TARGET = test_make
OBJECTS = stuff.o
CFLAGS = -g -Wall -O3 
LDLIBS = -lm
CC = clang

all: $(TARGET)
    
clean:
	 rm $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)