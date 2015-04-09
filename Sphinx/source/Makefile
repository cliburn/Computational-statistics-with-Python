SRC = $(wildcard *.ipynb)
TAR = $(SRC:.ipynb=.rst)

.PHONY: all clean

all: $(TAR)

%.rst: %.ipynb
	ipython nbconvert $< --to rst

clean:
	rm -f $(TAR)

