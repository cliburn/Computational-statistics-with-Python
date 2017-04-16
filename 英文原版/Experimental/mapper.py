#!/usr/bin/env python

import sys

def read_input(file):
    for line in file:
        yield line.split()

def main(sep='\t'):
    data = read_input(sys.stdin)
    for words in data:
        for word in words:
            print '%s%s%d' % (word, sep, 1)

if __name__ == '__main__':
    main()