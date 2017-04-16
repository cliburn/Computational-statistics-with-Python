#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys

def read_mapper_output(file, sep):
    for line in file:
        yield line.rstrip().split(sep, 1)

def main(sep='\t'):
    data = read_mapper_output(sys.stdin, sep=sep)
    for word, group in groupby(data, itemgetter(0)):
        total_count = sum(int(count) for word, count in group)
        print '%s%s%d' % (word, sep, total_count)

if __name__ == '__main__':
    main()