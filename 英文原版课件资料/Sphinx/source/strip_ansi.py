#!/usr/bin/env python

import glob

for f in glob.glob('*rst'):
    text = open(f).read()
    text = text.replace('[0;31m', '')
    text = text.replace('[0m', '')
    text = text.replace('^[[34m', './')
    text = text.replace('[m', '/')
    with open(f, 'w') as fout:
        fout.write(text)
