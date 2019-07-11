#!/usr/bin/python

import sys
import csv
import glob

files = glob.glob('./{}'.format(sys.argv[1]))

for filename in files:
    count = 0
    first_date = ""
    with open(filename, 'rt') as csvfile:
        rowreader = csv.reader(csvfile, delimiter=',')
        for row in rowreader:
            count = count + 1 
            if count == 2:
                    first_date = row[1]

    print(filename)
    print("{}: {}, first data: {}".format(filename.split('/')[2], count, first_date))