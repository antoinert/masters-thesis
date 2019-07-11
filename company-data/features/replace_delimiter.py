#!/usr/bin/python

import sys
import csv
import glob

file_arg = sys.argv[1]
delimiter_arg = sys.argv[2]

filename = glob.glob('./{}'.format(file_arg))
filename_new = '{}_new.csv'.format(filename[0])

rows = []

count_start = 0
count_end = 0

with open(filename[0], 'rt') as csvfile:
    rowreader = csv.reader(csvfile, delimiter=delimiter_arg)
    
    for row in rowreader:
        rows.append(row)
        count_start = count_start + 1

print("{}: {}".format(filename[0].split('/')[2], count_start))

with open(filename_new, 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        print(row)
        writer.writerow(row)
        count_end = count_end + 1
        
print("{}: {}".format(filename_new, count_end))
        