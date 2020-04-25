# pylint: skip-file
import os
import csv

exlxs="./pant3_rgb.csv"
f=open(exlxs,'r+')
w=csv.writer(f)
for path,dirs,files in os.walk("./mask_clothes/pant3_rgb/"):
    for filename in (files):
        print(filename)
        w.writerow([filename])
