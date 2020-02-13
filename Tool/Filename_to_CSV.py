import os
import csv

f=open('C:/Users/Li_Duan/Desktop/AE_P1_Second/Tool/label_RGB_hard.csv','r+')
w=csv.writer(f)
for path,dirs,files in os.walk("C:/Users/Li_Duan/Desktop/AE_P1_Second/Database/Train_Database/ClothingResources/hard"):
    for filename in (files):
        print(filename)
        w.writerow([filename])
