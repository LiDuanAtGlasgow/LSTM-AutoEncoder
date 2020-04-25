# pylint: skip-file
import numpy as np
import seaborn as sns
from openpyxl import load_workbook
import time
from matplotlib import pyplot as plt

file_location="./Heatmap.xlsx"
workbook=load_workbook(file_location)
sheet=workbook.active


z,o,tw=[],[],[]
"""
th,f=[],[]
"""

z_sheet=sheet["A18:A20"]
o_sheet=sheet["B18:B20"]
tw_sheet=sheet["C18:C20"]
"""
th_sheet=sheet["D12:D16"]
f_sheet=sheet["E12:E16"]
"""

for t in range (len(z_sheet)):
    z.append(100*z_sheet[t][0].value)
    o.append(100*o_sheet[t][0].value)
    tw.append(100*tw_sheet[t][0].value)
    """
    th.append(100*th_sheet[t][0].value)
    f.append(f_sheet[t][0].value*100)
    """

total=[z,o,tw]
annotation=["suit","sweater","tshirt"]

total=np.asarray(total).reshape(3,3)
ax=sns.heatmap(total,linewidth=1.0,annot=True,cmap="YlGnBu",vmin=0,vmax=100,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%) Color Bar"})
plt.title("Classification(%)")
plt.savefig("./heatmap_Real_Simulate.png",dip=100)
plt.show()