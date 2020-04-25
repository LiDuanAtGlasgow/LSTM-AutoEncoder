# pylint: skip-file
import os
from openpyxl import load_workbook
from matplotlib import pyplot as plt
import pandas as pd
import time as time

filename='./Experiments.xlsx'
save_figure='./data/comparison/'
if not os.path.exists(save_figure):
    os.makedirs(save_figure)
workbook=load_workbook(filename)
sheet=workbook.active

AEC_Train_Loss=[]
AEC_Train_Accuracy=[]
AEC_Test_Loss=[]
AEC_Test_Accuracy=[]

"""
AELC_Train_Loss=[]
AELC_Train_Accuracy=[]
AELC_Test_Loss=[]
AELC_Test_Accuracy=[]
"""


AEC_Train_Loss_sheet=sheet["A94:A115"]
AEC_Train_Accuracy_sheet=sheet["B94:B115"]
AEC_Test_Loss_sheet=sheet["C94:C115"]
AEC_Test_Accuracy_sheet=sheet["D94:D115"]

"""
AELC_Train_Loss_sheet=sheet["A25:A46"]
AELC_Train_Accuracy_sheet=sheet["B25:B46"]
AELC_Test_Loss_sheet=sheet["C25:C46"]
AELC_Test_Accuracy_sheet=sheet["D25:D46"]
"""
for t in range(22):
    AEC_Train_Loss.append(AEC_Train_Loss_sheet[t][0].value)
    AEC_Train_Accuracy.append(AEC_Train_Accuracy_sheet[t][0].value)
    AEC_Test_Loss.append(AEC_Test_Loss_sheet[t][0].value)
    AEC_Test_Accuracy.append(AEC_Test_Accuracy_sheet[t][0].value)
    """
    AELC_Train_Loss.append(AELC_Train_Loss_sheet[t][0].value)
    AELC_Train_Accuracy.append(AELC_Train_Accuracy_sheet[t][0].value)
    AELC_Test_Loss.append(AELC_Test_Loss_sheet[t][0].value)
    AELC_Test_Accuracy.append(AELC_Test_Accuracy_sheet[t][0].value)
    """


X=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

sbplt=plt.figure()
sbplt.add_subplot(111)

sbplt1=plt.subplot()
df=pd.DataFrame({'x':X,'AELC_Train_Loss':AEC_Train_Loss,'AELC_Train_Acc':AEC_Train_Accuracy, 'AELC_Test_Loss':AEC_Test_Loss, 'AELC_Test_Acc':AEC_Test_Accuracy})
sbplt1.plot('x','AELC_Train_Loss',data=df,color='red',label='Train_Loss')
sbplt1.plot('x','AELC_Test_Loss',data=df,color='blue',label='Test_Loss', linestyle='dashed')
sbplt1.set_xlabel('Epoch(s)')
sbplt1.set_ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right',bbox_to_anchor=(1.1,1.16))
sbplt2=sbplt1.twinx()
sbplt2.plot('x','AELC_Train_Acc',data=df,color='green',label='Train_Acc')
sbplt2.plot('x','AELC_Test_Acc',data=df,color='orange',label='Test_Acc', linestyle='dashed')
sbplt2.set_ylabel('Accuracy(%)')
plt.title('Stiffness(RGB) AELC')
plt.grid(True)
plt.legend(loc="upper left",bbox_to_anchor=(-0.1,1.16))
plt.tight_layout()
plt.savefig(os.path.join(save_figure,'%f.png'%(time.time())),dip=100)
plt.show()
print ('Finished!')


