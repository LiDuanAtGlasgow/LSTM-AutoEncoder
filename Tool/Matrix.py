import os
from openpyxl import load_workbook
from matplotlib import pyplot as plt
import pandas as pd

filename='C:/Users/Li_Duan/Desktop/LSTM-AutoEncoder/Tool/Tool_Statistics/Statistics_Table.xlsx'
save_figure='./data/comparison/'
if not os.path.exists(save_figure):
    os.makedirs(save_figure)
workbook=load_workbook(filename)
sheet=workbook.active

SDA_Depth_Test=[]
SDA_Depth_Train=[]
AE_RGB_Train=[]
AE_RGB_Test=[]

SDA_Depth_Train_sheet=sheet['A2:A23']
SDA_Depth_Test_sheet=sheet['B2:B23']
AE_RGB_Train_sheet=sheet['A25:A46']
AE_RGB_Test_sheet=sheet['B25:B46']
for t in range(len(SDA_Depth_Train_sheet)):
    SDA_Depth_Train.append(SDA_Depth_Train_sheet[t][0].value)
    SDA_Depth_Test.append(SDA_Depth_Test_sheet[t][0].value)
    AE_RGB_Train.append(AE_RGB_Train_sheet[t][0].value)
    AE_RGB_Test.append(AE_RGB_Test_sheet[t][0].value)

print(len(SDA_Depth_Train_sheet))
X=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

ax_t=plt.figure()
ax_t.add_subplot(111)
ax=plt.subplot()

df=pd.DataFrame({'x':X,'SDA_Depth_Train':SDA_Depth_Train,'SDA_Depth_Test':SDA_Depth_Test, 'AE_RGB_Train':AE_RGB_Train, 'AE_RGB_Test':AE_RGB_Test})

ax.plot('x','SDA_Depth_Train',data=df,color='red',label='SDA_Depth_Train')
ax.plot('x','SDA_Depth_Test',data=df,color='blue',label='SDA_Depth_Test', linestyle='dashed')
ax.set_xlabel('Epoch(s)')
ax.set_ylabel('MSE Loss')
plt.title('SDA (Simulated Depth Image)')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(save_figure,'(RGB_Image)'),dip=100)
plt.show()
print ('Finished!')


