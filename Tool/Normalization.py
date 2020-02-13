####Normalization of Images####
import torch
import numpy as np

def Normalization(dataloader):
    data_mean=[]
    data_std0=[]
    data_std1=[]
    for i, data in enumerate(dataloader):
        numpy_image=data['Image'].numpy()
        batch_mean=np.mean(numpy_image,axis=(0,2,3))
        batch_std0=np.std(numpy_image,axis=(0,2,3))
        batch_std1=np.std(numpy_image,axis=(0,2,3),ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)
        if ((i+1)%(int(len(dataloader)/10)+1)==0):
            print("I have finished %d"%i)
    data_mean=np.array(data_mean).mean(axis=0)
    data_std0=np.array(data_std0).mean(axis=0)
    data_std1=np.array(data_std1).mean(axis=0)

    print(data_mean,data_std0,data_std1)
    return data_mean,data_std0,data_std1

    