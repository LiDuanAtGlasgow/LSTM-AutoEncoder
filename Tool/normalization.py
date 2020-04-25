# pylint: skip-file
####Normalization of Images####
import torch
import numpy as np
import Picture_Dataset
import argparse
from torchvision.transforms import transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import time

parser=argparse.ArgumentParser()
parser.add_argument('--Image_Type',type=int,default=1,help='the type of image')
opt=parser.parse_args()

transforms=T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])

if opt.Image_Type==1:
    cloth_dataset=Picture_Dataset.PictureDataset(file_path='/home/kentuen/AutoEncoder-LSTM/LSTM-AutoEncoder/Database/Real/depth/',csv_path='/home/kentuen/AutoEncoder-LSTM/LSTM-AutoEncoder/csv_clothes/real/depth/full.csv',idx_column=3,transforms=transforms)
if opt.Image_Type==2:
    cloth_dataset=Picture_Dataset.PictureDataset(file_path='/home/kentuen/AutoEncoder-LSTM/LSTM-AutoEncoder/Database/Real/rgb/',csv_path='/home/kentuen/AutoEncoder-LSTM/LSTM-AutoEncoder/csv_clothes/real/rgb/full.csv',idx_column=3,transforms=transforms)

data_size=len(cloth_dataset)
indice=list(range(data_size))
np.random.shuffle(indice)

normalize_sampler=SubsetRandomSampler(indice)

normalize_dataloader=DataLoader(dataset=cloth_dataset,
                                batch_size=100,
                                sampler=normalize_sampler,
                                num_workers=4)


def normalization(dataloader):
    data_mean=[]
    data_std=[]
    start_time=time.time()
    item_time=int(len(dataloader)/10)
    for i, data in enumerate(dataloader):
        numpy_image=data['Image'].numpy()
        image_mean=np.mean(numpy_image,axis=(0,2,3))
        image_std=np.std(numpy_image,axis=(0,2,3))
        data_mean.append(image_mean)
        data_std.append(image_std)
        if ((i+1)%(item_time+1)==1):
            print("[Batch:%d/%d][Duration:%f] image has been completed "%(i+1,len(dataloader),time.time()-start_time))
            print ("Mean is: ",data_mean[i])
            print ("Std is:",data_std[i])
            start_time=time.time()
    total_mean=np.mean(data_mean,axis=0)
    total_std=np.mean(data_std,axis=0)
    return total_mean,total_std

mean,std=normalization(normalize_dataloader)

if opt.Image_Type==1:
    print ("[Depth]mean is:",mean,"[Depth]std is:",std)
if opt.Image_Type==2:
    print ("[RGB]mean is:",mean,"[RGB]std is:",std)

    