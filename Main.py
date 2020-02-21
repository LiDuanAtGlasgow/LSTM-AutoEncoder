#=======AutoEncoder-LSTM Architecture=====#

import argparse
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torch.utils.data import Dataset as Dataset
from Model import LSTM as LSTM
from Model import AutoEncoder as AutoEncoder
from Train import LSTM_Training_Step1
from Tool import Frozon_and_Free_Param as frozon_and_free_Get
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler as SequentialSampler
from Tool import Normalization
from Model import AutoEncoder_Depth as AutoEncoder_Depth
from Model import Classifier
from Tool import Window_Sliding_Test
from Tool import Picture_Dataset as Picture_Dataset
from Tool import Normalization
from Train import LSTM_Training_No_Label_Step1 as LSTM_Training_No_Label_Step1
from Train import AutEncoder_Train
from Train import classification


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--batch_size_for_prediction",type=int,default=20,help='Bacth Szie for the first test')
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--AutoEncoder_Type",type=int,default=1,help='AutoEncoder_Type Selection')
parser.add_argument("--n_class",type=int,default=3,help='number of classes')
parser.add_argument("--index_column",type=int,default=2,help='index of the class column')
parser.add_argument("--train_method",type=int,default=1,help='train method')
parser.add_argument("--extractor",type=int,default=1,help='extractor type')
parser.add_argument("--step1_type",type=int,default=1,help='step1 type of model_step1')
parser.add_argument("--real_image",type=int,default=1,help='check if it is the real image')
parser.add_argument("--dataset_type",type=int,default=1,help='the train indice of the dataset')

opt=parser.parse_args()
####################Loading Process#############

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transforms=T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
        ])
if opt.AutoEncoder_Type==1:
    path_step1='./save_model/model_Depth_step1.pth'
elif opt.AutoEncoder_Type==2:
    path_step1='./save_model/model_RGB_step1.pth'
            
##########################
if opt.AutoEncoder_Type==1:
    if opt.train_method==1:
        cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Depth_Dataset_Small/',transform=transforms)
        cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Depth_Dataset/Depth_Dataset/',csv_path='./Tool/label_Depth_part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    elif opt.train_method==2:
        cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Depth_Dataset/',transform=transforms)
        cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Depth_Dataset/Depth_Dataset/',csv_path='./Tool/label_Depth.csv',idx_column=opt.index_column,transforms=transforms,device=device)
if opt.AutoEncoder_Type==2:
    if opt.train_method==1:
        cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/RGB+Training+Set+Small/',transform=transforms)
        cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/RGB+Training+Set/RGB+Training+Set/',csv_path='./Tool/label_RGB_part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    elif opt.train_method==2:
        cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/RGB+Training+Set/',transform=transforms)
        cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/RGB+Training+Set/RGB+Training+Set/',csv_path='./Tool/label_RGB.csv',idx_column=opt.index_column,transforms=transforms,device=device)
if opt.real_image==2:
    if opt.AutoEncoder_Type==1:
        if opt.train_method==1:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_Depth_Complete_Small/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_Depth_Complete/ClothingResources/Clothes_DB_database_Depth_Complete/',csv_path='./Tool/Label_Part_toutput_Depth.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        elif opt.train_method==2:
            
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_Depth_Complete/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_Depth_Complete/Clothes_DB_database_Depth_Complete/',csv_path='./Tool/Label_All_toutput_Depth.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    if opt.AutoEncoder_Type==2:
        if opt.train_method==1:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete_small/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/Clothes_DB_database_RGB_Complete/',csv_path='./Tool/Label_Part_toutput_RGB.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        elif opt.train_method==2:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/Clothes_DB_database_RGB_Complete/',csv_path='./Tool/Label_All_toutput_RGB.csv',idx_column=opt.index_column,transforms=transforms,device=device)


path_folder='./save_model/'
if not os.path.exists(path_folder):
    os.makedirs(path_folder)
if opt.AutoEncoder_Type==1:
    path= './save_model/model_depth_autoencoder.pth'
elif opt.AutoEncoder_Type==2:
    path='./save_model/model_RGB_autoencoder.pth'
if opt.step1_type==1:
    step1_path='./save_model/model_depth_autoencoder_step1.pth'
if opt.step1_type==2:
    step1_path='./save_model/model_RGB_autoencoder_step1.pth'


if opt.real_image==2:
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    if opt.AutoEncoder_Type==1:
        path= './save_model/model_depth_autoencoder_real.pth'
    elif opt.AutoEncoder_Type==2:
        path='./save_model/model_RGB_autoencoder_real.pth'
    if opt.step1_type==1:
        step1_path='./save_model/model_depth_autoencoder_step1_real.pth'
    if opt.step1_type==2:
        step1_path='./save_model/model_RGB_autoencoder_step1_real.pth'

save_classification='./data/classification/'
if not os.path.exists(save_classification):
    os.makedirs(save_classification)


seed=42
np.random.seed(seed)
torch.manual_seed(seed)
data_size=len(cloth_dataset_c)
n_test=int(data_size*0.1)
n_train=data_size-n_test
random_seed=42
indices=list(range(data_size))
shuffle_dataset=True
if shuffle_dataset:
    np.random.seed(seed)
    np.random.shuffle(indices)
if opt.dataset_type==1:
    train_indices=indices
if opt.dataset_type==2:
    train_indices=indices[0:n_train]
valid_indices=indices[n_train+n_test:]
test_indices=indices[n_train:n_train+n_test]

train_sampler=SubsetRandomSampler(train_indices)
test_sampler=SubsetRandomSampler(test_indices)


train_lstm_sampler=SequentialSampler(train_indices)
test_lstm_sampler=SequentialSampler(test_indices)

model=LSTM.LSTM(input_feature=1024,n_layer=1,hidden_dim=1024)
model=model.to(device)
net_RGB=AutoEncoder.AutoEncoder(device)
net_depth=AutoEncoder_Depth.AutoEncoder(device)
Classifier=Classifier.Classifier(device,output_feature=opt.n_class,hidden_feature=128)
Classifier=Classifier.to(device)

if opt.extractor==1:
    net_opt=[net_depth]
elif opt.extractor==2:
    net_opt=[net_RGB]



cloth_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.batch_size,
                         sampler=train_sampler,
                         num_workers=1)
cloth_test_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.batch_size,
                         sampler=test_sampler,
                         num_workers=1)

cloth_lstm_train_loader=DataLoader(dataset=cloth_dataset,
                                   batch_size=opt.batch_size_for_prediction,
                                   sampler=train_lstm_sampler,
                                   num_workers=1)
cloth_lstm_loader_c=DataLoader(dataset=cloth_dataset_c,
                                   batch_size=opt.batch_size_for_prediction,
                                   sampler=train_lstm_sampler,
                                   num_workers=1)

                         


#############AutoEncoder Training##############

if __name__ == '__main__':
    print("Project Started!")
    epoch=22
    """
    epochs=10
    """
    """
    net=Discriminator_and_Generator_Training.train(net_opt,cloth_loader,cloth_test_loader,opt.batch_size,epochs,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(net,path)
    """
#############LSTM Training#######################
    net=torch.load(path)
    for n in range(len(net)):
        frozon_and_free_Get.frozon_Param(net[n])
    """
    model=LSTM_Training_Step1.train(model,epoch,cloth_lstm_loader_c,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type)
    """
    model=torch.load(step1_path)
    frozon_and_free_Get.frozon_Param(model)
    Classifier=classification.train(model,epoch,cloth_lstm_loader_c,net,opt.lr,device,opt.AutoEncoder_Type,Classifier)
    """
    Window_Sliding_Test.train(model_one,[cloth_lstm_train_loader],net,device,opt.AutoEncoder_Type)
    """
##################################################