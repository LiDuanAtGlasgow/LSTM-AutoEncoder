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
from Model import AutoEncoder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler as SequentialSampler
from Model import AutoEncoder_depth
from Model import Classifier
from Tool import Picture_Dataset as Picture_Dataset
from Tool import Normalization
from Train import autoencoder_train
from Train import classification
from Train import classification_random
from Train import loo_train
from Train import lstm_train
from Tool import freeze
from Train import lstm_loo_train


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--batch_size_for_prediction",type=int,default=20,help='bacth size for the first test')
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--AutoEncoder_Type",type=int,default=1,help='AutoEncoder_Type Selection')
parser.add_argument("--n_class",type=int,default=3,help='number of classes')
parser.add_argument("--index_column",type=int,default=2,help='index of the class column')
parser.add_argument("--train_method",type=int,default=1,help='train method')
parser.add_argument("--extractor",type=int,default=1,help='extractor type')
parser.add_argument("--step1_type",type=int,default=1,help='step1 type of model_step1')
parser.add_argument("--real_image",type=int,default=1,help='check if it is the real image')
parser.add_argument("--dataset_type",type=int,default=1,help='the train indice of the dataset')
parser.add_argument("--batch_size_for_classification",type=int,default=4,help='bacth size for classification')
parser.add_argument("--test",type=int,default=1,help='test images from LeaveOneOut strategy')
parser.add_argument("--batch_size_for_loo_test",type=int,default=20,help='Batch Size for LeaveOneOut Test')

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
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/real_depth_part/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_part/real_depth_part/',csv_path='./Tool/Real_depth_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        elif opt.train_method==2:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/real_depth_all/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_all/real_depth_all/',csv_path='./Tool/Real_depth_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    if opt.AutoEncoder_Type==2:
        if opt.train_method==1:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/real_rgb_part/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_part/real_rgb_part/',csv_path='./Tool/Real_rgb_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        elif opt.train_method==2:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/real_rgb_all/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_all_white/real_rgb_all /',csv_path='./Tool/Real_rgb_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)

if opt.test==2:
    if opt.train_method==1:
        if opt.AutoEncoder_Type==1:
            train_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_all/real_depth_all/',csv_path='./Tool/Train_Real_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
            test_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_all/real_depth_all/',csv_path='./Tool/Test_Real_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        if opt.AutoEncoder_Type==2:
            train_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_all/real_rgb_all/',csv_path='./Tool/Train_Real_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
            test_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_all/real_rgb_all/',csv_path='./Tool/Test_Real_Part.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    if opt.train_method==2:
        if opt.AutoEncoder_Type==1:
            train_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_all/real_depth_all/',csv_path='./Tool/Train_Real_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)
            test_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_depth_all/real_depth_all/',csv_path='./Tool/Test_Real_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        if opt.AutoEncoder_Type==2:
            train_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_all/real_rgb_all/',csv_path='./Tool/Train_Real_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)
            test_real=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/real_rgb_all/real_rgb_all/',csv_path='./Tool/Test_Real_All.csv',idx_column=opt.index_column,transforms=transforms,device=device)


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
data_size=len(cloth_dataset_c)
n_train=int(data_size*0.8)
n_test=int(data_size*0.1)
if opt.test==2:
    test_size=len(test_real)
    test_real_indices=list(range(test_size))
    train_size=len(train_real)
    train_real_indices=list(range(train_size))
indices=list(range(data_size))
lstm_indices=indices
train_indices=indices[0:n_train]
val_indices=indices[n_train:n_train+n_test]
test_indices=indices[n_train+n_test:]

train_sampler=SubsetRandomSampler(train_indices)
val_sampler=SubsetRandomSampler(val_indices)
test_sampler=SubsetRandomSampler(test_real_indices)
lstm_sampler=SequentialSampler(lstm_indices)

if opt.test==2:
    test_real_sampler=SequentialSampler(test_real_indices)
    train_real_sampler=SequentialSampler(train_real_indices)

model=LSTM.LSTM()
model=model.to(device)
net_RGB=AutoEncoder.AutoEncoder()
net_depth=AutoEncoder_depth.AutoEncoder()
Classifier=Classifier.Classifier(output_feature=opt.n_class)
Classifier=Classifier.to(device)

if opt.extractor==1:
    net_opt=net_depth
elif opt.extractor==2:
    net_opt=net_RGB



train_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.batch_size,
                         sampler=train_sampler,
                         num_workers=1)
val_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.batch_size,
                         sampler=val_sampler,
                         num_workers=1)
test_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.batch_size,
                         sampler=test_sampler,
                         num_workers=1)

lstm_loader=DataLoader(dataset=cloth_dataset_c,
                        batch_size=opt.batch_size_for_prediction,
                        sampler=lstm_sampler,
                        num_workers=1)
cloth_classification_loader=DataLoader(dataset=cloth_dataset_c,
                                   batch_size=opt.batch_size_for_classification,
                                   sampler=lstm_sampler,
                                   num_workers=1)
cloth_nonsequential_loader=DataLoader(dataset=cloth_dataset_c,
                                   batch_size=opt.batch_size_for_classification,
                                   sampler=lstm_sampler,
                                   num_workers=1)
if opt.test==2:
    test_real_loader=DataLoader(dataset=test_real,
                                batch_size=opt.batch_size_for_loo_test,
                                sampler=test_real_sampler,
                                num_workers=1)
    train_real_loader=DataLoader(dataset=train_real,
                                batch_size=opt.batch_size_for_loo_test,
                                sampler=train_real_sampler,
                                num_workers=1)


############Training################
if __name__ == '__main__':
    print("Project Started!")
    epoch=22
    epochs=13
    """
    net=autoencoder_train.train(net_opt,train_loader,val_loader,test_loader,opt.batch_size,epochs,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(net,path)
    """
    net=torch.load(path)
    freeze.frozon(net)
    """
    model=lstm_train.train(model,epoch,lstm_loader,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type)
    """
    model=torch.load(step1_path)
    freeze.frozon(model)
    if opt.test==2:
        """
        loo_model=lstm_loo_train.train(model,epoch,train_real_rgb_loader,test_real_rgb_loader,net,opt.lr,opt.batch_size_for_loo_test,device,opt.AutoEncoder_Type)
        """
        loo_classifier=loo_train.train(model,epoch,train_real_loader,test_real_loader,net,opt.lr,device,opt.AutoEncoder_Type,Classifier)
    """
    classifier=classification.train(model,epoch,cloth_classification_loader,net,opt.lr,device,opt.AutoEncoder_Type,Classifier)
    """
    """
    classifier_random=classification_random.train(net,cloth_nonsequential_loader,epoch,opt.lr,device,opt.AutoEncoder_Type,Classifier)
    """
##################################################