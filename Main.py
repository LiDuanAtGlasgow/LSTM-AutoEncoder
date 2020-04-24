# pylint: skip-file
#=======AutoEncoder-LSTM Architecture=====#

import argparse
import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
import torchvision.datasets as datasets
from Model import LSTM
from Model import AutoEncoder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from Model import Classifier
from Tool import Picture_Dataset
from Train import autoencoder_train
from Train import classification
from Train import lstm_train
from Tool import freeze
from Model import AutoEncoder_Depth
from Tool import BatchRandomSampler
from Train import random_classification
from Train import memo_random

parser = argparse.ArgumentParser()
parser.add_argument("--train_period",type=str,default="AUTOENCODER",help="The period of training")
parser.add_argument("--auto_size", type=int, default=20, help="bacth size for AUTOENCODER")
parser.add_argument("--memo_size",type=int,default=100,help='bacth size for LSTM')
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--AutoEncoder_Type",type=int,default=1,help='AutoEncoder_Type Selection')
parser.add_argument("--n_class",type=int,default=3,help='number of classes')
parser.add_argument("--index_column",type=int,default=2,help='index of the class column')
parser.add_argument("--train_method",type=int,default=1,help='train method')
parser.add_argument("--real_image",type=int,default=1,help='check if it is the real image')
parser.add_argument("--Epoch_Auto",type=int,default=13,help="The Epoch of AutoEncoder")
parser.add_argument("--Epoch_Memo",type=int,default=22,help="The Epoch of Memory LSTM")
parser.add_argument("--memo_sampler_size",type=int,default=100,help="The Batch Size of Memo Sampler")
parser.add_argument("--class_sampler_size",type=int,default=4,help="The Batch Size of Class Sampler")
parser.add_argument("--class_size",type=int,default=4,help="bacth size for CLASSIFIER")
parser.add_argument("--Epoch_Class",type=int, default=22,help="The Epoch of Class Stage")
opt=parser.parse_args()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transforms=T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
        ])

previous_version_path="./previous_version/"
if not os.path.exists(previous_version_path):
    os.makedirs(previous_version_path)
if opt.AutoEncoder_Type==1:
    auto_path='./previous_version/auto_depth.pth'
    memo_path='./previous_version/memo_depth.pth'
    class_path='./previous_version/class_depth.pth'
    rand_path='./previous_version/rand_depth.pth'
    memrand_path='./previous_version/memrand_depth.pth'
elif opt.AutoEncoder_Type==2:
    auto_path='./previous_version/auto_rgb.pth'
    memo_path='./previous_version/memo_rgb.pth'
    class_path='./previous_version/class_rgb.pth'
    rand_path='./previous_version/rand_rgb.pth'
    memrand_path='./previous_version/memrand_rgb.pth'


if opt.real_image==1:
    if opt.AutoEncoder_Type==1:
        if opt.train_method==1:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Simulation/depth/',csv_path='./csv_clothes/simulation/depth/trial.csv',idx_column=opt.index_column,transforms=transforms)
        elif opt.train_method==2:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Simulation/depth/',csv_path='./csv_clothes/simulation/depth/full.csv',idx_column=opt.index_column,transforms=transforms)
    if opt.AutoEncoder_Type==2:
        if opt.train_method==1:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Simulation/rgb/',csv_path='./csv_clothes/simulation/rgb/trial.csv',idx_column=opt.index_column,transforms=transforms)
        elif opt.train_method==2:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Simulation/rgb/',csv_path='./csv_clothes/simulation/rgb/full.csv',idx_column=opt.index_column,transforms=transforms)
if opt.real_image==2:
    if opt.AutoEncoder_Type==1:
        if opt.train_method==1:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/trial.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_25=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_25_trial.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_75=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_75_trial.csv',idx_column=opt.index_column,transforms=transforms)
        elif opt.train_method==2:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/full.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_25=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_25_full.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_75=Picture_Dataset.PictureDataset(file_path='./Database/Real/depth/',csv_path='./csv_clothes/real/depth/LOOD_75_full.csv',idx_column=opt.index_column,transforms=transforms)
    if opt.AutoEncoder_Type==2:
        if opt.train_method==1:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/trial.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_25=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/LOOD_25_trial.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_75=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/LOOD_75_trial.csv',idx_column=opt.index_column,transforms=transforms)
        elif opt.train_method==2:
            cloth_dataset=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/full.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_25=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/LOOD_25_full.csv',idx_column=opt.index_column,transforms=transforms)
            cloth_dataset_LOOD_75=Picture_Dataset.PictureDataset(file_path='./Database/Real/rgb/',csv_path='./csv_clothes/real/rgb/LOOD_75_full.csv',idx_column=opt.index_column,transforms=transforms)

save_classification='./data/classification/'
if not os.path.exists(save_classification):
    os.makedirs(save_classification)

model=LSTM.LSTM()
model=model.to(device)
net_RGB=AutoEncoder.AutoEncoder()
net_depth=AutoEncoder_Depth.AutoEncoder()
classifier=Classifier.Classifier(output_feature=opt.n_class)
classifier=classifier.to(device)
if opt.AutoEncoder_Type==1:
    net=net_depth
    net=net.to(device)
elif opt.AutoEncoder_Type==2:
    net=net_RGB
    net=net.to(device)


if __name__ == '__main__':
    print("Project Started!")
    epochs=opt.Epoch_Auto
    epoch=opt.Epoch_Memo
    epoch_classifier=opt.Epoch_Class
    if opt.train_period=='AUTOENCODER':
        data_size=len(cloth_dataset)
        n_train=int(data_size*0.8)
        n_val=int(data_size*0.1)
        indices=list(range(data_size))
        np.random.shuffle(indices)

        train_indices=indices[0:n_train]
        val_indices=indices[n_train:n_train+n_val]
        test_indices=indices[n_train+n_val:]

        train_sampler=SubsetRandomSampler(train_indices)
        val_sampler=SubsetRandomSampler(val_indices)
        test_sampler=SubsetRandomSampler(test_indices)

        train_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.auto_size,
                         sampler=train_sampler,
                         num_workers=4)
        val_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.auto_size,
                         sampler=val_sampler,
                         num_workers=4)
        test_loader=DataLoader(dataset=cloth_dataset,
                         batch_size=opt.auto_size,
                         sampler=test_sampler,
                         num_workers=4)
        net=autoencoder_train.train(net,train_loader,val_loader,test_loader,opt.auto_size,epochs,opt.lr,device,opt.AutoEncoder_Type)
        torch.save(net,auto_path)
    
    if opt.train_period=='MEMOR_STAGE':
        net=torch.load(auto_path)
        freeze.frozon(net)
        data_size=len(cloth_dataset)
        n_train=int(data_size*0.80)
        n_val=int(data_size*0.10)
        indice=list(range(data_size))
        batch_size=opt.memo_size
        iter_num=int(len(indice)/batch_size)
        random_num=torch.randperm(iter_num)
        for i in range(iter_num):
            t=random_num[i]
            random_indice=indice[t*batch_size:(t+1)*batch_size]
            indice[i*batch_size:(i+1)*batch_size]=random_indice
        train_indice=indice[:n_train]
        val_indice=indice[n_train:n_train+n_val]
        test_indice=indice[n_train+n_val:]

        train_sampler=BatchRandomSampler.BatchRandomSampler(train_indice,batch_size)
        val_sampler=BatchRandomSampler.BatchRandomSampler(val_indice,batch_size)
        test_sampler=BatchRandomSampler.BatchRandomSampler(test_indice,batch_size)

        memo_train_loader=DataLoader(dataset=cloth_dataset,
                                    batch_size=opt.memo_size,
                                    sampler=train_sampler,
                                    num_workers=4)
        memo_val_loader=DataLoader(dataset=cloth_dataset,
                                    batch_size=opt.memo_size,
                                    sampler=val_sampler,
                                    num_workers=4)
        memo_test_loader=DataLoader(dataset=cloth_dataset,
                                    batch_size=opt.memo_size,
                                    sampler=test_sampler,
                                    num_workers=4)
        memo=lstm_train.train(model,epoch,memo_train_loader,memo_val_loader,memo_test_loader,net,opt.lr,device,opt.AutoEncoder_Type)
        torch.save(memo,memo_path)                          

    if opt.train_period=='RAND_STAGE':
        net=torch.load(auto_path)
        freeze.frozon(net)
        data_size=len(cloth_dataset)
        n_train=int(data_size*0.8)
        n_val=int(data_size*0.1)
        indices=list(range(data_size))
        np.random.shuffle(indices)
        train_indices=indices[0:n_train]
        val_indices=indices[n_train:n_train+n_val]
        test_indices=indices[n_train+n_val:]

        rand_train_sampler=SubsetRandomSampler(train_indices)
        rand_val_sampler=SubsetRandomSampler(val_indices)
        rand_test_sampler=SubsetRandomSampler(test_indices)

        rand_train_loader=DataLoader(dataset=cloth_dataset,
                        batch_size=opt.auto_size,
                        sampler=rand_train_sampler,
                        num_workers=4)
        rand_val_loader=DataLoader(dataset=cloth_dataset,
                        batch_size=opt.auto_size,
                        sampler=rand_val_sampler,
                        num_workers=4)
        rand_test_loader=DataLoader(dataset=cloth_dataset,
                        batch_size=opt.auto_size,
                        sampler=rand_test_sampler,
                        num_workers=4)

        rand_classifier=random_classification.train(epoch_classifier,rand_train_loader,rand_val_loader,rand_test_loader,net,opt.lr,device,opt.AutoEncoder_Type,classifier,opt.index_column)
        torch.save(rand_classifier,rand_path)
    
    if opt.train_period=="MEMRAND_STAGE":
        net=torch.load(auto_path)
        freeze.frozon(net)
        classifier=torch.load(rand_path)
        freeze.frozon(classifier)
        data_25=len(cloth_dataset_LOOD_25)
        data_75=len(cloth_dataset_LOOD_75)
        n_train=int(data_75*0.80)
        indices_25=list(range(data_25))
        indices_75=list(range(data_75))
        iter_num=int(len(indices_75)/opt.memo_sampler_size)
        random_num=torch.randperm(iter_num)
        for i in range(iter_num):
            t=random_num[i]
            random_indice=indices_75[t*opt.memo_sampler_size:(t+1)*opt.memo_sampler_size]
            indices_75[i*opt.memo_sampler_size:(i+1)*opt.memo_sampler_size]=random_indice

        train_indices=indices_75[:n_train]
        val_indices=indices_75[n_train:]
        test_indices=indices_25

        train_sampler=BatchRandomSampler.BatchRandomSampler(train_indices,opt.memo_sampler_size)
        val_sampler=BatchRandomSampler.BatchRandomSampler(val_indices,opt.memo_sampler_size)
        test_sampler=BatchRandomSampler.BatchRandomSampler(test_indices,opt.memo_sampler_size)

        memrand_train_loader=DataLoader(dataset=cloth_dataset_LOOD_75,
                                batch_size=opt.memo_size,
                                sampler=train_sampler,
                                num_workers=4)
        memrand_val_loader=DataLoader(dataset=cloth_dataset_LOOD_75,
                                batch_size=opt.memo_size,
                                sampler=val_sampler,
                                num_workers=4)
        memrand_test_loader=DataLoader(dataset=cloth_dataset_LOOD_25,
                                batch_size=opt.memo_size,
                                sampler=test_sampler,
                                num_workers=4)
    
        memrand_model=memo_random.train(model,epoch,memrand_train_loader,memrand_val_loader,memrand_test_loader,net,opt.lr,device,opt.AutoEncoder_Type,classifier,opt.index_column)
        torch.save(memrand_model,memrand_path)

