#=======Future Project for Grasping Clothes Process=====#
# This Project is undertaken by Li Duan, a PhD student in Computing Science 
# from University of Glasgow, United Kingdom
# This Project is for the purpose to commemorize the Tony Stark, 
# the CEO of Stark Industry and the hero of our hearts.
# Thank you, Marvel, who gave me a dream 
# when I was child and helped me with becoming a PhD student 
# and even a Scientist in the furture. 
# Thank you, Tony Stark. 
# I will thank you three thousand.

import PIL.Image as Image
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms as T
import torchvision
import torchvision.datasets as datasets
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Dataset as Dataset
from Model import LSTM as LSTM
from Model import AutoEncoder_3layers as autocoder3
from Model import AutoEncoder_4layers as autocoder4
from Model import AutoEncoder_2layers as autocoder2
from Train import Discriminator_and_Generator_Training as Discriminator_and_Generator_Training
from Train import LSTM_Training_Step1
from Tool import Frozon_and_Free_Param as frozon_and_free_Get
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler as SequentialSampler
from Tool import Add_Noise
from Tool import Normalization
import glob
from Model import AutoEncoder_Depth as autocoder_depth
from Model import Classifier
from Tool import Test_for_the_Next_Two_Frames
from Tool.MySampler import CustomRandomSampler as CustomRandomSampler
from Train import LSTM_Training_Step1_classifier
from Tool import Picture_Dataset as Picture_Dataset
import matplotlib.pyplot as plt
from Model import LSTM_classifier as LSTM_classifier
from Model import Auto_Res as Auto_Res
from Model import LSTM_classifier_autres as LSTM_classifier_autres
from Tool import Normalization
from Train import Discriminator_and_Generator_Training_clean as Discriminator_and_Generator_Training_clean
from Train import Discriminator_and_Generator_second_clean as Discriminator_and_Generator_second_clean
from Model import LSTM_classifier_second as LSTM_classifier_second
from Model import AutoEnc_Stacked_Level1 as AutoEnc_Stacked_Level1
from Model import AutoEnc_Stacked_Level0 as AutoEnc_Stacked_Level0
from Train import Discriminator_and_Generator_Training_noise
from Train import LSTM_Training_Step1_classifier_clean as LSTM_Training_Step1_classifier_clean
from Train import Classifier_Training as Classifier_Training
from Train import LSTM_Training_Step1_clean as LSTM_Training_Step1_clean
from Model import AutoEnc_Stacked_Level1_Coloured as AutoEnc_Stacked_Level1_Coloured
from Model import AutoEnc_Stacked_Level0_Coloured as AutoEnc_Stacked_Level0_Coloured
from Train import LSTM_Training_No_Label_Step1 as LSTM_Training_No_Label_Step1
from Train import Discriminator_and_Generator_Training_fuer_AE as Discriminator_and_Generator_Training_fuer_AE
from Train import LSTM_Training_Label_Step
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=13, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--batch_size_for_test_of_textured_clothes",type=int,default=20,help='Batch Size of Tested Textured Clothes')
parser.add_argument("--batch_size_for_prediction",type=int,default=100,help='Bacth Szie for the first test')
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("---input_dim",type=int,default=10,help="input dimension of LSTM")
parser.add_argument("---num_layers",type=int,default=2,help="number of layers of LSTM")
parser.add_argument("---output_dim",type=int,default=1,help="output dimension of LSTM")
parser.add_argument("---hidden_dim",type=int,default=32,help="hidden dimension of LSTM")
parser.add_argument("--epochs_for_test_textured_clothes",type=int,default=4,help='Epoch for test of textured clothes')
parser.add_argument("--batch_size_textured_original",type=int,default=100,help='Batch size of textured clothes original part')
parser.add_argument("--batch_size_textured_target",type=int,default=100,help='Batch size of textured clothes target part')
parser.add_argument("--AutoEncoder_Type",type=int,default=1,help='AutoEncoder_Type Selection')
parser.add_argument("--n_class",type=int,default=3,help='number of classes')
parser.add_argument("--index_column",type=int,default=2,help='index of the class column')
parser.add_argument("--train_method",type=int,default=1,help='train method')
parser.add_argument("--extractor",type=int,default=1,help='extractor type')
parser.add_argument("--dataset_type",type=int,default=1,help='training_dataset type')
parser.add_argument("--model_type",type=int,default=1,help='training_dataset type')
parser.add_argument("--is_clean",type=int,default=1,help='judge if it is on clean type')
parser.add_argument("--step1_type",type=int,default=1,help='step1 type of model_step1')
parser.add_argument("--real_image",type=int,default=1,help='check if it is the real image')

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
if opt.AutoEncoder_Type==1:
    path_step1_c='./save_model/depth_lstm_classifier_model.pth'
elif opt.AutoEncoder_Type==2:
    path_step1_c='./save_model/RGB_lstm_classifier_model.pth'
            
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
            #cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_Depth_Complete/Clothes_DB_database_Depth_Complete/',csv_path='./Tool/Label_All_toutput_Depth.csv',idx_column=opt.index_column,transforms=transforms,device=device)
    if opt.AutoEncoder_Type==2:
        if opt.train_method==1:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete_small/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/Clothes_DB_database_RGB_Complete/',csv_path='./Tool/Label_Part_toutput_RGB.csv',idx_column=opt.index_column,transforms=transforms,device=device)
        elif opt.train_method==2:
            cloth_dataset=datasets.ImageFolder(root='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/',transform=transforms)
            cloth_dataset_c=Picture_Dataset.PictureDataset(file_path='./Database/Train_Database/ClothingResources/Clothes_DB_database_RGB_Complete/Clothes_DB_database_RGB_Complete/',csv_path='./Tool/Label_All_toutput_RGB.csv',idx_column=opt.index_column,transforms=transforms,device=device)
##########################
##########################
##########################
test_prediction_dataset=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources/',transform=transforms)
test_prediction_dataset_suit=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-Suit-Original',transform=transforms)
test_prediction_dataset_tshirt=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-Tshirt-Original',transform=transforms)
test_prediction_dataset_womansweater=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-WomanSweater-Original',transform=transforms)
test_prediction_dataset_suit_target=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-Suit',transform=transforms)
test_prediction_dataset_tshirt_target=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-Tshirt',transform=transforms)
test_prediction_dataset_womansweater_target=datasets.ImageFolder(root='./Database/Test_Database/TestClothingResources-WomanSweater',transform=transforms)


path_folder='./save_model/'
if not os.path.exists(path_folder):
    os.makedirs(path_folder)
if opt.model_type==1:
    if opt.AutoEncoder_Type==1:
        path= './save_model/model_depth_autoencoder.pth'
    elif opt.AutoEncoder_Type==2:
        path='./save_model/model_RGB_autoencoder.pth'
if opt.model_type==2:
    if opt.AutoEncoder_Type==1:
        path='./save_model/model_depth_autres.pth'
    elif opt.AutoEncoder_Type==2:
        path='./save_model/model_RGB_autres.pth'
path_classifier='./save_model/Classifier.pth'
if opt.step1_type==1:
    step1_path='./save_model/model_depth_autoencoder_step1.pth'
if opt.step1_type==2:
    step1_path='./save_model/model_RGB_autoencoder_step1.pth'
if opt.step1_type==3:
    step1_path='./save_model/model_depth_autres_step1.pth'
if opt.step1_type==4:
    step1_path='./save_model/model_RGB_autres_step1.pth'
if opt.step1_type==5:
    step1_path='./save_model/model_depth_sdae_step1.pth'
if opt.step1_type==6:
    step1_path='./save_model/model_RGB_sdae_step1.pth'

if opt.AutoEncoder_Type==1:
    path_noise='./save_model/model_depth_autoencoder_noise.pth'
    path_clean='./save_model/model_depth_autoencoder_clean.pth'
    path_second='./save_model/model_depth_autoencoder_second.pth'
    path_second_noise='./save_model/model_depth_autoencoder_second_noise.pth'
if opt.AutoEncoder_Type==2:
    path_noise='./save_model/model_depth_autoencoder_noise_coloured.pth'
    path_clean='./save_model/model_depth_autoencoder_clean_coloured.pth'
    path_second='./save_model/model_depth_autoencoder_second_coloured.pth'
    path_second_noise='./save_model/model_depth_autoencoder_second_noise_coloured.pth'

if opt.real_image==2:
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    if opt.model_type==1:
        if opt.AutoEncoder_Type==1:
            path= './save_model/model_depth_autoencoder_real.pth'
        elif opt.AutoEncoder_Type==2:
            path='./save_model/model_RGB_autoencoder_real.pth'
    if opt.model_type==2:
        if opt.AutoEncoder_Type==1:
            path='./save_model/model_depth_autres_real.pth'
        elif opt.AutoEncoder_Type==2:
            path='./save_model/model_RGB_autres_real.pth'
    path_classifier='./save_model/Classifier_real.pth'
    if opt.step1_type==1:
        step1_path='./save_model/model_depth_autoencoder_step1_real.pth'
    if opt.step1_type==2:
        step1_path='./save_model/model_RGB_autoencoder_step1_real.pth'
    if opt.step1_type==3:
        step1_path='./save_model/model_depth_autres_step1_real.pth'
    if opt.step1_type==4:
        step1_path='./save_model/model_RGB_autres_step1_real.pth'
    if opt.step1_type==5:
        step1_path='./save_model/model_depth_sdae_step1_real.pth'
    if opt.step1_type==6:
        step1_path='./save_model/model_RGB_sdae_step1_real.pth'

    if opt.AutoEncoder_Type==1:
        path_noise='./save_model/model_depth_autoencoder_noise_real.pth'
        path_clean='./save_model/model_depth_autoencoder_clean_real.pth'
        path_second='./save_model/model_depth_autoencoder_second_real.pth'
        path_second_noise='./save_model/model_depth_autoencoder_second_noise_real.pth'
    if opt.AutoEncoder_Type==2:
        path_noise='./save_model/model_depth_autoencoder_noise_coloured_real.pth'
        path_clean='./save_model/model_depth_autoencoder_clean_coloured_real.pth'
        path_second='./save_model/model_depth_autoencoder_second_coloured_real.pth'
        path_second_noise='./save_model/model_depth_autoencoder_second_noise_coloured_real.pth'

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
if opt.is_clean==1:
    data_size=len(cloth_dataset_c)
if opt.is_clean==3:
    data_size=len(cloth_dataset)
else:
    data_size=len(cloth_dataset_c)
data_size_original_textured=len(test_prediction_dataset_suit)
data_size_target_textured=len(test_prediction_dataset_tshirt_target)
n_test=int(data_size*0.2)
n_train=data_size-n_test
random_seed=42
indices=list(range(data_size))
indices_original_textured=list(range(data_size_original_textured))
indices_target_textured=list(range(data_size_target_textured))
shuffle_dataset=True
if shuffle_dataset:
    np.random.seed(seed)
    np.random.shuffle(indices)
    np.random.shuffle(indices_original_textured)
    np.random.shuffle(indices_target_textured)
if opt.dataset_type==1:
    train_indices=indices
if opt.dataset_type==2:
    train_indices=indices[0:n_train]
test_indices=indices[n_train:n_train+n_test]
test_prediction_indices=indices
test_prediction_indices_original_textured=indices_original_textured
test_prediction_indices_target_textured=indices_target_textured

train_sampler=SubsetRandomSampler(train_indices)
test_sampler=SubsetRandomSampler(test_indices)


train_lstm_sampler=SequentialSampler(train_indices)
test_lstm_sampler=SequentialSampler(test_indices)
test_prediction_sampler=SequentialSampler(test_prediction_indices)
test_original_textured_sampler=SequentialSampler(test_prediction_indices_original_textured)
test_target_textured_sampler=SequentialSampler(test_prediction_indices_target_textured)


model=LSTM.LSTM(input_feature=1024,n_layer=1,hidden_dim=1024)
model=model.to(device)
net_RGB=autocoder3.AutoEncoder(device)
net_depth=autocoder_depth.AutoEncoder(device)
model_lstm=LSTM_classifier.LSTM(input_feature=1024,n_layer=1,hidden_dim=1024,output_classifier=opt.n_class,hidden_classifier=128)
model_lstm=model_lstm.to(device)
classifier=Classifier.Classifier(device,output_feature=opt.n_class,hidden_feature=128)
classifier=classifier.to(device)
net_autoresdepth=Auto_Res.Auto_Res(in_chanels=1,hidden_layer=32)
net_autoresdepth=net_autoresdepth.to(device)
net_autoresRGB=Auto_Res.Auto_Res(in_chanels=3,hidden_layer=32)
net_autoresRGB=net_autoresRGB.to(device)
model_autres=LSTM_classifier_autres.LSTM(input_feature=1024,n_layer=1,hidden_dim=1024,output_classifier=opt.n_class,hidden_classifier=128)
model_autes=model_autres.to(device)
net_stackedDepth1=AutoEnc_Stacked_Level1.AutoEncoder(device)
net_stackedDepth0=AutoEnc_Stacked_Level0.AutoEncoder(device)
net_stackedDepth0=net_stackedDepth0.to(device)
net_stackedDepth1=net_stackedDepth1.to(device)
model_classifier_clean=LSTM_classifier_second.LSTM(input_feature=1024,n_layer=1,hidden_dim=1024,output_classifier=opt.n_class,hidden_classifier=128)
model_classifier_clean=model_classifier_clean.to(device)
net_stackedDepth0_coloured=AutoEnc_Stacked_Level0_Coloured.AutoEncoder(device)
net_stackedDepth1_coloured=AutoEnc_Stacked_Level1_Coloured.AutoEncoder(device)
net_stackedDepth0_coloured=net_stackedDepth0_coloured.to(device)
net_stackedDepth1_coloured=net_stackedDepth1_coloured.to(device)

if opt.extractor==1:
    net_opt=[net_depth]
elif opt.extractor==2:
    net_opt=[net_RGB]
elif opt.extractor==3:
    net_opt=[net_autoresRGB]
elif opt.extractor==4:
    net_opt=[net_autoresdepth]
elif opt.extractor==5:
    net_opt=[net_stackedDepth0]
    net_sec=[net_stackedDepth1]
elif opt.extractor==6:
    net_opt=[net_stackedDepth0_coloured]
    net_sec=[net_stackedDepth1_coloured]



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
cloth_lstm_loader_classifier=DataLoader(dataset=cloth_dataset_c,
                                   batch_size=opt.batch_size_for_prediction,
                                   sampler=train_sampler,
                                   num_workers=1)
cloth_lstm_test_loader=DataLoader( dataset=cloth_dataset,
                                   sampler=test_lstm_sampler,
                                   batch_size=opt.batch_size_for_prediction,
                                   num_workers=1)
test_prediction_loader=DataLoader(dataset=test_prediction_dataset,
                                  sampler=test_prediction_sampler,
                                  batch_size=opt.batch_size_for_prediction,
                                  num_workers=2)
test_prediction_suit=DataLoader(dataset=test_prediction_dataset_suit,
                                sampler=test_original_textured_sampler,
                                batch_size=100,
                                num_workers=2)
test_prediction_tshirt=DataLoader(dataset=test_prediction_dataset_tshirt,
                                sampler=test_original_textured_sampler,
                                batch_size=100,
                                num_workers=2)
test_prediction_womansweater=DataLoader(dataset=test_prediction_dataset_womansweater,
                                sampler=test_original_textured_sampler,
                                batch_size=100,
                                num_workers=2)

test_prediction_suit_target=DataLoader(dataset=test_prediction_dataset_suit_target,
                                sampler=test_target_textured_sampler,
                                batch_size=opt.batch_size_textured_target,
                                num_workers=2)
test_prediction_tshirt_target=DataLoader(dataset=test_prediction_dataset_tshirt_target,
                                sampler=test_target_textured_sampler,
                                batch_size=opt.batch_size_textured_target,
                                num_workers=2)
test_prediction_womansweater_target=DataLoader(dataset=test_prediction_dataset_womansweater_target,
                                sampler=test_target_textured_sampler,
                                batch_size=opt.batch_size_textured_target,
                                num_workers=2)

test_original_textured=[cloth_lstm_train_loader]                           
test_target_textured=[test_prediction_suit_target,test_prediction_tshirt_target,test_prediction_womansweater_target] 

epochs=opt.epochs

#############AutoEncoder Training##############

if __name__ == '__main__':
    print("We have our Marvel Initiatives Started")
    epoch=22
    epochs=10
    """
    clean_net=torch.load(path_clean)
    noise_net=torch.load(path_noise)
    second_clean=torch.load(path_second)
    clean_net=Discriminator_and_Generator_Training_clean.train(net_opt,cloth_loader,cloth_test_loader,opt.batch_size,epoch ,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(clean_net,path_clean)
    noise_net=Discriminator_and_Generator_Training_noise.train(net_opt,cloth_loader,cloth_test_loader,opt.batch_size,epoch ,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(noise_net,path_noise)
    second_clean=Discriminator_and_Generator_second_clean.train(noise_net,clean_net,net_sec,cloth_loader,cloth_test_loader,opt.batch_size,epoch ,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(second_clean,path_second)
    """
    """
    net=Discriminator_and_Generator_Training.train(net_opt,cloth_loader,cloth_test_loader,opt.batch_size,epochs,opt.lr,device,opt.AutoEncoder_Type)
    torch.save(net,path)
    """
#############LSTM Training#######################
    net=torch.load(path)
    """
    for n in range(len(clean_net)):
        frozon_and_free_Get.frozon_Param(clean_net[n])
    for n in range(len(noise_net)):
        frozon_and_free_Get.frozon_Param(noise_net[n])
    for n in range(len(second_clean)):
        frozon_and_free_Get.frozon_Param(second_clean[n])
    """
    for n in range(len(net)):
        frozon_and_free_Get.frozon_Param(net[n])
    model_one=LSTM_Training_Step1.train(model,epoch,cloth_lstm_train_loader,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type)
    """
    model_one=LSTM_Training_No_Label_Step1.train(model,epoch,cloth_lstm_train_loader,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type)
    torch.save(model_one,lstm_path)
    models=torch.load(step1_path)
    frozon_and_free_Get.frozon_Param(models)
    model=LSTM_Training_Label_Step.train(models,epochs,cloth_lstm_loader_c,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type,opt.index_column,path_step1_c,classifier)
    Classifier_Training=Classifier_Training.train(model,epochs,cloth_lstm_loader_classifier,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type,opt.index_column,path_classifier,classifier)
    classifier_trained=torch.load(path_classifier)
    frozon_and_free_Get.free_Param(classifier)
    model_LSTM=LSTM_Training_Step1_classifier.train(model,epoch,cloth_lstm_loader_c,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type,opt.index_column,path_step1_c,classifier)
    model_one_classifier=Classifier_Training.train(model,epoch,cloth_lstm_loader,net,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type,opt.index_column,path_step1_c,classifier)
    model_one_classifier_clean=LSTM_Training_Step1_clean.train(models,epochs,cloth_lstm_loader_c,noise_net,second_clean,opt.lr,opt.batch_size,device,opt.AutoEncoder_Type,opt.index_column,path_step1_c,classifier)
    model_one=torch.load(path_step1)
    model_one_classifier=torch.load(path_step1_c)
    frozon_and_free_Get.frozon_Param(model_one)
    frozon_and_free_Get.free_Param(model_one_classifier)
    Test_for_the_Next_Two_Frames.train(model_one,test_original_textured,net,device,opt.AutoEncoder_Type)
###########Thank you, Marvel ####################
###########I love you three thousand ############
###########Viva Marvel ##########################