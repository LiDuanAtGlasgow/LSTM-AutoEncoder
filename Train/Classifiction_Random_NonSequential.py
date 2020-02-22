import torch
import torch.random as random_torch
import random
import statistics as statistics
from statistics import mean as mean_stat
from statistics import stdev as stdev
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
from Model import LSTM
import numpy as np
import pandas as pd
from Tool import SelfNoise
from torch.autograd import Variable
from Model import Classifier
from sklearn.preprocessing import LabelBinarizer
import csv
from matplotlib import pyplot as plt
#########################
# Train Model for AutoEncoder-Classifier
#########################


def train(net,data_loader,epochs,lr,device,AutoEncoder_Type,Classifier):
    save_pic='./data/classification/randomc_image/'
    if not os.path.exists(save_pic):
        os.mkdir(save_pic)
    save_fig='./data/classification/randomc_figure/'
    if not os.path.exists(save_fig):
        os.mkdir(save_fig)
    save_model='./data/classification/randomc_model/'
    if not os.path.exists(save_model):
        os.mkdir(save_model)
    path_randomc='./data/classification/%f.pth'%time.time()
    selfnoise=SelfNoise.Gaussian_Nosie()
    optimiser_label=optim.Adam(Classifier.parameters(),lr=lr)
    net=net[0].to(device)

    """
    Data Preperation Period
    """
    inputs_random=[]
    start_time=time.time()
    
    for t,input_ in enumerate(data_loader):
        if AutoEncoder_Type==1:
            image=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            image=input_['Image']
        label=input_['Label']
        item_time=int(len(data_loader)/10)
        if (t+1)%(item_time+1)==1:
            print("Batch_Size[%d/%d]Duration[%f]"
                %(t+1,len(data_loader),time.time()-start_time))
            start_time=time.time()
        samples={"picture":image,"label":label}
        inputs_random.append(samples)
    encoder=LabelBinarizer()
    exchange_matrix=[] 
    for i in range(len(inputs_random)):
        for t in range(len(inputs_random[i]['label'])):
            exchange_matrix.append(inputs_random[i]['label'][t])
    exchange_matrix_enc=encoder.fit_transform(exchange_matrix)
    for i in range(len(inputs_random)):
        for t in range(len(inputs_random[i]['label'])):
            inputs_random[i]['label'][t]=exchange_matrix_enc[i*len(inputs_random[i]['label'])+t]
            inputs_random[i]['label'][t]=torch.tensor([inputs_random[i]['label'][t]],device=device).long()
    print('Data Preparation Ends')


    random.shuffle(inputs_random)
    inputs_randperm_train=inputs_random[0:int(len(inputs_random)*0.8)]
    inputs_randperm_val=inputs_random[int(len(inputs_random)*0.8):int(len(inputs_random)*0.9)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.9):len(inputs_random)]

    batch_train=int(len(inputs_randperm_train)/10)
    batch_val=int(len(inputs_randperm_val)/10)

    train_epoch=[]
    val_epoch=[]
    x_epoch=[]

    for n_epoch in range(epochs):
        random.shuffle(inputs_randperm_train)
        start_time=time.time()
        train_loss=[]
        for z in range(len(inputs_randperm_train)):
            optimiser_label.zero_grad()
            sample=inputs_randperm_train[z]
            inputs=sample["picture"]
            label=sample["label"]
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            input_enc=net.encoder(inputs_noise)
            pred_label=Classifier(input_enc)
            loss_fn=Classifier.loss_fn(pred_label,label)
            loss_fn.backward()
            train_loss.append(loss_fn.item())
            if (z+1)%batch_train==1:
                print ("[Train Part][Epoch:%d/%d][Loss:%f][Duration:%f]"
                %(n_epoch+1,epochs,mean_stat(train_loss),time.time()-start_time),"pred_label is:",pred_label)
                start_time=time.time()
        train_mean=mean_stat(train_loss)
        train_epoch.append(train_mean)
        print ("train_mean is:",train_mean)

        random.shuffle(inputs_randperm_val)
        start_time=time.time()
        val_loss=[]
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            label=sample['label']
            inputs_noise=selfnoise(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            input_enc=net.encoder(inputs_noise)
            pred_label=Classifier(input_enc)
            loss_fn=Classifier.loss_fn(pred_label,label)
            val_loss.append(loss_fn.item())
            if (z+1)%batch_val==1:
                print ("[Test Part][Epoch:%d/%d][Loss is:%f][Duration:%f]"
                %(n_epoch+1,epochs,mean_stat(val_loss),time.time()-start_time),"pred_lable is:",pred_label)
                start_time=time.time()
                n=min(len(inputs_noise),8)
                save_image(inputs_noise[:n],os.path.join(save_pic,"%f.png"%time.time()))
        val_mean=mean_stat(val_loss)
        val_epoch.append(val_mean)
        x_epoch.append(n_epoch)
        print ("val_mean is:",val_mean)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':x_epoch,'train':train_epoch,'val':val_epoch})
    ax_t=plt.figure()
    ax_t.add_subplot(111)
    ax=plt.subplot()
    ax.plt('x','train', data=df,color='purple',label='train_loss')
    ax.plt('x','test', data=df,color='blue',linestyle='dashed',label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy_loss')
    plt.title("Cross-Entropy and Accuarcy of "+AutoEncoder_Type_Name)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_fig,"%f.png"%time.time()),dip=100)
    print ("train_epoch is:",train_epoch)
    print ("val_epoch is:",val_epoch)
    plt.show()
    torch.save(Classifier,path_randomc)
    print("The Training is Finished!")
    return Classifier