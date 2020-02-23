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
    path_randomc='./data/classification/randomc_model/%f.pth'%time.time()
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

    batch_train=int(len(inputs_randperm_train)/10)+1
    batch_val=int(len(inputs_randperm_val)/10)+1

    train_epoch=[]
    val_epoch=[]
    x_epoch=[]
    acc_train=[]
    acc_val=[]

    for n_epoch in range(epochs):
        random.shuffle(inputs_randperm_train)
        start_time=time.time()
        train_loss=[]
        train_acc=0
        for z in range(len(inputs_randperm_train)):
            optimiser_label.zero_grad()
            sample=inputs_randperm_train[z]
            inputs=sample["picture"]
            inputs_label=sample["label"]
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            for i in range (len(inputs_noise)):
                image=inputs_noise[i:i+1]
                label=inputs_label[i]
                image_enc=net.encoder(image)
                pred_label=Classifier(image_enc)
                _,label_index=torch.max(label,1)
                loss_fn=Classifier.loss_fn(pred_label,label_index)
                loss_fn.backward()
                optimiser_label.step()
                train_loss.append(loss_fn.item())
                if (z+1)%batch_train==1 and (i+1)%(len(inputs_noise))==1:
                    print ("[Train Part][Epoch:%d/%d][Loss:%f][Duration:%f]"
                    %(n_epoch+1,epochs,mean_stat(train_loss),time.time()-start_time),"pred_label is:",pred_label)
                    start_time=time.time()
                _,pred_index=torch.max(pred_label,1)
                if label_index==pred_index:
                    train_acc+=1
        train_mean=mean_stat(train_loss)
        train_epoch.append(train_mean)
        total_number=len(inputs_randperm_train)*len(inputs_noise)
        acc_rate=(train_acc/total_number)*100
        acc_train.append(acc_rate)
        print ("train_mean is:",train_mean)
        print ("acc_rate is:",acc_rate)

        random.shuffle(inputs_randperm_val)
        start_time=time.time()
        val_loss=[]
        val_acc=0
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            inputs_label=sample['label']
            inputs_noise=selfnoise(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            for i in range(len(inputs_noise)):
                image=inputs_noise[i:i+1]
                label=inputs_label[i]
                image_enc=net.encoder(image)
                pred_label=Classifier(image_enc)
                _,label_index=torch.max(label,1)
                loss_fn=Classifier.loss_fn(pred_label,label_index)
                val_loss.append(loss_fn.item())
                if (z+1)%batch_val==1 and (i+1)%(len(inputs_noise))==1:
                    print ("[Test Part][Epoch:%d/%d][Loss is:%f][Duration:%f]"
                    %(n_epoch+1,epochs,mean_stat(val_loss),time.time()-start_time),"pred_lable is:",pred_label)
                    start_time=time.time()
                    save_image(image.cpu(),os.path.join(save_pic,"%f.png"%time.time()))
                _,pred_index=torch.max(pred_label,1)
                if label_index==pred_index:
                    val_acc+=1
        val_mean=mean_stat(val_loss)
        val_epoch.append(val_mean)
        x_epoch.append(n_epoch)
        total_number=len(inputs_randperm_val)*len(inputs_noise)
        acc_rate=(val_acc/total_number)*100
        acc_val.append(acc_rate)
        print ("val_mean is:",val_mean)
        print ("acc_rate is:",acc_rate)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':x_epoch,'train_loss':train_epoch,'val_loss':val_epoch, "acc_train":acc_train,"acc_val":acc_val})
    ax_t=plt.figure()
    ax_t.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.plot('x','train_loss', data=df,color='red',linestyle="-.",label='train_loss')
    sbplt1.plot('x','val_loss', data=df,color='blue',linestyle='--',label='val_loss')
    sbplt1.set_xlabel('Epoch')
    sbplt1.set_ylabel('Entropy_loss')
    plt.legend(loc='upper right')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x','acc_train', data=df,color='yellow',linestyle="-.",label='acc_train')
    sbplt2.plot('x','acc_val', data=df,color='green',linestyle='--',label='acc_val')
    sbplt2.set_ylabel('Accuracy (%)')
    plt.legend(loc='upper left')
    plt.title("Cross-Entropy and Accuarcy of "+AutoEncoder_Type_Name)
    plt.grid(True)
    plt.savefig(os.path.join(save_fig,"%f.png"%time.time()),dip=100)
    print ("train_epoch is:",train_epoch)
    print ("val_epoch is:",val_epoch)
    print ("acc_train is:",acc_train)
    print ("acc_val is:",acc_val)
    plt.show()
    torch.save(Classifier,path_randomc)
    print("The Training is Finished!")
    return Classifier