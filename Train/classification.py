import torch
import random
from statistics import mean as mean_stat
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
import pandas as pd
from Tool import SelfNoise
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer

def train(model,epochs,mario_lstm_loader,net,lr,batch_size,device,AutoEncoder_Type):
    selfnoise=SelfNoise.Gaussian_Nosie()
    train_start_time=time.time()
    epoch=[]
    net=net[0]
    
    """
    Data Preperation Period
    """
    inputs_random=[]
    start_time=time.time()
    
    for t,input_ in enumerate(mario_lstm_loader):
        if AutoEncoder_Type==1:
            image=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            image=input_['Image']
        label=input_['Label']
        item_time=int(len(mario_lstm_loader)/10)
        if (t+1)%(item_time+1)==1:
            print("Batch_Size[%d/%d]Duration[%f]"
                %(t+1,len(mario_lstm_loader),time.time()-start_time))
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
    print("the encoding and from numpy tensor is finished!")

    start_time=time.time()
    random.shuffle(inputs_random)
    inputs_randperm_train=inputs_random[0:int(len(inputs_random)*0.80)]
    inputs_randperm_val=inputs_random[int(len(inputs_random)*0.80):int(len(inputs_random)*0.90)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.90):int(len(inputs_random))]
    sequ_length=1
    
    for n in range(epochs):

        random.shuffle(inputs_randperm_train)
        for z in range(len(inputs_randperm_train)):
            sample=inputs_randperm_train[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            item_time=int(len(inputs_randperm_train)/10)
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs_noise[i:i+3*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                if ((i+1)%(len(inputs)-4*sequ_length)==1)and((z+1)%(item_time+1)==1):
                    print("[Epochs:%d/%d][Train Time:%d/%d][Duration:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm_train),time.time()-train_start_time))
                    train_start_time=time.time()
        

        
        random.shuffle(inputs_randperm_val)
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            item_time=int(len(inputs_randperm_test)/10)
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs_noise[i:i+3*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                if((i+1)%(len(inputs)-4*sequ_length)==1) and ((z+1)%(item_time+1)==1):
                    print("[Epochs:%d/%d][Test Time:%d/%d][Duration:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm_val),time.time()-train_start_time,mean_stat(ssim_stat),mean_stat(mse_stat)))
                    train_start_time=time.time()

        epoch.append(n)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':epoch})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.plot('x', data=df,color='red',label='train_ssim')
    sbplt1.plot('x', data=df,color='blue',label='test_ssim',linestyle='dashed')
    sbplt1.set_xlabel('Epoch')
    plt.legend(loc='upper left')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x', data=df,color='green',label='train_mse')
    sbplt2.plot('x', data=df,color='yellow',label='test_mse',linestyle='dashed')
    plt.legend(loc='upper right')
    plt.grid(True)
    


