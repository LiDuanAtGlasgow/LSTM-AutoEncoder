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



##############################
#####Tarin Model(LSTM)########
##############################
def train(models,epochs,mario_lstm_loader,net,lr,batch_size,device,AutoEncoder_Type):
    average_train_loss_set=0.00
    average_test_loss_set=0.00
    selfnoise=SelfNoise.Gaussian_Nosie()
    train_start_time=time.time()
    save_image_lstm='./data/lstm_result/'
    if not os.path.exists(save_image_lstm):
       os.makedirs(save_image_lstm)
    save_model_path='./save_model/'
    if not os.path.exists(save_model_path):
       os.makedirs(save_model_path)
    path_step1='./save_model/model_step1.pth'
    loss_fn_step1 = torch.nn.MSELoss(size_average=False)
    optimiser_step1 = torch.optim.Adam(models.parameters(), lr=lr)
    lossstatistics=[]
    training_epoch=[]
    lossstatistics_train=[]
    epochs=int(epochs)
    net=net[0]
    save_figure_lstm='./data/figure/'
    if not os.path.exists(save_figure_lstm):
        os.makedirs(save_figure_lstm)
    
    """
    Data Preperation Period
    """
    inputs_random=[]
    start_time=time.time()
    
    for t,input_ in enumerate(mario_lstm_loader):
        if AutoEncoder_Type==1:
            image=input_[0][:,0:1,:,:]
        if AutoEncoder_Type==2:
            image=input_[0]
        item_time=int(len(mario_lstm_loader)/10)
        if (t+1)%(item_time+1)==1:
            print("Batch_Size[%d/%d]Duration[%f]"
                %(t+1,len(mario_lstm_loader),time.time()-start_time))
            start_time=time.time()
        inputs_random.append(image)
    """
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
    """

    random.shuffle(inputs_random)
    inputs_randperm=inputs_random[0:int(len(inputs_random)*0.80)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.80):int(len(inputs_random))]
    
    for n in range(epochs):
        random.shuffle(inputs_randperm)
        total_loss=[]
        for z in range(len(inputs_randperm)):
            inputs=inputs_randperm[z]
            sequ_length=1
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            item_time=int(len(inputs_randperm)/10)
            for i in range(len(inputs)-3*sequ_length):
                optimiser_step1.zero_grad()
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=models(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                inputs_decor=net.decoder(inputs_out)
                loss_function=loss_fn_step1(target,inputs_decor)
                loss_function.backward()
                total_loss.append(loss_function.item())
                optimiser_step1.step()
                if ((i+1)%(len(inputs)-4*sequ_length)==1)and((z+1)%(item_time+1)==1):
                    print("[Epochs:%d/%d][Train Time:%d/%d][Duration:%f][Train Loss:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm),time.time()-train_start_time,mean_stat(total_loss)))
                    train_start_time=time.time()
        average_train_loss_set=mean_stat(total_loss)  
        lossstatistics_train.append(average_train_loss_set)
        
        random.shuffle(inputs_randperm_test)
        total_loss=[]
        for z in range (len(inputs_randperm_test)):
            inputs=inputs_randperm_test[z]
            sequ_length=1
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            item_time=int(len(inputs_randperm_test)/10)
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=models(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                inputs_decor=net.decoder(inputs_out)
                loss_comparison=loss_fn_step1(target,inputs_decor)
                total_loss.append(loss_comparison.item())
                if((i+1)%(len(inputs)-4*sequ_length)==1) and ((z+1)%(item_time+1)==1):
                    print("[Epochs:%d/%d][Test Time:%d/%d][Duration:%f][Test Loss:%f]"
                        %(n+1,epochs,t+1,len(inputs_randperm_test),time.time()-train_start_time,mean_stat(total_loss)))
                    train_start_time=time.time()
                cat=torch.cat([input_pred,target,inputs_decor])
                save_image(cat.cpu(),os.path.join(save_image_lstm,"%d_%d.png"%(time.time(),i)))
        average_test_loss_set=mean_stat(total_loss)
        lossstatistics.append(average_test_loss_set)
        training_epoch.append(n)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':training_epoch,'train':lossstatistics_train,'test':lossstatistics})
    asplot=plt.figure()
    asplot.add_subplot(111)
    plt.plot('x','train', data=df,color='red',label='train')
    plt.plot('x','test', data=df,color='blue',label='test',linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('LSTM Loss')
    plt.title('Loss Function for LSTM of '+AutoEncoder_Type_Name)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_figure_lstm,'LossFunction_Textured_%d.png'%AutoEncoder_Type),dip=100)
    torch.save(models,path_step1)
    print ('train loss is:',lossstatistics_train)
    print ('test loss is:',lossstatistics)
    print ('The Training is finished!')
    return models


