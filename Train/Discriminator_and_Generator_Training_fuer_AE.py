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
from matplotlib import pyplot as plot
#########################
# Train Model for AutoEncoder-Classifier
#########################


def train(net,mario_lstm_loader,batch_size,epochs,learning_rate,device,AutoEncoder_Type,classifier):
    selfnoise=SelfNoise.Gaussian_Nosie()
    save_dir='./data/Recon_Image'
    save_autoencoder_figure='./data/AutoEncoderLoss'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_autoencoder_figure):
        os.mkdir(save_autoencoder_figure)
    optimiser_label=optim.Adam(classifier.parameters(),lr=learning_rate)
    tra_los_eps=[]
    tes_los_eps=[]
    tra_all_eps=[]
    tra_acc_epo=[]
    tes_acc_epo=[]
    epochs=int(epochs)
    net[0]=net[0].to(device)

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


    random.shuffle(inputs_random)
    inputs_randperm=inputs_random[0:int(len(inputs_random)*0.8)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.8):int(len(inputs_random))]

    item_time_train=int(len(inputs_randperm)/10)
    item_time_test=int(len(inputs_randperm_test)/10)

    for n_epochs in range(epochs):
        random.shuffle(inputs_randperm)
        for z in range(len(inputs_randperm)):
            sample=inputs_randperm[z]
            inputs=sample["picture"]
            Label=sample["label"]
            sequ_length=1
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            train_acc=0
            start_time=time.time()
            total_num=len(inputs)
            total_loss=[]
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            for n in range(len(inputs)):
                optimiser_label.zero_grad()
                input_noise=inputs_noise[n:n+1]
                label_w=Label[n]
                input_noise=input_noise.to(device)
                input_noise=Variable(input_noise)
                recon_batch_enc=net[0].encoder(input_noise)
                _,label_max=torch.max(label_w,1)
                recon_batch_l=classifier(recon_batch_enc)
                loss_fn=classifier.loss_fn(recon_batch_l,label_max)
                _,pre_label_max=torch.max(recon_batch_l,1)
                if (label_max==pre_label_max):
                    train_acc+=1
                total_loss.append(loss_fn.item())
                loss_fn.backward()
                optimiser_label.step()
                if((n+1)%(len(inputs)-4*sequ_length)==0) and ((z+1)%(item_time_train+1)==0):
                    print ('[Epoch:%d/%d][Batch:%d/%d][Train Loss:%f][Accuracy:%f][Duration:%f]'
                    %(n_epochs+1,epochs,n,len(inputs),mean_stat(total_loss),100*((train_acc)/total_num),time.time()-start_time),'Label is:',recon_batch_l)
                    start_time=time.time()
        avg_loss=mean_stat(total_loss)
        avg_accuracy=100*((train_acc)/total_num)
        tra_los_eps.append(avg_loss)
        tra_acc_epo.append(avg_accuracy)
        print (tra_los_eps[n_epochs])
        print (tra_acc_epo[n_epochs])

        random.shuffle(inputs_randperm_test)
        for z in range (len(inputs_randperm_test)):
            sample=inputs_randperm_test[z]
            inputs=sample['picture']
            Label=sample['label']
            sequ_length=1
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            total_loss=[]
            total_num=len(inputs)
            test_acc=0
            inputs=Variable(inputs)
            start_time=time.time()
            for n in range (len(inputs)):
                input_w=inputs_noise[n:n+1]
                recon_batch_enc=net[0].encoder(input_w)
                label_w=Label[n]
                _,label_max=torch.max(label_w,1)
                recon_batch_l=classifier(recon_batch_enc)
                _,pre_label_max=torch.max(recon_batch_l,1)
                loss_fn=classifier.loss_fn(recon_batch_l,label_max)
                if label_max==pre_label_max:
                    test_acc+=1
                total_loss.append(loss_fn.item())
                if((n+1)%(len(inputs)-4*sequ_length)==0) and ((z+1)%(item_time_test+1)==0):
                    print("[Epoch:%d/%d][Batch:%d/%d][train_test_loss:%f][accuracy: %f][Duration: %f]"
                    %(n_epochs+1,epochs,i,len(inputs),mean_stat(total_loss),100*((test_acc)/total_num),time.time()-start_time))
                start_time=time.time()
        avg_loss=mean_stat(total_loss)
        avg_accuracy=100*((test_acc)/total_num)
        tes_los_eps.append(avg_loss)
        tes_acc_epo.append(avg_accuracy)
        tra_all_eps.append(n_epochs)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':tra_all_eps,'train':tra_los_eps,'test':tes_los_eps,'train_acc':tra_acc_epo,'test_acc':tes_acc_epo})
    ax_t=plot.figure()
    ax_t.add_subplot(111)
    ax=plot.subplot()
    ax.plot('x','train', data=df,color='purple',label='train')
    ax.plot('x','test', data=df,color='blue',linestyle='dashed',label='test')
    ax2=ax.twinx()
    ax2.plot('x','train_acc', data=df,color='yellow',label='Train_Acc')
    ax2.plot('x','test_acc', data=df,color='red',label='Test_Acc',linestyle='dashed')
    ax2.set_ylabel('Accuracy Rate(%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy_loss')
    plot.title("Loss Function for Auto Encoder of "+AutoEncoder_Type_Name)
    plot.grid(True)
    plot.legend(loc='upper right')
    plot.savefig(os.path.join(save_autoencoder_figure,'Loss_AutoEncoder_%d_General.png'%AutoEncoder_Type),dip=100)
    print("The Training is Finished!")
    return classifier