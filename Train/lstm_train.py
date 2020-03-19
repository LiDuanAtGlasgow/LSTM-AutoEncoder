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
from Tool import pytorch_ssim as ssim_loss



##############################
#####Tarin Model(LSTM)########
##############################
def train(models,epochs,mario_lstm_loader,net,lr,batch_size,device,AutoEncoder_Type):
    selfnoise=SelfNoise.Gaussian_Nosie()
    train_start_time=time.time()
    save_image_lstm='./data/lstm_result/'
    if not os.path.exists(save_image_lstm):
       os.makedirs(save_image_lstm)
    save_model_path='./save_model/'
    if not os.path.exists(save_model_path):
       os.makedirs(save_model_path)
    path_step1='./save_model/model_%f_step1.pth'%(time.time())
    optimiser_step1 = torch.optim.Adam(models.parameters(), lr=lr)
    mse_train=[]
    mse_val=[]
    epoch=[]
    ssim_train=[]
    ssim_val=[]
    net=net
    save_figure_lstm='./data/figure/'
    if not os.path.exists(save_figure_lstm):
        os.makedirs(save_figure_lstm)
    ssim=ssim_loss.SSIM()
    mse=nn.MSELoss(reduction='sum')
    scheduler=optim.lr_scheduler.StepLR(optimiser_step1,step_size=4,gamma=0.9,last_epoch=-1)
    
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
    inputs_randperm=inputs_random[0:int(len(inputs_random)*0.80)]
    inputs_randperm_val=inputs_random[int(len(inputs_random)*0.80):int(len(inputs_random)*0.90)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.90):len(inputs_random)]
    
    for n in range(epochs):
        ssim_stat=[]
        mse_stat=[]
        for z in range(len(inputs_randperm)):
            sample=inputs_randperm[z]
            inputs=sample['picture']
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
                ssim_optim=-ssim(target,inputs_decor)
                mse_optim=mse(target,inputs_decor)
                mse_optim.backward()
                ssim_value=-ssim_optim.item()
                mse_value=mse_optim.item()
                ssim_stat.append(ssim_value)
                mse_stat.append(mse_value)
                optimiser_step1.step()
                if ((i+1)%(len(inputs)-4*sequ_length)==1)and((z+1)%(item_time+1)==1):
                    print("[Train][Epochs:%d/%d][Train Time:%d/%d][Duration:%f][SSIM:%f][MSE:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm),time.time()-train_start_time,mean_stat(ssim_stat),mean_stat(mse_stat)))
                    train_start_time=time.time()
        ssim_average_train=mean_stat(ssim_stat)  
        ssim_train.append(ssim_average_train)
        mse_average_train=mean_stat(mse_stat)
        mse_train.append(mse_average_train)

        ssim_stat=[]
        mse_stat=[]
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            sequ_length=1
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            item_time=int(len(inputs_randperm_val)/10)
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=models(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                inputs_decor=net.decoder(inputs_out)
                ssim_out=-ssim(target,inputs_decor)
                ssim_value=-ssim_out.item()
                ssim_stat.append(ssim_value)
                mse_out=mse(target,inputs_decor)
                mse_value=mse_out.item()
                mse_stat.append(mse_value)
                if((i+1)%(len(inputs)-4*sequ_length)==1) and ((z+1)%(item_time+1)==1):
                    print("[Val][Epochs:%d/%d][Test Time:%d/%d][Duration:%f][SSIM:%f][MSE:%d]"
                        %(n+1,epochs,z+1,len(inputs_randperm_val),time.time()-train_start_time,mean_stat(ssim_stat),mean_stat(mse_stat)))
                    train_start_time=time.time()
        ssim_average_val=mean_stat(ssim_stat)
        ssim_val.append(ssim_average_val)
        mse_average_val=mean_stat(mse_stat)
        mse_val.append(mse_average_val)
        scheduler.step()
        epoch.append(n)
    ssim_stat=[]
    mse_stat=[]
    for z in range (len(inputs_randperm_test)):
        sample=inputs_randperm_test[z]
        inputs=sample['picture']
        sequ_length=1
        inputs=inputs.to(device)
        inputs=Variable(inputs)
        item_time=int(len(inputs_randperm_val)/10)
        for i in range(len(inputs)-3*sequ_length):
            input_pred=inputs[i:i+3*sequ_length]
            target=inputs[i+3*sequ_length:i+4*sequ_length]
            inputs_encor=net.encoder(input_pred)
            inputs_lstm=models(inputs_encor,device)
            inputs_out=inputs_lstm.view(-1,64,32,32)
            inputs_decor=net.decoder(inputs_out)
            ssim_out=-ssim(target,inputs_decor)
            ssim_value=-ssim_out.item()
            ssim_stat.append(ssim_value)
            mse_out=mse(target,inputs_decor)
            mse_value=mse_out.item()
            mse_stat.append(mse_value)
            if((i+1)%(len(inputs)-4*sequ_length)==1) and ((z+1)%(item_time+1)==1):
                print("[Test][Test Time:%d/%d][Duration:%f][SSIM:%f][MSE:%d]"
                    %(z+1,len(inputs_randperm_test),time.time()-train_start_time,mean_stat(ssim_stat),mean_stat(mse_stat)))
                train_start_time=time.time()
                cat=torch.cat([input_pred,target,inputs_decor])
                save_image(cat.cpu(),os.path.join(save_image_lstm,"%f.png"%time.time()))
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':epoch,'train_ssim':ssim_train,'val_ssim':ssim_val,'train_mse':mse_train,'val_mse':mse_val})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.plot('x','train_ssim', data=df,color='red',label='train_ssim')
    sbplt1.plot('x','val_ssim', data=df,color='blue',label='val_ssim',linestyle='dashed')
    sbplt1.set_xlabel('Epoch')
    sbplt1.set_ylabel('SSIM Loss')
    plt.legend(loc='upper left')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x','train_mse', data=df,color='green',label='train_mse')
    sbplt2.plot('x','val_mse', data=df,color='yellow',label='val_mse',linestyle='dashed')
    sbplt2.set_ylabel('MSE Loss')
    plt.legend(loc='upper right')
    plt.title('Loss for LSTM of '+AutoEncoder_Type_Name)
    plt.grid(True)
    plt.savefig(os.path.join(save_figure_lstm,'LSTM_%f.png'%(time.time())),dip=100)
    torch.save(models,path_step1)
    print ('ssim train loss is:',ssim_train)
    print ('ssim test loss is:',ssim_val)
    print ('mse train loss is:',mse_train)
    print ('mse test loss is:',mse_val)
    print ('Contragts! Finished! :)')
    return models


