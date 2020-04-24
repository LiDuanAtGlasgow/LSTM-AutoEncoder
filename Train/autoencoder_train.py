# pylint: skip-file
import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plot
import pandas as pand
from torchvision.utils import save_image
import time
from torch.autograd import Variable
import pandas as pd
from Tool import SelfNoise

#########################
# Train Model for AutoEncoder
#########################


def train(net,train_loader,val_loader,test_loader,batch_size,epochs,learning_rate,device,AutoEncoder_Type):
    save_dir='./data/Recon_Image'
    save_autoencoder_figure='./data/AutoEncoderLoss'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_autoencoder_figure):
        os.mkdir(save_autoencoder_figure)
    n_batches=len(train_loader)
    train_start_time=time.time()
    optimiser=optim.Adam(net.parameters(),lr=learning_rate)
    tra_los_eps=[]
    val_los_eps=[]
    all_eps=[]
    scheduler=optim.lr_scheduler.StepLR(optimiser,step_size=4,gamma=0.9,last_epoch=-1)

    for n_epochs in range(epochs):
        every_time=int(n_batches/10)
        tra_los_round=0.00
        net=net.to(device)
        for i,input_ in enumerate(train_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            optimiser.zero_grad()
            recon_batch=net(inputs)
            loss=net.loss_function(recon_batch,inputs)
            loss.backward()
            optimiser.step()
            tra_los_round+=loss.item()
            if (i+1)%(every_time+1)==0:
                print("[Train][Epochs: %d/%d][Batch: %d/%d][Train Loss: %f][Duration: %f]"
                %(n_epochs+1,epochs,i,len(train_loader),tra_los_round/(every_time),time.time()-train_start_time))
                train_start_time=time.time()
                n=min(len(inputs),8)
                comparison=torch.cat([inputs[:n],recon_batch[:n]])
                save_image(comparison.cpu(),os.path.join(save_dir,'%d.png'%(time.time())))
        avg_loss=tra_los_round/(len(train_loader)*batch_size)
        tra_los_eps.append(avg_loss)
        print (tra_los_eps[n_epochs])

        val_batch_size=len(val_loader)
        every_time=int(val_batch_size/10)
        val_los_round=0.00
        start_time=time.time()
        for i, input_ in enumerate(val_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            inputs=Variable(inputs)
            inputs=inputs.to(device)
            recon_batch=net(inputs)
            loss=net.loss_function(recon_batch,inputs)
            val_los_round+=loss.item()
            if (i+1)%(every_time+1)==0:
                print("[Val][Epoch:%d/%d][Batch:%d/%d][Val Loss:%f][Duration: %f]"
                %(n_epochs+1,epochs,i,len(val_loader),val_los_round/len(val_loader),time.time()-start_time))
                start_time=time.time()
                n=min(len(inputs),8)
                comparison=torch.cat([inputs[:n],recon_batch[:n]])
                save_image(comparison.cpu(),os.path.join(save_dir,'%d.png'%(time.time())))
        avg_loss=val_los_round/(len(val_loader)*batch_size)
        val_los_eps.append(avg_loss)
        scheduler.step()
        all_eps.append(n_epochs+1)
    
    test_batch_size=len(test_loader)
    every_time=int(test_batch_size/10)
    test_los_round=0.00
    test_start_time=time.time()
    for i, input_ in enumerate(test_loader):
        if AutoEncoder_Type==1:
            inputs=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            inputs=input_['Image']
        inputs=Variable(inputs)
        inputs=inputs.to(device)
        recon_batch=net(inputs)
        loss=net.loss_function(recon_batch,inputs)
        test_los_round+=loss.item()
        if (i+1)%(every_time+1)==0:
            print("[Test][Batch:%d/%d][Test Loss:%f][Duration: %f]"
            %(i,len(test_loader),test_los_round/len(test_loader),time.time()-test_start_time))
            n=min(len(inputs),8)
            comparison=torch.cat([inputs[:n],recon_batch[:n]])
            save_image(comparison.cpu(),os.path.join(save_dir,'%d.png'%(time.time())))
            test_start_time=time.time()
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':all_eps,'train':tra_los_eps,'val':val_los_eps})
    asplot=plot.figure()
    asplot.add_subplot(111)
    plot.plot('x','train', data=df,color='red',label='train')
    plot.plot('x','val', data=df,color='blue',linestyle='dashed',label='test')
    plot.xlabel('Epoch')
    plot.ylabel('AutoEncoder Loss')
    plot.title("Loss Function for Auto Encoder of "+AutoEncoder_Type_Name)
    plot.grid(True)
    plot.legend(loc='upper right')
    plot.savefig(os.path.join(save_autoencoder_figure,'Loss_AutoEncoder_%d_General.png'%AutoEncoder_Type),dip=100)
    print("The Training is Finished!")
    return net