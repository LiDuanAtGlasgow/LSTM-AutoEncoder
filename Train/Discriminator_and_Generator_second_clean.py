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
# Train Model for Genera-Discrim Machine
#########################


def train(noise_net,clean_net,second_net,train_loader,test_loader,batch_size,epochs,learning_rate,device,AutoEncoder_Type):
    selfnoise=SelfNoise.Gaussian_Nosie()
    selfnoise=selfnoise.to(device)
    save_dir='./data/Recon_Image'
    save_autoencoder_figure='./data/AutoEncoderLoss'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_autoencoder_figure):
        os.mkdir(save_autoencoder_figure)
    n_batches=len(train_loader)
    train_start_time=time.time()
    optimiser=optim.Adam(second_net[0].parameters(),lr=learning_rate)
    tra_los_eps=[]
    tes_los_eps=[]
    tra_all_eps=[]
    epochs=int(epochs)
    stdev=0.06

    for n_epochs in range(epochs):
        every_time=int(n_batches/10)
        tra_los_round=0.00
        for i,input_ in enumerate(train_loader):
            if AutoEncoder_Type==1:
                inputs=input_[0][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_[0]
            first_level_inputs_noise=selfnoise.forward(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            first_level_inputs_noise=first_level_inputs_noise.to(device)
            first_level_inputs_noise=Variable(first_level_inputs_noise)
            optimiser.zero_grad()
            clean_inputs_noise=clean_net[0].encoder(inputs)
            first_level_inputs_noise_part=torch.randn(clean_inputs_noise.size())*stdev
            first_level_inputs_noise_part=first_level_inputs_noise_part.to(device)
            first_level_inputs_noise=clean_inputs_noise+first_level_inputs_noise_part
            recon_batch=second_net[0](first_level_inputs_noise)
            loss=second_net[0].loss_function(recon_batch,clean_inputs_noise)
            loss.backward()
            optimiser.step()
            tra_los_round+=loss.item()
            if (i+1)%(every_time+1)==0:
                print("[Epochs: %d/%d][Batch: %d/%d][Train_loss: %f][Duration: %f]"
                %(n_epochs+1,epochs,i,len(train_loader),tra_los_round/(every_time),time.time()-train_start_time))
                train_start_time=time.time()
        avg_loss=tra_los_round/(len(train_loader)*batch_size)
        tra_los_eps.append(avg_loss)
        print (tra_los_eps[n_epochs])

        test_batch_size=len(test_loader)
        every_time=int(test_batch_size/10)
        tes_los_round=0.00
        test_start_time=time.time()
        for i, input_ in enumerate(test_loader):
            second_net[0].eval()
            if AutoEncoder_Type==1:
                inputs=input_[0][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_[0]
            inputs=Variable(inputs)
            first_level_inputs_noise=selfnoise.forward(inputs)
            inputs=inputs.to(device)
            first_level_inputs_noise=first_level_inputs_noise.to(device)
            first_level_inputs_noise=Variable(first_level_inputs_noise)
            clean_inputs_noise=clean_net[0].encoder(inputs)
            first_level_inputs_noise_part=torch.randn(clean_inputs_noise.size())*stdev
            first_level_inputs_noise_part=first_level_inputs_noise_part.to(device)
            first_level_inputs_noise=clean_inputs_noise+first_level_inputs_noise_part
            recon_batch=second_net[0](first_level_inputs_noise)
            loss=second_net[0].loss_function(recon_batch,clean_inputs_noise)
            tes_los_round+=loss.item()
            if (i+1)%(every_time+1)==0:
                print("[Epoch:%d/%d][Batch:%d/%d][train_test_loss:%f][Duration: %f]"
                %(n_epochs+1,epochs,i,len(test_loader),tes_los_round/len(test_loader),time.time()-test_start_time))
                test_start_time=time.time()
        avg_loss=tes_los_round/(len(test_loader)*batch_size)
        tes_los_eps.append(avg_loss)
        tra_all_eps.append(n_epochs)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':tra_all_eps,'train':tra_los_eps,'test':tes_los_eps})
    asplot=plot.figure()
    asplot.add_subplot(111)
    plot.plot('x','train', data=df,color='red',label='train')
    plot.plot('x','test', data=df,color='blue',linestyle='dashed',label='test')
    plot.xlabel('Epoch')
    plot.ylabel('AutoEncoder Loss')
    plot.title("Loss Function for Auto Encoder of "+AutoEncoder_Type_Name)
    plot.grid(True)
    plot.legend(loc='upper right')
    plot.savefig(os.path.join(save_autoencoder_figure,'Loss_AutoEncoder_%d_second_clean.png'%AutoEncoder_Type),dip=100)
    print("The Training is Finished!")
    return second_net