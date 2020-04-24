# pylint: skip-file
import torch
from statistics import mean as mean_stat
import os
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
import pandas as pd
from torch.autograd import Variable

##############################
#####Tarin Model(LSTM)########
##############################
def train(model,epochs,train_loader,val_loader,test_loader,net,lr,device,AutoEncoder_Type):
    train_start_time=time.time()
    save_memo_images='./data/memo/images'
    if not os.path.exists(save_memo_images):
        os.makedirs(save_memo_images)
    optimiser_step1 = torch.optim.Adam(model.parameters(), lr=lr)
    memo_train=[]
    memo_val=[]
    epoch=[]
    net=net
    save_memo_figure='./data/memo/figures'
    if not os.path.exists(save_memo_figure):
        os.makedirs(save_memo_figure)
    scheduler=optim.lr_scheduler.StepLR(optimiser_step1,step_size=4,gamma=0.1,last_epoch=-1)
    
    for n in range(epochs):
        memo_stat=[]
        for t,input_ in enumerate(train_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            sequ_length=1
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            item_time=int(len(train_loader)/10)
            for i in range(len(inputs)-3*sequ_length):
                optimiser_step1.zero_grad()
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                inputs_decor=net.decoder(inputs_out)
                loss_memo=model.loss_fn(target,inputs_decor)
                loss_memo.backward()
                memo_stat.append(loss_memo.item())
                optimiser_step1.step()
                if ((i+1)%(len(inputs)-2*sequ_length)==1)and((t+1)%(item_time+1)==1):
                    print("[Train][Epochs:%d/%d][Train Time:%d/%d][Duration:%f]MEMO_LOSS:%f]"
                        %(n+1,epochs,t+1,len(train_loader),time.time()-train_start_time,mean_stat(memo_stat)))
                    cat=torch.cat([input_pred,target,inputs_decor])
                    save_image(cat.cpu(),os.path.join(save_memo_images,"%f_train.png"%time.time()))
                    train_start_time=time.time()
        memo_average_train=mean_stat(memo_stat)
        memo_train.append(memo_average_train)

        memo_stat=[]
        val_start_time=time.time()
        for t,input_ in enumerate(val_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            sequ_length=1
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            item_time=int(len(val_loader)/10)
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                inputs_out=inputs_lstm.view(-1,64,32,32)
                inputs_decor=net.decoder(inputs_out)
                loss_memo=model.loss_fn(target,inputs_decor)
                memo_stat.append(loss_memo.item())
                if((i+1)%(len(inputs)-2*sequ_length)==1) and ((t+1)%(item_time+1)==1):
                    print("[Val][Epochs:%d/%d][Test Time:%d/%d][Duration:%f][MEMO_LOSS:%f]"
                        %(n+1,epochs,t+1,len(val_loader),time.time()-val_start_time,mean_stat(memo_stat)))
                    cat=torch.cat([input_pred,target,inputs_decor])
                    save_image(cat.cpu(),os.path.join(save_memo_images,"%f_val.png"%time.time()))
                    val_start_time=time.time()
        memo_average_val=mean_stat(memo_stat)
        memo_val.append(memo_average_val)
        scheduler.step()
        epoch.append(n+1)
    
    memo_stat=[]
    test_start_time=time.time()
    for t,input_ in enumerate(test_loader):
        if AutoEncoder_Type==1:
            inputs=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            inputs=input_['Image']
        sequ_length=1
        inputs=inputs.to(device)
        inputs=Variable(inputs)
        item_time=int(len(test_loader)/10)
        for i in range(len(inputs)-3*sequ_length):
            input_pred=inputs[i:i+3*sequ_length]
            target=inputs[i+3*sequ_length:i+4*sequ_length]
            inputs_encor=net.encoder(input_pred)
            inputs_lstm=model(inputs_encor,device)
            inputs_out=inputs_lstm.view(-1,64,32,32)
            inputs_decor=net.decoder(inputs_out)
            loss_memo=model.loss_fn(target,inputs_decor)
            memo_stat.append(loss_memo.item())
            if((i+1)%(len(inputs)-2*sequ_length)==1) and ((t+1)%(item_time+1)==1):
                print("[Test][Test Time:%d/%d][Duration:%f][MEMO_LOSS:%f]"
                    %(i+1,len(test_loader),time.time()-test_start_time,mean_stat(memo_stat)))
                cat=torch.cat([input_pred,target,inputs_decor])
                save_image(cat.cpu(),os.path.join(save_memo_images,"%f_test.png"%time.time()))
                test_start_time=time.time()
    
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':epoch,'memo_train':memo_train,'memo_val':memo_val})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt=plt.subplot()
    sbplt.plot('x','memo_train', data=df,color='red',label='memo_train')
    sbplt.plot('x','memo_val', data=df,color='blue',label='memo_val',linestyle='dashed')
    sbplt.set_xlabel('Epoch')
    sbplt.set_ylabel('MEMO LOSS (Full Dataset)')
    plt.legend(loc='upper left')
    plt.title('Loss for MEMO (Full Dataset) '+'('+AutoEncoder_Type_Name+')')
    plt.grid(True)
    plt.savefig(os.path.join(save_memo_figure,'memo_%f.png'%(time.time())),dip=100)
    plt.show()
    print ('memo train loss is:',memo_train)
    print ('memo val loss is:',memo_val)
    print ('Contragts! Finished! :)')
    return model


