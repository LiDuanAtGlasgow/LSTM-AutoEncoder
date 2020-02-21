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

def train(model,epochs,data_loader,net,lr,device,AutoEncoder_Type,Classifier):
    selfnoise=SelfNoise.Gaussian_Nosie()
    train_start_time=time.time()
    epoch=[]
    net=net[0]
    optimizer_Classifier=optim.Adam(Classifier.parameters(),lr=lr)
    save_figure='./data/classification/result'
    if not os.path.exists(save_figure):
        os.mkdir(save_figure)
    save_classifier='./data/classification/classifier'
    if not os.path.exists(save_classifier):
        os.mkdir(save_classifier)
    
    path_classifier='./data/classification/classifier/%f.pth'%(time.time())
    
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
    print("the encoding and from numpy tensor is finished!")

    start_time=time.time()
    random.shuffle(inputs_random)
    inputs_randperm_train=inputs_random[0:int(len(inputs_random)*0.80)]
    inputs_randperm_val=inputs_random[int(len(inputs_random)*0.80):int(len(inputs_random)*0.90)]
    inputs_randperm_test=inputs_random[int(len(inputs_random)*0.90):int(len(inputs_random))]
    sequ_length=1

    train_epoch=[]
    acc_train=[]
    test_epoch=[]
    acc_test=[]
    
    for n in range(epochs):

        train_acc=0
        train_loss=[]
        random.shuffle(inputs_randperm_train)
        for z in range(len(inputs_randperm_train)):
            sample=inputs_randperm_train[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            label=sample["label"]
            item_time=int(len(inputs_randperm_train)/10)
            batch_acc=0
            number=len(inputs)-3*sequ_length
            for i in range(len(inputs)-3*sequ_length):
                optimizer_Classifier.zero_grad()
                input_pred=inputs_noise[i:i+3*sequ_length]
                target_label=label[i+3*sequ_length]
                target_label=target_label.to(device)
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                pred_label=Classifier(inputs_lstm)
                _,target_index=torch.max(target_label,1)
                _,pred_index=torch.max(pred_label,1)
                loss_fn=Classifier.loss_fn(pred_label,target_index)
                loss_fn.backward()
                train_loss.append(loss_fn.item())
                optimizer_Classifier.step()
                if pred_index==target_index:
                    batch_acc+=1
                    train_acc+=1
                if ((i+1)%(len(inputs)-3*sequ_length)==1)and((z+1)%(item_time)==1):
                    acc_rate=100*(batch_acc/number)
                    print("[Epochs:%d/%d][Train Time:%d/%d][Duration:%f][Loss:%f][Accuaracy:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm_train),time.time()-train_start_time,mean_stat(train_loss),acc_rate),"Predicted Label is:",pred_label)
                    train_start_time=time.time()
        loss_mean=mean_stat(train_loss)
        total_number=len(inputs_randperm_train)*(len(inputs)-3*sequ_length)
        total_acc=100*(train_acc/total_number)
        train_epoch.append(loss_mean)
        acc_train.append(total_acc)

        test_loss=[]
        test_acc=0
        random.shuffle(inputs_randperm_val)
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            label=sample["label"]
            item_time=int(len(inputs_randperm_test)/10)
            batch_acc=0
            number=len(inputs)-3*sequ_length
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs_noise[i:i+3*sequ_length]
                target_label=label[i+3*sequ_length]
                target_label=target_label.to(device)
                inputs_encor=net.encoder(input_pred)
                inputs_lstm=model(inputs_encor,device)
                pred_label=Classifier(inputs_lstm)
                _,target_index=torch.max(target_label,1)
                _,pred_index=torch.max(pred_label,1)
                loss_fn=Classifier.loss_fn(pred_label,target_index)
                if pred_index==target_index:
                    batch_acc+=1
                    test_acc+=1
                test_loss.append(loss_fn.item())
                if((i+1)%(len(inputs)-4*sequ_length)==1) and ((z+1)%(item_time+1)==1):
                    acc_rate=(batch_acc/number)*100
                    print("[Epochs:%d/%d][Test Time:%d/%d][Duration:%f][Loss: %f][Accuracy: %f]"
                        %(n+1,epochs,z+1,len(inputs_randperm_val),time.time()-train_start_time,mean_stat(test_loss),acc_rate),"Predicted label is:",pred_label)
                    train_start_time=time.time()
        loss_mean=mean_stat(test_loss)
        total_number=len(inputs_randperm_val)*(len(inputs)-3*sequ_length)
        total_acc=100*(test_acc/total_number)
        test_epoch.append(loss_mean)
        acc_test.append(total_acc)
        epoch.append(n)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':epoch, 'train_loss':train_epoch, 'train_acc':acc_train,'test_loss':test_epoch,'test_acc':acc_test})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.plot('x', 'train_loss',data=df,color='red',label='train_loss')
    sbplt1.plot('x', 'test_loss',data=df,color='blue',label='test_loss',linestyle='dashed')
    sbplt1.set_xlabel('Epoch')
    plt.legend(loc='upper left')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x', 'train_acc',data=df,color='green',label='train_acc')
    sbplt2.plot('x', 'test_acc',data=df,color='yellow',label='test_acc',linestyle='dashed')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("Cross-Entropy Loss and Accuracy of "+AutoEncoder_Type_Name)
    plt.savefig(os.path.join(save_figure,'%f.png'%(time.time())),dip=100)
    print ("Train Loss is:", train_epoch)
    print ("Train acc is:", acc_train)
    print ("Test Loss is:",test_epoch)
    print ("Test acc is:", acc_test)
    torch.save(Classifier,path_classifier)
    plt.show()
    print ("Train Ends!")
    return Classifier

