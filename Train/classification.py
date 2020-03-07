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
from Tool import freeze
import seaborn as sns
import numpy as np

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
    scheduler=optim.lr_scheduler.StepLR(optimizer_Classifier,step_size=4,gamma=0.1,last_epoch=-1)


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

    train_epoch=[]
    acc_train=[]
    test_epoch=[]
    acc_test=[]
    
    for n in range(epochs):
        train_acc=0
        train_loss=[]
        random.shuffle(inputs_randperm_train)
        item_time=int(len(inputs_randperm_train)/10)
        for z in range(len(inputs_randperm_train)):
            sample=inputs_randperm_train[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            inputs_label=sample["label"]
            optimizer_Classifier.zero_grad()
            label=inputs_label[0]
            inputs_sec=inputs[0:3]
            inputs_encor=net.encoder(inputs_sec)
            inputs_lstm=model(inputs_encor,device)
            pred_label=Classifier(inputs_lstm)
            _,target_index=torch.max(label,1)
            _,pred_index=torch.max(pred_label,1)
            loss_fn=Classifier.loss_fn(pred_label,target_index)
            loss_fn.backward()
            train_loss.append(loss_fn.item())
            optimizer_Classifier.step()
            if pred_index==target_index:
                train_acc+=1
            if (z+1)%(item_time)==1:
                print("[Train][Epochs:%d/%d][Train Batch:%d/%d][Duration:%f][Loss:%f]"
                    %(n+1,epochs,z+1,len(inputs_randperm_train),time.time()-train_start_time,mean_stat(train_loss)),"Predicted Label is:",pred_label)
                train_start_time=time.time()
        loss_mean=mean_stat(train_loss)
        total_number=len(inputs_randperm_train)
        total_acc=100*(train_acc/total_number)
        train_epoch.append(loss_mean)
        acc_train.append(total_acc)
        print ("The Average Loss is (Train):",loss_mean)
        print ("The Total Accurarcy is (Train):",total_acc)

        test_loss=[]
        test_acc=0
        random.shuffle(inputs_randperm_val)
        item_time=int(len(inputs_randperm_val)/10)
        for z in range (len(inputs_randperm_val)):
            sample=inputs_randperm_val[z]
            inputs=sample['picture']
            inputs_noise=selfnoise(inputs)
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            inputs_noise=inputs_noise.to(device)
            inputs_noise=Variable(inputs_noise)
            inputs_label=sample["label"]
            label=inputs_label[0]
            inputs_sec=inputs_noise[0:3]
            inputs_encor=net.encoder(inputs_sec)
            inputs_lstm=model(inputs_encor,device)
            pred_label=Classifier(inputs_lstm)
            _,target_index=torch.max(label,1)
            _,pred_index=torch.max(pred_label,1)
            loss_fn=Classifier.loss_fn(pred_label,target_index)
            if pred_index==target_index:
                test_acc+=1
            test_loss.append(loss_fn.item())
            if((z+1)%(item_time+1)==1):
                print("[Valid][Epochs:%d/%d][Valid Batch:%d/%d][Duration:%f][Loss: %f]"
                    %(n+1,epochs,z+1,len(inputs_randperm_val),time.time()-train_start_time,mean_stat(test_loss)),"Predicted label is:",pred_label)
                train_start_time=time.time()
        loss_mean=mean_stat(test_loss)
        total_number=len(inputs_randperm_val)
        total_acc=100*(test_acc/total_number)
        test_epoch.append(loss_mean)
        acc_test.append(total_acc)
        print ("The Average Loss is (Test):",loss_mean)
        print ("The Total Accurarcy is (Test):",total_acc)
        epoch.append(n)
        scheduler.step()

    test=[]   
    freeze.frozon(net)
    freeze.frozon(model)
    freeze.frozon(Classifier)
    random.shuffle(inputs_randperm_test)
    item_time=int(len(inputs_randperm_test)/10)
    start_time=time.time()
    for z in range (len(inputs_randperm_test)):
        sample=inputs_randperm_test[z]
        inputs=sample['picture']
        inputs_noise=selfnoise(inputs)
        inputs_noise=inputs_noise[0:3]
        inputs_noise=inputs_noise.to(device)
        inputs_noise=Variable(inputs_noise)
        inputs_encor=net.encoder(inputs_noise)
        inputs_lstm=model(inputs_encor,device)
        pred_label=Classifier(inputs_lstm)
        inputs_label=sample['label']
        label=inputs_label[0]
        _,target_index=torch.max(label,1)
        _,pred_index=torch.max(pred_label,1)
        result={"target":target_index.item(),"prediction":pred_index.item()}
        test.append(result)
        if (z+1)%item_time==1:
            print("[Test][Test Batch:%d/%d][Duration:%f]"%(z+1,len(inputs_randperm_test),time.time()-start_time),
            "Target Index:",target_index.item(),"Prediction Index:",pred_index.item())
            start_time=time.time()
    zero_to_zero=0
    zero_to_one=0
    zero_to_two=0
    zero_to_three=0
    zero_to_four=0
    one_to_zero=0
    one_to_one=0
    one_to_two=0
    one_to_three=0
    one_to_four=0
    two_to_zero=0
    two_to_one=0
    two_to_two=0
    two_to_three=0
    two_to_four=0
    three_to_zero=0
    three_to_one=0
    three_to_two=0
    three_to_three=0
    three_to_four=0
    four_to_zero=0
    four_to_one=0
    four_to_two=0
    four_to_three=0
    four_to_four=0
    
    for t in range (len(test)):
        if test[t]["target"]==0:
            if test[t]["prediction"]==0:
                zero_to_zero+=1
            if test[t]["prediction"]==1:
                zero_to_one+=1
            if test[t]["prediction"]==2:
                zero_to_two+=1
            if test[t]["prediction"]==3:
                zero_to_three+=1
            if test[t]["prediction"]==4:
                zero_to_four+=1
        if test[t]["target"]==1:
            if test[t]["prediction"]==0:
                one_to_zero+=1
            if test[t]["prediction"]==1:
                one_to_one+=1
            if test[t]["prediction"]==2:
                one_to_two+=1
            if test[t]["prediction"]==3:
                one_to_three+=1
            if test[t]["prediction"]==4:
                one_to_four+=1
        if test[t]["target"]==2:
            if test[t]["prediction"]==0:
                two_to_zero+=1
            if test[t]["prediction"]==1:
                two_to_one+=1
            if test[t]["prediction"]==2:
                two_to_two+=1
            if test[t]["prediction"]==3:
                two_to_three+=1
            if test[t]["prediction"]==4:
                two_to_four+=1
        if test[t]["target"]==3:
            if test[t]["prediction"]==0:
                three_to_zero+=1
            if test[t]["prediction"]==1:
                three_to_one+=1
            if test[t]["prediction"]==2:
                three_to_two+=1
            if test[t]["prediction"]==3:
                three_to_three+=1
            if test[t]["prediction"]==4:
                three_to_four+=1
        if test[t]["target"]==4:
            if test[t]["prediction"]==0:
                four_to_zero+=1
            if test[t]["prediction"]==1:
                four_to_one+=1
            if test[t]["prediction"]==2:
                four_to_two+=1
            if test[t]["prediction"]==3:
                four_to_three+=1
            if test[t]["prediction"]==4:
                four_to_four+=1
    zero=zero_to_zero+zero_to_one+zero_to_two+zero_to_three+zero_to_four+1
    one=one_to_zero+one_to_one+one_to_two+one_to_three+one_to_four+1
    two=two_to_zero+two_to_one+two_to_two+two_to_three+two_to_four+1
    three=three_to_zero+three_to_one+three_to_two+three_to_three+three_to_four+1
    four=four_to_zero+four_to_one+four_to_two+four_to_three+four_to_four+1

    z_z=zero_to_zero/zero
    z_o=zero_to_one/zero
    z_tw=zero_to_two/zero
    z_th=zero_to_three/zero
    z_f=zero_to_four/zero
    o_z=one_to_zero/one
    o_o=one_to_one/one
    o_tw=one_to_two/one
    o_th=one_to_three/one
    o_f=one_to_four/one
    tw_z=two_to_zero/two
    tw_o=two_to_one/two
    tw_tw=two_to_two/two
    tw_th=two_to_three/two
    tw_f=two_to_four/two
    th_z=three_to_zero/three
    th_o=three_to_one/three
    th_tw=three_to_two/three
    th_th=three_to_three/three
    th_f=three_to_four/three
    f_z=four_to_zero/four
    f_o=four_to_one/four
    f_tw=four_to_two/four
    f_th=four_to_three/four
    f_f=four_to_four/four

    z=[z_z,z_o,z_tw,z_th,z_f]
    o=[o_z,o_o,o_tw,o_th,o_f]
    tw=[tw_z,tw_o,tw_tw,tw_th,tw_f]
    th=[th_z,th_o,th_tw,th_th,th_f]
    f=[f_z,f_o,f_tw,f_th,f_f]
    total=[z,o,tw,th,f]
    total=np.asarray(total,dtype=np.float32).reshape(5,5)
    ax=sns.heatmap(total,annot=True,cmap="YlGnBu",vmin=0,vmax=100,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%) Color Bar"})
    plt.title("Classification(%)")
    plt.savefig(os.path.join(save_figure,'%f.png'%(time.time())),dip=100)

    print (total)

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
    print ("Train Ends!")
    return Classifier

