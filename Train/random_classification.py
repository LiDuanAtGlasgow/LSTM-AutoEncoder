# pylint: skip-file
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
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from Tool import freeze
import seaborn as sns
import numpy as np
from Tool import custom_binary_encoding as binary_encoding
from Tool import Weight_binary_encoding as weight_encoding

def train(epochs,train_loader,val_loader,test_loader,net,lr,device,AutoEncoder_Type,classifier,index_column):
    train_start_time=time.time()
    epoch=[]
    net=net
    optimizer_classifier=optim.Adam(classifier.parameters(),lr=lr)
    save_figure='./data/random_classification/result'
    if not os.path.exists(save_figure):
        os.makedirs(save_figure)
    scheduler=optim.lr_scheduler.StepLR(optimizer_classifier,step_size=4,gamma=0.01,last_epoch=-1)

    sequ_length=1
    train_epoch=[]
    acc_train=[]
    val_epoch=[]
    acc_val=[]
    for n in range(epochs):
        train_acc=0
        train_loss=[]
        item_time=int(len(train_loader)/10)
        for t,input_ in enumerate(train_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            Label=input_['Label']
            for i in range (len(inputs)-3):
                optimizer_classifier.zero_grad()
                input_pred=inputs[i:i+1*sequ_length]
                label=Label[i]
                if index_column==2:
                    hot_key=binary_encoding.encoding(label)
                    label=hot_key.binary_encoding()
                if index_column==3:
                    hot_key=weight_encoding.encoding(label)
                    label=hot_key.binary_encoding()
                label=label.to(device)
                input_encor=net.encoder(input_pred)
                pred_label=classifier(input_encor)
                _,target_index=torch.max(label,1)
                _,pred_index=torch.max(pred_label,1)
                loss_fn=classifier.loss_fn(pred_label,target_index)
                loss_fn.backward()
                train_loss.append(loss_fn.item())
                optimizer_classifier.step()
                if pred_index==target_index:
                    train_acc+=1
                if (t+1)%(item_time+1)==0 and (i+1)%(len(inputs)-2)==1:
                    print("[Train][Epochs:%d/%d][Train Batch:%d/%d][Duration:%f][Loss:%f]"
                        %(n+1,epochs,t+1,len(train_loader),time.time()-train_start_time,mean_stat(train_loss)),"Predicted Label is:",pred_label)
                    train_start_time=time.time()
        loss_mean=mean_stat(train_loss)
        total_number=len(train_loader)*(len(inputs)-3)
        total_acc=100*(train_acc/total_number)
        train_epoch.append(loss_mean)
        acc_train.append(total_acc)
        print ("The Average Loss is (Train):",loss_mean)
        print ("The Total Accurarcy is (Train):",total_acc)
        """
        """
        val_start_time=time.time()
        val_loss=[]
        val_acc=0
        item_time=int(len(val_loader)/10)
        for t,input_ in enumerate(val_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:][0:3]
            if AutoEncoder_Type==2:
                inputs=input_['Image'][0:3]
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            Label=input_['Label']
            for i in range(len(inputs)):
                input_pred=inputs[i:i+1*sequ_length]
                label=Label[i]
                if index_column==2:
                    hot_key=binary_encoding.encoding(label)
                    label=hot_key.binary_encoding()
                if index_column==3:
                    hot_key=weight_encoding.encoding(label)
                    label=hot_key.binary_encoding()
                label=label.to(device)
                input_encor=net.encoder(input_pred)
                pred_label=classifier(input_encor)
                _,target_index=torch.max(label,1)
                _,pred_index=torch.max(pred_label,1)
                loss_fn=classifier.loss_fn(pred_label,target_index)
                val_loss.append(loss_fn.item())
                if pred_index==target_index:
                    val_acc+=1
                if (t+1)%(item_time+1)==1 and (i+1)%(len(inputs)+1)==1:
                    print("[Val][Epochs:%d/%d][Val Batch:%d/%d][Duration:%f][Loss:%f]"
                        %(n+1,epochs,t+1,len(val_loader),time.time()-val_start_time,mean_stat(val_loss)),"Predicted Label is:",pred_label)
                    val_start_time=time.time()
        loss_mean=mean_stat(val_loss)
        total_number=len(val_loader)*(len(inputs))
        total_acc=100*(val_acc/total_number)
        val_epoch.append(loss_mean)
        acc_val.append(total_acc)
        print ("The Average Loss is (Val):",loss_mean)
        print ("The Total Accurarcy is (Val):",total_acc)
        epoch.append(n+1)
        scheduler.step()
    
    test=[]   
    freeze.frozon(net)
    freeze.frozon(classifier)
    test_start_time=time.time()
    item_time=int(len(test_loader)/10)
    for t,input_ in enumerate(test_loader):
        if AutoEncoder_Type==1:
            inputs=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            inputs=input_['Image']
        inputs=Variable(inputs)
        inputs=inputs.to(device)
        Label=input_['Label']
        for i in range(len(inputs)):
            input_pred=inputs[i:i+1*sequ_length]
            label=Label[i]
            if index_column==2:
                hot_key=binary_encoding.encoding(label)
                label=hot_key.binary_encoding()
            if index_column==3:
                hot_key=weight_encoding.encoding(label)
                label=hot_key.binary_encoding()
            label=label.to(device)
            input_encor=net.encoder(input_pred)
            pred_label=classifier(input_encor)
            _,target_index=torch.max(label,1)
            _,pred_index=torch.max(pred_label,1)
            loss_fn=classifier.loss_fn(pred_label,target_index)
            if (t+1)%(item_time+1)==1 and (i+1)%(len(inputs)+1)==1:
                print(pred_label)
                print("[Test][Test Batch:%d/%d][Duration:%f]"%(t+1,len(test_loader),time.time()-test_start_time),
                    "Target Index:",target_index.item(),"Prediction Index:",pred_index.item())
                test_start_time=time.time()
            result={"target":target_index.item(),"prediction":pred_index.item()}
            test.append(result)
    zero_to_zero=0
    zero_to_one=0
    zero_to_two=0
    """
    zero_to_three=0
    zero_to_four=0
    """
    one_to_zero=0
    one_to_one=0
    one_to_two=0
    """
    one_to_three=0
    one_to_four=0
    """
    two_to_zero=0
    two_to_one=0
    two_to_two=0
    """
    two_to_three=0
    two_to_four=0
    """
    """
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
    """
    
    for t in range (len(test)):
        if test[t]["target"]==0:
            if test[t]["prediction"]==0:
                zero_to_zero+=1
            if test[t]["prediction"]==1:
                zero_to_one+=1
            if test[t]["prediction"]==2:
                zero_to_two+=1
            """
            if test[t]["prediction"]==3:
                zero_to_three+=1
            if test[t]["prediction"]==4:
                zero_to_four+=1
            """
        if test[t]["target"]==1:
            if test[t]["prediction"]==0:
                one_to_zero+=1
            if test[t]["prediction"]==1:
                one_to_one+=1
            if test[t]["prediction"]==2:
                one_to_two+=1
            """
            if test[t]["prediction"]==3:
                one_to_three+=1
            if test[t]["prediction"]==4:
                one_to_four+=1
            """
        if test[t]["target"]==2:
            if test[t]["prediction"]==0:
                two_to_zero+=1
            if test[t]["prediction"]==1:
                two_to_one+=1
            if test[t]["prediction"]==2:
                two_to_two+=1
            """
            if test[t]["prediction"]==3:
                two_to_three+=1
            if test[t]["prediction"]==4:
                two_to_four+=1
            """
        """
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
        """
    """
    zero=zero_to_zero+zero_to_one+zero_to_two+zero_to_three+zero_to_four+1
    one=one_to_zero+one_to_one+one_to_two+one_to_three+one_to_four+1
    two=two_to_zero+two_to_one+two_to_two+two_to_three+two_to_four+1
    three=three_to_zero+three_to_one+three_to_two+three_to_three+three_to_four+1
    four=four_to_zero+four_to_one+four_to_two+four_to_three+four_to_four+1
    """
    zero=zero_to_zero+zero_to_one+zero_to_two+1
    one=one_to_zero+one_to_one+one_to_two+1
    two=two_to_zero+two_to_one+two_to_two+1

    z_z=zero_to_zero/zero
    z_o=zero_to_one/zero
    z_tw=zero_to_two/zero
    """
    z_th=zero_to_three/zero
    z_f=zero_to_four/zero
    """
    o_z=one_to_zero/one
    o_o=one_to_one/one
    o_tw=one_to_two/one
    """
    o_th=one_to_three/one
    o_f=one_to_four/one
    """
    tw_z=two_to_zero/two
    tw_o=two_to_one/two
    tw_tw=two_to_two/two
    """
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
    """
    """
    z=[z_z,z_o,z_tw,z_th,z_f]
    o=[o_z,o_o,o_tw,o_th,o_f]
    tw=[tw_z,tw_o,tw_tw,tw_th,tw_f]
    th=[th_z,th_o,th_tw,th_th,th_f]
    f=[f_z,f_o,f_tw,f_th,f_f]
    """
    z=[z_z,z_o,z_tw]
    o=[o_z,o_o,o_tw]
    tw=[tw_z,tw_o,tw_tw]
    """
    total=[z,o,tw,th,f]
    """
    total=[z,o,tw]
    """
    total=np.asarray(total,dtype=np.float32).reshape(5,5)
    """
    total=np.array(total,dtype=np.float32).reshape(3,3)
    ax=sns.heatmap(total,annot=True,cmap="YlGnBu",vmin=0,vmax=1,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%) Color Bar"})
    plt.title("Classification(%)")
    plt.savefig(os.path.join(save_figure,'%f.png'%(time.time())),dip=100)
    print (total)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df=pd.DataFrame({'x':epoch, 'train_loss':train_epoch, 'train_acc':acc_train,'val_loss':val_epoch,'val_acc':acc_val})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.plot('x', 'train_loss',data=df,color='red',label='train_loss')
    sbplt1.plot('x', 'val_loss',data=df,color='blue',label='val_loss',linestyle='dashed')
    sbplt1.set_xlabel('Epoch')
    plt.legend(loc='upper left')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x', 'train_acc',data=df,color='green',label='train_acc')
    sbplt2.plot('x', 'val_acc',data=df,color='yellow',label='val_acc',linestyle='dashed')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("Cross-Entropy Loss and Accuracy of "+AutoEncoder_Type_Name)
    plt.savefig(os.path.join(save_figure,'%f_random_classifica.png'%(time.time())),dip=100)
    plt.show()
    print ("Train Loss is:", train_epoch)
    print ("Train acc is:", acc_train)
    print ("Test Loss is:",val_epoch)
    print ("Test acc is:", acc_val)
    print ("Congrats, Training Finished!")
    return classifier
    
