# pylint: skip-file
import torch
from statistics import mean as mean_stat
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from Tool import custom_binary_encoding as binary_encoding
from Tool import freeze
import seaborn as sns
from Tool import Weight_binary_encoding as weight_encoding


##############################
#####Tarin Model(LSTM)########
##############################
def train(model,epochs,train_loader,val_loader,test_loader,net,lr,device,AutoEncoder_Type,classifier,index_column):
    train_start_time=time.time()
    sequ_length=1
    optimiser_step1 = torch.optim.Adam(model.parameters(), lr=lr)
    memo_train=[]
    memo_val=[]
    epoch=[]
    acc_train=[]
    acc_val=[]
    class_train=[]
    class_val=[]
    save_radmemo_image='./data/random_memo/images'
    if not os.path.exists(save_radmemo_image):
        os.makedirs(save_radmemo_image)
    save_radmemo_figure='./data/random_memo/figure'
    if not os.path.exists(save_radmemo_figure):
        os.makedirs(save_radmemo_figure)
    scheduler=optim.lr_scheduler.StepLR(optimiser_step1,step_size=4,gamma=0.01,last_epoch=-1)
    
    for n in range(epochs):
        train_acc=0
        memo_stat=[]
        class_stat=[]
        for t,input_ in enumerate(train_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            item_time=int(len(train_loader)/10)
            Label=input_['Label']
            for i in range(len(inputs)-3*sequ_length):
                optimiser_step1.zero_grad()
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                label=Label[i+3*sequ_length]
                if index_column==2:
                    Hot_Key=binary_encoding.encoding(label)
                    label=Hot_Key.binary_encoding()
                if index_column==3:
                    Hot_Key=weight_encoding.encoding(label)
                    label=Hot_Key.binary_encoding()
                label=label.to(device)
                input_encor=net.encoder(input_pred)
                input_lstm=model(input_encor,device)
                input_out=input_lstm.view(-1,64,32,32)
                input_decor=net.decoder(input_out)
                pred_label=classifier(input_lstm)
                _,target_index=torch.max(label,1)
                _,pred_index=torch.max(pred_label,1)
                if target_index==pred_index:
                    train_acc+=1
                loss_memo=model.loss_fn(input_decor,target)
                loss_class=classifier.loss_fn(pred_label,target_index)
                loss_total=loss_memo+50*loss_class
                class_stat.append(loss_class.item())
                loss_total.backward()
                memo_stat.append(loss_memo.item())
                optimiser_step1.step()
                if ((i+1)%(len(inputs)-2*sequ_length)==1)and((t+1)%(item_time+1)==1):
                    print("[Train][Epochs:%d/%d][Train Time:%d/%d][Duration:%f][MSE:%f][Cross-Entropy:%f]"
                        %(n+1,epochs,t+1,len(train_loader),time.time()-train_start_time,mean_stat(memo_stat),mean_stat(class_stat)))
                    cat=torch.cat([input_pred,target,input_decor])
                    save_image(cat.cpu(),os.path.join(save_radmemo_image,"train_%f.png"%time.time()))
                    train_start_time=time.time()
        memo_average_train=mean_stat(memo_stat)
        memo_train.append(memo_average_train)
        total_num=len(train_loader)*(len(inputs)-3)
        acurracy_train=100*(train_acc/total_num)
        class_average_train=mean_stat(class_stat)
        acc_train.append(acurracy_train)
        class_train.append(class_average_train)

        val_start_time=time.time()
        memo_stat=[]
        val_acc=0
        class_stat=[]
        for t,input_ in enumerate(val_loader):
            if AutoEncoder_Type==1:
                inputs=input_['Image'][:,0:1,:,:]
            if AutoEncoder_Type==2:
                inputs=input_['Image']
            sequ_length=1
            inputs=inputs.to(device)
            inputs=Variable(inputs)
            item_time=int(len(val_loader)/10)
            Label=input_['Label']
            for i in range(len(inputs)-3*sequ_length):
                input_pred=inputs[i:i+3*sequ_length]
                target=inputs[i+3*sequ_length:i+4*sequ_length]
                label=Label[i+3*sequ_length]
                if index_column==2:
                    Hot_Key=binary_encoding.encoding(label)
                    label=Hot_Key.binary_encoding()
                if index_column==3:
                    Hot_Key=weight_encoding.encoding(label)
                    label=Hot_Key.binary_encoding()
                label=label.to(device)
                input_encor=net.encoder(input_pred)
                input_lstm=model(input_encor,device)
                input_out=input_lstm.view(-1,64,32,32)
                input_decor=net.decoder(input_out)
                pred_label=classifier(input_lstm)
                _,target_index=torch.max(label,1)
                _,pred_index=torch.max(pred_label,1)
                if target_index==pred_index:
                    val_acc+=1
                loss_class=classifier.loss_fn(pred_label,target_index)
                loss_memo=model.loss_fn(input_decor,target)     
                memo_stat.append(loss_memo.item())
                class_stat.append(loss_class.item())
                if((i+1)%(len(inputs)-2*sequ_length)==1) and ((t+1)%(item_time+1)==1):
                    print("[Val][Epochs:%d/%d][Test Time:%d/%d][Duration:%f][MSE:%f][Cross-Entropy:%f]"
                        %(n+1,epochs,t+1,len(val_loader),time.time()-val_start_time,mean_stat(memo_stat),mean_stat(class_stat)))
                    cat=torch.cat([input_pred,target,input_decor])
                    save_image(cat.cpu(),os.path.join(save_radmemo_image,"val_%f.png"%time.time()))
                    val_start_time=time.time()
        memo_average_val=mean_stat(memo_stat)
        memo_val.append(memo_average_val)
        scheduler.step()
        total_num=len(val_loader)*(len(inputs)-3)
        acurracy_val=100*(val_acc/total_num)
        acc_val.append(acurracy_val)
        class_average_val=mean_stat(class_stat)
        class_val.append(class_average_val)
        epoch.append(n+1)

    test_start_time=time.time()
    freeze.frozon(model)
    memo_stat=[]
    test_acc=0
    class_stat=[]
    test=[]
    for t,input_ in enumerate(test_loader):
        if AutoEncoder_Type==1:
            inputs=input_['Image'][:,0:1,:,:]
        if AutoEncoder_Type==2:
            inputs=input_['Image']
        sequ_length=1
        inputs=inputs.to(device)
        inputs=Variable(inputs)
        item_time=int(len(test_loader)/10)
        Label=input_['Label']
        for i in range(len(inputs)-3*sequ_length):
            input_pred=inputs[i:i+3*sequ_length]
            target=inputs[i+3*sequ_length:i+4*sequ_length]
            label=Label[i+3*sequ_length]
            if index_column==2:
                Hot_Key=binary_encoding.encoding(label)
                label=Hot_Key.binary_encoding()
            if index_column==3:
                Hot_Key=weight_encoding.encoding(label)
                label=Hot_Key.binary_encoding()
            label=label.to(device)
            _,target_index=torch.max(label,1)
            input_encor=net.encoder(input_pred)
            input_lstm=model(input_encor,device)
            input_out=input_lstm.view(-1,64,32,32)
            input_decor=net.decoder(input_out)
            pred_label=classifier(input_lstm)
            _,pred_index=torch.max(pred_label,1)
            if pred_index==target_index:
                test_acc+=1
            loss_memo=model.loss_fn(input_decor,target)
            memo_stat.append(loss_memo.item())
            loss_class=classifier.loss_fn(pred_label,target_index)
            class_stat.append(loss_class.item())
            result={"target_index":target_index,"predicted_index":pred_index}
            if((i+1)%(len(inputs)-2*sequ_length)==1) and ((t+1)%(item_time+1)==1):
                print("[Test][Test Time:%d/%d][Duration:%f][MSE:%f][Cross-Entropy:%f]"
                    %(t+1,len(test_loader),time.time()-test_start_time,mean_stat(memo_stat),mean_stat(class_stat)))
                cat=torch.cat([input_pred,target,input_decor])
                save_image(cat.cpu(),os.path.join(save_radmemo_image,"test_%f.png"%time.time()))
                test_start_time=time.time()
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
        if test[t]["target_index"]==0:
            if test[t]["predicted_index"]==0:
                zero_to_zero+=1
            if test[t]["predicted_index"]==1:
                zero_to_one+=1
            if test[t]["predicted_index"]==2:
                zero_to_two+=1
            """
            if test[t]["predicted_index"]==3:
                zero_to_three+=1
            if test[t]["predicted_index"]==4:
                zero_to_four+=1
            """
        if test[t]["target_index"]==1:
            if test[t]["predicted_index"]==0:
                one_to_zero+=1
            if test[t]["predicted_index"]==1:
                one_to_one+=1
            if test[t]["predicted_index"]==2:
                one_to_two+=1
            """
            if test[t]["predicted_index"]==3:
                one_to_three+=1
            if test[t]["predicted_index"]==4:
                one_to_four+=1
            """
        if test[t]["target_index"]==2:
            if test[t]["predicted_index"]==0:
                two_to_zero+=1
            if test[t]["predicted_index"]==1:
                two_to_one+=1
            if test[t]["predicted_index"]==2:
                two_to_two+=1
            """
            if test[t]["predicted_index"]==3:
                two_to_three+=1
            if test[t]["predicted_index"]==4:
                two_to_four+=1
        if test[t]["target_index"]==3:
            if test[t]["predicted_index"]==0:
                three_to_zero+=1
            if test[t]["predicted_index"]==1:
                three_to_one+=1
            if test[t]["predicted_index"]==2:
                three_to_two+=1
            if test[t]["predicted_index"]==3:
                three_to_three+=1
            if test[t]["predicted_index"]==4:
                three_to_four+=1
        if test[t]["target_index"]==4:
            if test[t]["predicted_index"]==0:
                four_to_zero+=1
            if test[t]["predicted_index"]==1:
                four_to_one+=1
            if test[t]["predicted_index"]==2:
                four_to_two+=1
            if test[t]["predicted_index"]==3:
                four_to_three+=1
            if test[t]["predicted_index"]==4:
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
    total=np.asarray(total,dtype=np.float32).reshape(3,3)
    ax=sns.heatmap(total,annot=True,cmap="YlGnBu",vmin=0,vmax=1,fmt=".2f",xticklabels=False,yticklabels=False,cbar_kws={"label":"Classification Accuracy(%) Color Bar"})
    plt.title("classification(%)")
    plt.savefig(os.path.join(save_radmemo_figure,'memrandheatmap_%f.png'%(time.time())),dip=100)

    print (total)

    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    if AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    df1=pd.DataFrame({'x':epoch, 'class_train':class_train, 'acc_train':acc_train, 'class_val':class_val, 'acc_val':acc_val})
    df2=pd.DataFrame({'x':epoch, 'memo_train':memo_train, 'memo_val':memo_val})
    asplot=plt.figure()
    asplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.set_xlabel('Epoch')
    sbplt1.set_ylabel('Accuracy')
    sbplt1.plot('x','acc_train',data=df1,color='blue',label='acc_train')
    sbplt1.plot('x','acc_val',data=df1,color='red',label='acc_val',linestyle='dashed')
    plt.legend(loc='upper left')
    sbplt2=sbplt1.twinx()
    sbplt2.plot('x','class_train', data=df1,color='green',label='class_train')
    sbplt2.plot('x','class_val', data=df1,color='yellow',label='class_val',linestyle='dashed')
    sbplt2.set_ylabel('Cross Entropy')
    plt.legend(loc='upper right')
    plt.title(' Leave One Out Training of MEMO '+'('+AutoEncoder_Type_Name+')')
    plt.grid(True)
    plt.savefig(os.path.join(save_radmemo_figure,'memrand_%f.png'%(time.time())),dip=100)
    atplot=plt.figure()
    atplot.add_subplot(111)
    sbplt1=plt.subplot()
    sbplt1.set_xlabel('Epoch')
    sbplt1.set_ylabel('MSE Loss')
    sbplt1.plot('x','memo_train',data=df2,color='blue',label='memo_train')
    sbplt1.plot('x','memo_val',data=df2,color='red',label='memo_val')
    plt.legend(loc='upper right')
    plt.title('Leave One Output Memory Module Training of MEMO'+'('+AutoEncoder_Type_Name+')')
    plt.grid(True)
    plt.savefig(os.path.join(save_radmemo_figure,'memmod_%f.png'%time.time()),dip=100)
    plt.show()
    print ('memo train loss is:',memo_train)
    print ('memo val loss is:',memo_val)
    print ('class train loss is:',class_train)
    print ('class val loss is:',class_val)
    print ('acc train accuracy is:',acc_train)
    print ('acc val accuracy is:',acc_val)

    print ('Contragts! Finally Finished! :)')

    return model