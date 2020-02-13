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
def train(models,epochs,mario_lstm_loader,net,lr,batch_size,device,AutoEncoder_Type,category,path_step1,classifier):
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
    loss_fn_step1 = nn.MSELoss(reduction='sum')
    loss_fn_step1=loss_fn_step1.to(device)
    optimiser_image = torch.optim.Adam(models.parameters(), lr=lr)
    optimiser_label = torch.optim.Adam(classifier.parameters(), lr=lr)
    lossstatistics_test=[]
    lossstatistics_test_lstm=[]
    training_epoch=[]
    lossstatistics_train=[]
    lossstatistics_train_lstm=[]
    net=net[0]
    epochs=int(epochs)
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

    
    with torch.autograd.set_detect_anomaly(False):
        for n in range(epochs):
            total_loss=[]
            mean_loss=[]
            total_loss_lstm=[]
            mean_loss_lstm=[]
            for z in range(len(inputs_randperm)):
                sample=inputs_randperm[z]
                inputs=sample["picture"]
                image_label=sample["label"]
                sequ_length=1
                inputs_noise=selfnoise(inputs)
                inputs=inputs.to(device)
                inputs=Variable(inputs)
                inputs_noise=inputs_noise.to(device)
                inputs_noise=Variable(inputs_noise)
                item_time=int(len(inputs_randperm)/10)
                for i in range(len(inputs)-3*sequ_length):
                    optimiser_label.zero_grad()
                    optimiser_image.zero_grad()
                    input_pred=inputs_noise[i:i+3*sequ_length]
                    input_enc=net.encoder(input_pred)
                    print("The shape of input_enc is:",input_enc.shape)
                    image_label_target=image_label[i+3*sequ_length]
                    target=inputs_noise[i+3*sequ_length:i+4*sequ_length]
                    pred_lstm_t=models(input_enc,device)
                    print('The shape of pred_lstm_t is',pred_lstm_t.shape)
                    pred_lstm=pred_lstm_t.view(-1,64,32,32)
                    pred_dec=net.decoder(pred_lstm)
                    pred_label=classifier(pred_lstm_t)
                    _,image_label_max=torch.max(image_label_target,1)
                    loss_function=models.loss_fn(pred_dec,target)
                    loss_function_classifier=classifier.loss_fn(pred_label,image_label_max)
                    total_loss=loss_function+loss_function_classifier
                    total_loss.backward()
                    optimiser_image.step()
                    optimiser_label.step()
                    total_loss.append(loss_function_classifier.item())
                    mean_loss.append(loss_function_classifier.item())
                    total_loss_lstm.append(loss_function.item())
                    mean_loss_lstm.append(loss_function.item())
                    if((i+1)%(len(inputs)-4*sequ_length)==0) and ((z+1)%(item_time+1)==0):
                        print("[Epochs:%d/%d][Bacth Time:%d/%d][Duration:%3f][Train Loss is %f]"
                            %(n+1,epochs,z+1,len(inputs_randperm),time.time()-train_start_time,mean_stat(mean_loss)),'Pred_Label:',pred_label)
                        train_start_time=time.time()
            average_train_loss_set=mean_stat(total_loss)  
            lossstatistics_train.append(average_train_loss_set)
            average_train_loss_set_lstm=mean_stat(total_loss_lstm)  
            lossstatistics_train_lstm.append(average_train_loss_set_lstm)

            total_loss=[]
            test_acc=0
            test_total=len(inputs_randperm_test)*(len(inputs_noise)-3)
            for z in range (len(inputs_randperm_test)):
                sample=inputs_randperm_test[z]
                inputs=sample['picture']
                image_label=sample['label']
                sequ_length=1
                inputs_noise=selfnoise(inputs)
                inputs=inputs.to(device)
                inputs=Variable(inputs)
                inputs_noise=inputs_noise.to(device)
                inputs_noise=Variable(inputs_noise)
                item_time=int(len(inputs_randperm_test)/10)
                for i in range(0,len(inputs)-3*sequ_length):
                    input_pred=inputs_noise[i:i+3*sequ_length]
                    input_enc=net.encoder(input_pred)
                    target=inputs_noise[i+3*sequ_length:i+4*sequ_length]
                    image_label_target=image_label[i+3*sequ_length]
                    pred_lstm_t=models(input_enc,device)
                    pred_lstm=pred_lstm_t.view(-1,64,32,32)
                    pred_dec=net.decoder(pred_lstm)
                    pred_label=classifier(pred_lstm_t)
                    loss_comparison=models.loss_fn(pred_dec,target)
                    total_loss.append(loss_comparison.item())
                    _,max_label=torch.max(pred_label,1)
                    _,image_label_target=torch.max(image_label_target,1)
                    if (max_label==image_label_target):
                        test_acc+=1
                    if((i+1)%(len(inputs)-4*sequ_length)==0) and ((z+1)%(item_time+1)==0):
                        print("[Epochs:%d/%d][Test Time:%d/%d][Duration:%3f][Accuracy Rate:%d][Loss Function:%f]"
                        %(n+1,epochs,z+1,len(inputs_randperm_test),time.time()-train_start_time,(100*(test_acc/test_total)),loss_comparison.item()))
                        train_start_time=time.time()
                        cat=torch.cat([input_pred,target,pred_dec])
                        save_image(cat.cpu(),os.path.join(save_image_lstm,"%d_%d.png"%(time.time(),i)))
            average_test_loss_set=100*(test_acc/test_total)
            lossstatistics_test.append(average_test_loss_set)
            training_epoch.append(n)
            average_test_loss_set_lstm=mean_stat(total_loss)
            lossstatistics_test_lstm.append(average_test_loss_set_lstm)
    if AutoEncoder_Type==1:
        AutoEncoder_Type_Name='Depth Image'
    elif AutoEncoder_Type==2:
        AutoEncoder_Type_Name='RGB Image'
    if category==1:
        category_name='Stiffness'
    elif category==2:
        category_name='Material'
    elif category==3:
        category_name='Starting Rotational Radius'
    df=pd.DataFrame({'x':training_epoch,'train':lossstatistics_train,'test':lossstatistics_test,'train_lstm':lossstatistics_train_lstm,'test_lstm':lossstatistics_test_lstm})
    ax_t=plt.figure()
    ax_t.add_subplot(111)
    ax=plt.subplot()
    ax.plot('x','train', data=df,color='red',label='train_loss')
    ax.plot('x','train_lstm', data=df,color='yellow',label='train_loss')
    ax.plot('x','test_lstm', data=df,color='green',label='train_loss',linestyle='dashed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross Entropy Training Loss')
    ax2=ax.twinx()
    ax2.plot('x','test', data=df,color='blue',label='test_ACC',linestyle='dashed')
    ax2.set_ylabel('Accuracy Rate(%)')
    plt.title('CrossEntropy Loss Function for LSTM and Accuracy Rate of '+AutoEncoder_Type_Name)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_figure_lstm,'CrossEntropy_Loss_of'+'_'+AutoEncoder_Type_Name+'('+category_name+')'+str(time.time())+'.png'),dip=100)
    torch.save(models,path_step1)
    torch.save(classifier,'./save_model/lstmclassifier.pth')
    print("Train Loss is:",lossstatistics_train)
    print("Test Loss is:",lossstatistics_test)
    print("Train LSTM Loss is:",lossstatistics_train_lstm)
    print("Test LSTM Loss is:",lossstatistics_test_lstm)
    print("Tranining is finishded!")
    return models

    

