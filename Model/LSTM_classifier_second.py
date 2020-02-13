import PIL.Image as Image
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F

#======LTSM Model=========#
class LSTM(nn.Module):

    def __init__(self,input_feature,n_layer,hidden_dim,output_classifier,hidden_classifier):
        super(LSTM, self).__init__()
        self.input_dim = input_feature
        self.hidden_dim = hidden_dim
        self.num_layers = n_layer
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers)
        self.output_classifier=output_classifier
        self.hidden_classifier=hidden_classifier
        self.fc1=nn.Linear(64*1024,self.hidden_classifier)
        self.fc2=nn.Linear(self.hidden_classifier,self.output_classifier)
        self.relu=nn.ReLU(True)
        self.dropout=nn.Dropout(0.2)
        self.softmax=nn.Softmax()
        self.logsoftmax=nn.LogSoftmax()


    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_dim))

    def forward(self,inputs,device):
        time_step=inputs.size(0)
        batch_size=inputs.size(1)
        inputs=inputs.view(time_step,batch_size,-1)
        hidden_0=self.init_hidden(batch_size)
        c_0=self.init_hidden(batch_size)
        hidden_0=hidden_0.to(device)
        c_0=c_0.to(device)
        lstm_out,(hidden_0,c_0)= self.lstm(inputs,(hidden_0 ,c_0))
        lstm_out=lstm_out[-1,:,:]
        lstm_out=F.relu(lstm_out)
        x=lstm_out.view(-1,64*1024)
        x_1=self.fc1(x)
        x_2=self.fc2(x_1)
        return x_2
    
    def loss_fn(self,output,target):
        loss_fn=nn.CrossEntropyLoss()
        return(loss_fn(output,target))



        