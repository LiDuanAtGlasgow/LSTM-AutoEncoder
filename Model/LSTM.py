
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

    def __init__(self,input_feature,n_layer,hidden_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_feature
        self.hidden_dim = hidden_dim
        self.num_layers = n_layer
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,self.num_layers)

    def init_hidden(self,batch_size):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_dim))

    def forward(self, input,device):
        time_step=input.size(0)
        batch_size=input.size(1)
        input=input.view(time_step,batch_size,-1)
        hidden_0=self.init_hidden(batch_size)
        c_0=self.init_hidden(batch_size)
        hidden_0=hidden_0.to(device)
        c_0=c_0.to(device)
        lstm_out,(hidden_0,c_0)= self.lstm(input,(hidden_0 ,c_0))
        lstm_out=lstm_out[-1,:,:]
        lstm_out=F.relu(lstm_out)
        return lstm_out
    
    def loss_fn(self,output,target):
        loss_fn=nn.MSELoss(reduction='sum')
        loss_function=loss_fn(output,target)
        return loss_function



        