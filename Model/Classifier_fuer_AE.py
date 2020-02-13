import torch
import torch.nn as nn
class Classifier (nn.Module):
    def __init__(self,device,output_feature,hidden_feature):
        super(Classifier,self).__init__()
        self.device=device
        self.output_feature=output_feature
        self.hidden_feature=hidden_feature
        self.fc1=nn.Linear(64*1024,self.hidden_feature )
        self.fc2=nn.Linear(self.hidden_feature,self.output_feature)
        self.relu=nn.ReLU(True)
        self.logsoftmax=nn.LogSoftmax(True)
        self.leakyrelu=nn.LeakyReLU(True)
        self.elu=nn.ELU(True)
    


    def forward(self,x):
        x=x.view(-1,x.view(-1))
        x_1=self.leakyrelu(self.fc1(x))
        x_2=self.fc2(x_1)    
        return x_2
    
    def loss_fn(self,output,target):
        loss_fn=nn.CrossEntropyLoss()
        return(loss_fn(output,target))



        