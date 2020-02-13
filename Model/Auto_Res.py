import torch.nn as nn
import torch

class Auto_Res(nn.Module):
    def __init__(self,in_chanels,hidden_layer,downsample=None):
        super(Auto_Res,self).__init__()
        self.in_chanels=in_chanels
        self.hidden_layer=hidden_layer
        self.downsample=downsample
        self.conv1=nn.Conv2d(self.in_chanels,self.hidden_layer,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(self.hidden_layer)
        self.conv2=nn.Conv2d(self.hidden_layer,self.hidden_layer,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv3=nn.Conv2d(self.hidden_layer,self.hidden_layer*4,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(self.hidden_layer*4)
        self.relu=nn.ReLU()
        self.deconv1=nn.ConvTranspose2d(self.hidden_layer*4,self.hidden_layer,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv2=nn.ConvTranspose2d(self.hidden_layer,self.hidden_layer,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv3=nn.ConvTranspose2d(self.hidden_layer,self.in_chanels,kernel_size=4,stride=2,padding=1)
        self.tanh=nn.Tanh()
    
    def encoder (self,x):
        self.residual=x
        out=self.conv1(x)
        out=self.bn1(out)
        self.residual2=self.relu(out)
        out=self.relu(out)
        #----------------#
        out=self.conv2(out)
        out=self.bn1(out)
        self.residual3=self.relu(out)
        out=self.relu(out)
        #----------------#
        out=self.conv3(out)
        out=self.bn2(out)
        self.residual4=self.relu(out)
        return out
    
    def decoder (self,x):
        out=self.deconv1(x)
        residual3=self.residual3
        #print("Output1 is :",out.shape)
        #print("Residual is:",self.residual.shape)
        #print("Residual2 is:",residual3.shape)
        out+=residual3 
        #----------------#       
        out=self.bn1(out)
        out=self.relu(out)
        #----------------#
        #print("The shape of out in phrase 2 is:",out.shape)
        out=self.deconv2(out)
        #print("The shape of out in phrase 3 is:",out.shape)
        out=self.bn1(out)
        out=self.relu(out)
        #----------------#
        residual2=self.residual2
        #print("The shape of out in phrase 4 is:",out.shape)
        out+=residual2
        #----------------#
        out=self.deconv3(out)
        out=self.relu(out)
        #----------------#

        out=self.tanh(out)

        return out
    def forward(self,x):
        feat=self.encoder(x)
        out=self.decoder(feat)
        return out

    def loss_function(self,recon_image,target):
        loss=nn.MSELoss(reduction='sum')(recon_image,target)

        return loss
    
        




        


        

    
