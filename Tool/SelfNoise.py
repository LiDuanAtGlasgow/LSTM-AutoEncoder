# pylint: skip-file
import torch
from torch.autograd import Variable
import torch.nn as nn

class Gaussian_Nosie(nn.Module):
    def __init__(self,sigma=0.06):
        super(Gaussian_Nosie,self).__init__()
        self.stedev=sigma
    
    def forward(self,inputs):
        din=inputs+Variable(torch.randn(inputs.size())*self.stedev)
        return din
    




