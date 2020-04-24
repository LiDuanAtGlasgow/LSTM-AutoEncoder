# pylint: skip-file
import torch
import numpy as np
class encoding(object):
    def __init__(self,strings):
        self.strings=strings
    def binary_encoding(self):
        strings=self.strings
        if strings=='PANT':
            binary_num=torch.Tensor(np.array([[1,0,0,0,0]]))
        if strings=='SHIRT':
            binary_num=torch.Tensor(np.array([[0,1,0,0,0]]))
        if strings=='SWEATER':
            binary_num=torch.Tensor(np.array([[0,0,1,0,0]]))
        if strings=='TOWEL':
            binary_num=torch.Tensor(np.array([[0,0,0,1,0]]))
        if strings=='TSHIRT':
            binary_num=torch.Tensor(np.array([[0,0,0,0,1]]))
        return binary_num