import torch
import numpy as np
class encoding(object):
    def __init__(self,strings):
        self.strings=strings
    def binary_encoding(self):
        strings=self.strings
        if strings=='light':
            binary_num=torch.Tensor(np.array([[1,0,0]]))
        if strings=='medium':
            binary_num=torch.Tensor(np.array([[0,1,0]]))
        if strings=='heavy':
            binary_num=torch.Tensor(np.array([[0,0,1]]))
        return binary_num