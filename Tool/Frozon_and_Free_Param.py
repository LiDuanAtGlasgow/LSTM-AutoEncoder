
import PIL.Image as Image
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms as transforms
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Dataset as Dataset

def frozon_Param(model:nn.Module):
    for p in model.parameters():
        p.requires_grad=False

def free_Param(model:nn.Module):
    for p in model.parameters():
        p.requires_grad=True