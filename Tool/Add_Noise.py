#Add Noise to an Image#
import numpy as np
import os
import random
import torch

def noisy (image):
    mean=0
    var=10
    sigma=var**0.5
    row,col,ch=image.shape
    image_noisy=np.random.normal(mean,sigma,(row,col,ch))
    image_compound=image_noisy
    return image_compound

