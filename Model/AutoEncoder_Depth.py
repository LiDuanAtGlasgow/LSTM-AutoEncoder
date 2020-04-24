# pylint: skip-file
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # b, 32, 128, 128
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # b, 32, 64, 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # b, 64, 32, 32
    

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # b, 32, 64, 64
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # b, 32, 128, 128
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1) # b, 3, 256, 256

        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
    
    def encoder(self, x):
        out = self.elu(self.conv1(x))
        out = self.elu(self.conv2(out))
        out = self.elu(self.conv3(out))
        
        return out

    def decoder(self,x):
        out = self.elu(self.deconv2(x))
        out = self.elu(self.deconv3(out))
        out = self.elu(self.deconv4(out))
        out = self.tanh(out)
        return out

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out

    @staticmethod
    def loss_function(recon_x, x):
        loss = nn.MSELoss(reduction='sum')(recon_x, x)
        return loss

    @staticmethod
    def to_img(x):
        # Function when image is normalised...
        # x = 0.5 * (x + 1)
        # x = x.clamp(0, 1)
        # x = x.view(x.size(0), 3, img_size, img_size)
        return x
