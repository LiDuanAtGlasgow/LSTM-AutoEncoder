import torch
import torch.nn as nn

class AutoEncoder (nn.Module):
    def __init__(self, device):
        super(AutoEncoder, self).__init__()
        self._device = device

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # b, 128, 16, 16
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # b, 64, 32,32

        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
    
    def encoder(self, x):
        out = self.elu(self.conv1(x))
        
        return out

    def decoder(self,x):
        out = self.elu(self.deconv4(x))
        out = self.tanh(out)
        return out

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out

    def loss_function(self,recon_x, x):
        loss = nn.MSELoss(reduction='sum')(recon_x, x)
        return loss
