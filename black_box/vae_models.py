from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F

def loss_lambda(recon_x, x, mu, logvar, r):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + r * KLD, BCE, KLD

### 4 convolutional layer
class VAE18(nn.Module):
    def __init__(self):
        super(VAE18, self).__init__()

        # Encoder
        self.conv0 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias= False)
        self.conv0_bn = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias= False)
        self.conv3_bn = nn.BatchNorm2d(256)
        # Latent space
        self.fc21 = nn.Linear(4096, 128)
        self.fc22 = nn.Linear(4096, 128)

        # Decoder
        self.fc3 = nn.Linear(128, 4096)
        self.fc3_bn = nn.BatchNorm1d(4096)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias= False)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=3, bias= False)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv0_bn(self.conv0(x)))
        out = self.relu(self.conv1_bn(self.conv1(out)))
        out = self.relu(self.conv2_bn(self.conv2(out)))
        out = self.relu(self.conv3_bn(self.conv3(out)))
        h1 = out.view(out.size(0), -1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = h3.view(h3.size(0), 256, 4, 4)
        out = self.relu(self.deconv1_bn(self.deconv1(out)))
        out = self.relu(self.deconv2_bn(self.deconv2(out)))
        out = self.relu(self.deconv3_bn(self.deconv3(out)))
        out = self.sigmoid(self.deconv4(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

