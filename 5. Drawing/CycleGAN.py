import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
def upsample(in_channels, out_channels, dropout_rate=0):
    layers = []
    layers.append(nn.Upsample(scale_factor=2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if dropout_rate:
        layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)

def discriminator_block(in_channels, out_channels, stride=2, use_norm=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
        
    return nn.Sequential(*layers)

def init_weight_randn(submodule):
    if isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.Linear) or isinstance(submodule, nn.ConvTranspose2d):
        torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02)


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 final=False):
        super(ConvBlock, self).__init__()

        layer_list = []
        layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'))
        if final:
            layer_list.append(nn.Tanh())
        else:
            layer_list.append(nn.InstanceNorm2d(out_channels))
            layer_list.append(nn.ReLU(inplace=True))
            
        self.layers = nn.Sequential(*layer_list)
        self.layers.apply(init_weight_randn)
    
    def forward(self, x):
        return self.layers(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layers.apply(init_weight_randn)
        
    def forward(self, x):
        return self.layers(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layers.apply(init_weight_randn)
        
    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(num_channels),
        )
        
        self.layers.apply(init_weight_randn)
        
    def forward(self, x):
        out = self.layers(x)
        out = out + x
        return out

class GeneratorUNet(nn.Module):
    def __init__(self, dropout_rate=0):
        super(GeneratorUNet, self).__init__()
        
        channels = [3, 32, 64, 128, 256]
        
        self.downsample1 = downsample(channels[0], channels[1])
        self.downsample2 = downsample(channels[1], channels[2])
        self.downsample3 = downsample(channels[2], channels[3])
        self.downsample4 = downsample(channels[3], channels[4])
        
        self.upsample1 = upsample(channels[4], channels[3])
        self.upsample2 = upsample(channels[3] * 2, channels[2])
        self.upsample3 = upsample(channels[2] * 2, channels[1])
        self.upsample4 = nn.Upsample(scale_factor=2)
        
        self.last_conv = nn.Conv2d(channels[1] * 2, channels[0], kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        
        up1 = torch.cat([self.upsample1(down4), down3], dim=1)
        up2 = torch.cat([self.upsample2(up1), down2], dim=1)
        up3 = torch.cat([self.upsample3(up2), down1], dim=1)
        up4 = self.upsample4(up3)
        
        out = self.last_conv(up4)
        out = self.tanh(out)
        return out
    
    
class GeneratorResNet(nn.Module):
    def __init__(self):
        super(GeneratorResNet, self).__init__()
        
        channels = [3, 32, 64, 128]
        
        self.layers = nn.Sequential(
            ConvBlock(channels[0], channels[1], False),
            DownsampleBlock(channels[1], channels[2]),
            DownsampleBlock(channels[2], channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            ResidualBlock(channels[3]),
            UpsampleBlock(channels[3], channels[2]),
            UpsampleBlock(channels[2], channels[1]),
            ConvBlock(channels[1], channels[0], True),
        )   
        
    def forward(self, x):
        return self.layers(x)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        channels = [3, 32, 64, 128, 256]
        
        self.b1 = discriminator_block(channels[0], channels[1], use_norm=False)
        self.b2 = discriminator_block(channels[1], channels[2])
        self.b3 = discriminator_block(channels[2], channels[3])
        self.b4 = discriminator_block(channels[3], channels[4], stride=1)
    
        self.conv = nn.Conv2d(channels[4], 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.conv(out)
        
        return out