import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight_randn(submodule):
    if isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.Linear):
        torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # 100
        # 8192
        # 512, 4, 4
        # 256, 8, 8
        # 128, 16, 16
        # 64, 32, 32
        # 3, 64, 64
        self.linear = nn.Linear(z_dim, 8192)
        self.batch_norm = nn.BatchNorm1d(8192)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.layers = self._make_layers(4,
                                        [1, 1, 1, 1],
                                        [512, 256, 128, 64, 3],
                                        [4, 4, 4, 4],
                                        [2, 2, 2, 2],
                                        [1, 1, 1, 1],
                                        0.8)
        self.tanh = nn.Tanh()
        
        self.linear.apply(init_weight_randn)
        self.layers.apply(init_weight_randn)
        
    def _make_layers(self,
                     num_layers,
                     upsamples,
                     channels,
                     kernel_sizes,
                     strides,
                     paddings,
                     batch_norm_momentum):
        layers = []
        for i in range(num_layers):
            if upsamples[i] > 1:
                layers.append(nn.Upsample(scale_factor=upsamples[i]))
                layers.append(nn.Conv2d(channels[i],
                                        channels[i + 1],
                                        kernel_sizes[i],
                                        strides[i],
                                        padding=paddings[i],
                                        bias=False))
            else:
                layers.append(nn.ConvTranspose2d(channels[i],
                                                 channels[i + 1],
                                                 kernel_sizes[i],
                                                 strides[i],
                                                 padding=paddings[i],
                                                 bias=False))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(channels[i + 1],
                                             momentum=batch_norm_momentum))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = out.view([-1, 512, 4, 4])
        out = self.layers(out)
        out = self.tanh(out)
        return out
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # 3, 64, 64
        # 64, 32, 32
        # 128, 16, 16
        # 256, 8, 8
        # 512, 4, 4
        # 8192
        # 100
        self.layers = self._make_layers(4,
                                        [3, 64, 128, 256, 512],
                                        [5, 5, 5, 5],
                                        [2, 2, 2, 2],
                                        [2, 2, 2, 2])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8192, 1)
        
        self.linear.apply(init_weight_randn)
        self.layers.apply(init_weight_randn)
        
    def _make_layers(self,
                     num_layers,
                     channels,
                     kernel_sizes,
                     strides,
                     paddings):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels[i],
                                    channels[i + 1],
                                    kernel_sizes[i],
                                    strides[i],
                                    padding=paddings[i],
                                    bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        out = self.flatten(out)
        out = self.linear(out)
        return out