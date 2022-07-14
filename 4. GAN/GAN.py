import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # 100
        # 3236
        # 64, 7, 7
        # 128, 14, 14
        # 128, 28, 28
        # 64, 28, 28
        # 64, 28, 28
        # 1, 28, 28
        batch_norm_momentum = 0.9
        
        self.linear = nn.Linear(100, 3136)
        self.batch_norm = nn.BatchNorm1d(3136, momentum=batch_norm_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self._make_layers(4,
                                        [2, 2, 1, 1],
                                        [64, 128, 64, 64, 1],
                                        [3, 3, 3, 3],
                                        [1, 1, 1, 1],
                                        batch_norm_momentum, False)
        self.tanh = nn.Tanh()
        
    def _make_layers(self,
                     num_layers,
                     upsamples,
                     channels,
                     kernel_sizes,
                     strides,
                     batch_norm_momentum,
                     drop_out_rate):
        layers = []
        
        if drop_out_rate:
            layers.append(nn.Dropout(p=drop_out_rate))
        
        for i in range(num_layers):
            layers.append(nn.Upsample(scale_factor=upsamples[i]))
            layers.append(nn.Conv2d(channels[i],
                                    channels[i + 1],
                                    kernel_sizes[i],
                                    strides[i],
                                    padding=1,
                                    bias=False))
            if i < num_layers - 1:
                if batch_norm_momentum:
                    layers.append(nn.BatchNorm2d(channels[i + 1],
                                                momentum=batch_norm_momentum))
                layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out.view(-1, 64, 7, 7)
        out = self.layers(out)
        out = self.tanh(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        # 1, 28, 28
        # 64, 14, 14
        # 64, 7, 7
        # 128, 4, 4
        # 128, 4, 4
        # 2048
        # 1
        self.layers = self._make_layers(4,
                                        [1, 64, 64, 128, 128],
                                        [4, 4, 3, 3],
                                        [2, 2, 2, 1],
                                        None, 0.4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layers(self,
                     num_layers,
                     channels,
                     kernel_sizes,
                     strides,
                     batch_norm_momentum,
                     drop_out_rate):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels[i],
                                    channels[i + 1],
                                    kernel_sizes[i],
                                    strides[i],
                                    padding=1,
                                    bias=False))
            if batch_norm_momentum and i > 0:
                layers.append(nn.BatchNorm2d(channels[i + 1],
                                            momentum=batch_norm_momentum))
            layers.append(nn.ReLU(inplace=True))
            if drop_out_rate:
                layers.append(nn.Dropout(p=drop_out_rate))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out