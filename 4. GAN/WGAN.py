import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight_randn(submodule):
    if isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.Linear):
        torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02)

class Generator(nn.Module):
    def __init__(self, z_dim, initial_linear_layer_size):
        super(Generator, self).__init__()
        self.initial_linear_layer_size = initial_linear_layer_size
        # 100
        # 512
        # 128, 4, 4
        # 128, 8, 8
        # 64, 16, 16
        # 32, 32, 32
        # 3, 32, 32
        self.linear = nn.Linear(z_dim, 2048)
        self.batch_norm = nn.BatchNorm1d(2048)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.layers = self._make_layers(4,
                                        [2, 2, 2, 1],
                                        [128, 128, 64, 32, 3],
                                        [3, 3, 3, 3],
                                        [1, 1, 1, 1],
                                        0.8, None)
        self.tanh = nn.Tanh()
        
        self.linear.apply(init_weight_randn)
        self.layers.apply(init_weight_randn)
        
    def _make_layers(self,
                     num_layers,
                     upsamples,
                     channels,
                     kernel_sizes,
                     strides,
                     batch_norm_momentum,
                     drop_out_rate):
        layers = []
        for i in range(num_layers):
            if upsamples[i] > 1:
                layers.append(nn.Upsample(scale_factor=upsamples[i]))
            layers.append(nn.Conv2d(channels[i],
                                    channels[i + 1],
                                    kernel_sizes[i],
                                    strides[i],
                                    padding=1,
                                    bias=False))
            # layers.append(nn.ConvTranspose2d(channels[i],
            #                                  channels[i + 1],
            #                                  kernel_sizes[i],
            #                                  strides[i],
            #                                  bias=False))
            if upsamples[i] > 1:
                layers.append(nn.BatchNorm2d(channels[i + 1],
                                             momentum=batch_norm_momentum))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            if drop_out_rate is not None:
                layers.append(nn.Dropout(p=drop_out_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = out.view(self.initial_linear_layer_size)
        out = self.layers(out)
        out = self.tanh(out)
        return out
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # 3, 32, 32
        # 32, 16, 16
        # 64, 8, 8
        # 128, 4, 4
        # 128, 4, 4
        # 2048
        # 100
        self.layers = self._make_layers(4,
                                        [3, 32, 64, 128, 128],
                                        [3, 3, 3, 3],
                                        [2, 2, 2, 1],
                                        None, None)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2048, 1)
        
        self.linear.apply(init_weight_randn)
        self.layers.apply(init_weight_randn)
        
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
            #layers.append(nn.BatchNorm2d(channels[i + 1]))
                                         #momentum=batch_norm_momentum))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if drop_out_rate is not None:
                layers.append(nn.Dropout(p=drop_out_rate))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        out = self.flatten(out)
        out = self.linear(out)
        return out