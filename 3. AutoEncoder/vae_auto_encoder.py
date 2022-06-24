import torch.nn as nn
import torch
from torchvision.transforms import Lambda


class VAEAutoEncoder(nn.Module):
    def __init__(self,
                 num_layers,
                 encoder_channels,
                 encoder_kernel_sizes,
                 encoder_strides,
                 decoder_channels,
                 decoder_kernel_sizes,
                 decoder_strides,
                 linear_sizes,
                 view_size,
                 use_batch_norm=False,
                 use_dropout=False):
        
        super(VAEAutoEncoder, self).__init__()
        
        # encoder
        self.view_size = view_size
        self.encoder = self._make_encoder_layers(num_layers,
                                                 encoder_channels,
                                                 encoder_kernel_sizes,
                                                 encoder_strides,
                                                 use_batch_norm,
                                                 use_dropout)
        
        self.mu_layer = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.log_var_layer = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.sampling_layer = self.sampling()
        
        # decoder
        self.linear = nn.Linear(linear_sizes[1], linear_sizes[0])
        self.decoder = self._make_decoder_layers(num_layers,
                                                 decoder_channels,
                                                 decoder_kernel_sizes,
                                                 decoder_strides,
                                                 use_batch_norm,
                                                 use_dropout)
        
    def _make_encoder_layers(self,
                             num_layers,
                             channels,
                             kernel_sizes,
                             strides,
                             use_batch_norm,
                             use_dropout):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels[i],
                                    channels[i + 1],
                                    kernel_sizes[i],
                                    strides[i],
                                    padding=1,
                                    bias=False))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.LeakyReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(p=0.25))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)
        
    def _make_decoder_layers(self,
                             num_layers,
                             channels,
                             kernel_sizes,
                             strides,
                             use_batch_norm,
                             use_dropout):
        layers = []
        for i in range(num_layers):
            if strides[i] == 1:
                layers.append(nn.ConvTranspose2d(channels[i],
                                                 channels[i + 1],
                                                 kernel_sizes[i],
                                                 strides[i],
                                                 padding=1,
                                                 bias=False))
            else:
                layers.append(nn.ConvTranspose2d(channels[i],
                                                 channels[i + 1],
                                                 kernel_sizes[i],
                                                 strides[i],
                                                 padding=1,
                                                 output_padding=1,
                                                 bias=False))
                
            if i < num_layers - 1:
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(channels[i + 1]))
                layers.append(nn.LeakyReLU(inplace=True))
                if use_dropout:
                    layers.append(nn.Dropout(p=0.25))
            else:
                layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
        
    def sampling(self):
        return (lambda mu, log_var:
            torch.randn_like(log_var) * torch.exp(log_var * 0.5) + mu)
        
    def encode(self, x):
        out = self.encoder(x)
        mu = self.mu_layer(out)
        log_var = self.log_var_layer(out)
        out = self.sampling_layer(mu, log_var)
        return out, mu, log_var
    
    def decode(self, x):
        out = self.linear(x)
        out = out.view(self.view_size)
        out = self.decoder(out)
        return out
            
    def forward(self, x):
        out, mu, log_var = self.encode(x)
        out = self.decode(out)
        return out, mu, log_var
    