import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3136, 2)
        )
        
        # Decoder
        self.linear = nn.Linear(2, 3136)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        out = self.encoder(x)
        return out
    
    def decode(self, x):
        out = self.linear(x)
        out = out.view(-1, 64, 7, 7)
        out = self.decoder(out)
        
        return out
            
    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out
    