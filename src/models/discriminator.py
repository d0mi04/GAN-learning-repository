import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(ndf*2, 1, 7, 1, 0))
            # bez Sigmoid
        )

    def forward(self, x):
        return self.net(x).view(-1)