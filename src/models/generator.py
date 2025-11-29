import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, ngf=64, nc=1):  # nc=1 → grayscale
        super().__init__()

        self.net = nn.Sequential(
            # z → (ngf*4) × 7 × 7
            nn.ConvTranspose2d(z_dim, ngf*4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # (ngf*4) × 7×7 → (ngf*2) × 14×14
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # (ngf*2) × 14×14 → (nc) × 28×28
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
