import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from src.models.generator import Generator
from src.models.discriminator import Discriminator

def weights_init(m):
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def train(dataloader, epochs=50, z_dim=128, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    for epoch in range(epochs):
        for real, _ in dataloader:
            real = real.to(device)
            b = real.size(0)

            # -----------------------
            #  Train Discriminator
            # -----------------------
            optD.zero_grad()

            noise = torch.randn(b, z_dim, 1, 1, device=device)
            fake = G(noise).detach()

            # hinge loss
            d_real = D(real)
            d_fake = D(fake)

            lossD = torch.mean(torch.relu(1 - d_real)) + \
                    torch.mean(torch.relu(1 + d_fake))

            lossD.backward()
            optD.step()

            # -----------------------
            #  Train Generator
            # -----------------------
            optG.zero_grad()

            fake = G(noise)
            d_fake = D(fake)

            lossG = -torch.mean(d_fake)

            lossG.backward()
            optG.step()

        # save preview images
        save_image((G(fixed_noise) + 1) / 2, f"outputs-MNIST/epoch_{epoch}.png")
        print(f"[{epoch}]  lossD={lossD.item():.4f}  lossG={lossG.item():.4f}")
