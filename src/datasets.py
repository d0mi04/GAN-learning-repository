from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloader(data_path="data/", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        mnist,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
