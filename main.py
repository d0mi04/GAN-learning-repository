from src.datasets import create_dataloader
from src.train import train

def main():
    dataloader = create_dataloader("data/")
    train(dataloader)

if __name__ == "__main__":
    main()
