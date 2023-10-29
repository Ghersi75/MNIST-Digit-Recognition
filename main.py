import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# If data is already downloaded, the code below wont download the data again

# Download training data from open dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Download test data
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Batch size for DataLoader return size
batch_size = 64

# Create DataLoader objects
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Print shape of data
for X, y in test_dataloader:
    # Not sure what N and C are here, but H and W are height and width
    # N seems to be number of items since it matches with batch size, but C im not sure of
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
