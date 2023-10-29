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
# for X, y in test_dataloader:
#     # Not sure what N and C are here, but H and W are height and width
#     # N seems to be number of items since it matches with batch size, but C im not sure of
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

device = (
    "cuda" 
    if torch.cuda.is_available()
    else 
        "mps" if torch.backends.mps.is_available()
    else 
        "cpu"        
)

# Should be cuda
print(f"Using device {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 28 * 28 inputs -> 512 ouputs
            nn.Linear(28 * 28, 512),
            # max(0, x) is most likely what this function uses
            nn.ReLU(),
            # 512 inputs -> 512 outputs
            nn.Linear(512, 512),
            # max(0, x) is most likely what this function uses
            nn.ReLU(),
            # 512 inputs -> 10 outputs
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

# Loss function for figuring out how far off the neural network is from the right guess ... I think???
loss_fn = nn.CrossEntropyLoss()
# Function used to update the network over time to get better results. Maybe this is run when back propagating? No idea
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Pass the data over to the device that will use it
        # Data can't be shared between CPU and GPU directly from my understanding
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        # So yeah, the optimizer function is called when going backwards, but I have no idea how optimizer.zero_grad() works or what it even does
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= sum
    print(f"Test error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")