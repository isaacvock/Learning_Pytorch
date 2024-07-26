import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

# Download MNIST fashion data
train_data = datasets.FashionMNIST(
    root = "data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

### Inspect first sample
image, label = train_data[0]

# 1 x 28 x 28, so grayscale (1) 28 x 28 image
image.shape
image[0,0:4,0:4]
label

# Ordering here a bit weird, called NCHW where
# the channel is the first dimension rather than
# the last, even though NHWC is better??

len(train_data.data)
len(train_data.targets)
len(test_data.data)
len(test_data.targets)

class_names = train_data.classes
class_names

# Visualize the data
image, label = train_data[0]
plt.imshow(image.squeeze())
    # Removes size 1 dimensions
plt.title(label);
plt.show()

# Make grayscale
plt.imshow(image.squeeze(), cmap = "gray")
plt.title(class_names[label]);
plt.show()

# View some more
torch.manual_seed(42)
fig = plt.figure(figsize =(9,9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size = [1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);

plt.show()


### Prepare DataLoader to load data into model

from torch.utils.data import DataLoader

# Batch size (number of data points for which 
# gradient is calculated at once)
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True # shuffle data every epoch
                              )

test_dataloader = DataLoader(
    test_data,
    batch_size = BATCH_SIZE,
    shuffle = False)


print(f"Dataloaders: {train_dataloader, test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")


# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape

### Baseline model

# Flatten model (turn image into 1-tensor)
flatten_model = nn.Flatten()

# Get one sample
x = train_features_batch[0]

# Flatten data
output = flatten_model(x)
output.shape

# Model
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__() # creates class from superclass, nn.Module
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)

model_0 = FashionMNISTModelV0(input_shape = 784,
                              hidden_units = 10,
                              output_shape = len(class_names))

model_0.to("cpu")


# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr = 0.1)

### Custom accuracy function
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

### Strategy for timing to compare CPU vs. GPU speed
from timeit import default_timer as timer
def print_train_time(start: float, end: float,
                     device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds") 
    return total_time


### Have to loop through batches and data

# Progress bar spice
from tqdm.auto import tqdm

# Set seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# number of epochs
epochs = 3

# Training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    ### Training
    train_loss = 0

    # Loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumatively add up the loss per epoch

        # 3. Zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
