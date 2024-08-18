##### CHAPTER 6: TRANSFER LEARNING

### Step 0: Getting setup

import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary

from going_modular.going_modular import data_setup, engine

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


### Step 1: Get data
# Already did this previously so just have to define directories

import os
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

train_dir = image_path / "train"
test_dir = image_path / "test"


### Step 2: Create Datasets and DataLoaders

# Create a transforms pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Reshape images to 224x224
    transforms.ToTensor(), # Turn image values to between 0 & 1
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]) # Calcuated from data by someone else
])

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform= manual_transforms,
    batch_size=32
)

train_dataloader, test_dataloader, class_names


### STEP 3: Get a pretrained model

# Get weigths from our designed model
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Use torchinfo to get succinct summary
summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 

# Transfer learning vibe is to freeze feature layers and
# change the output layer to your need
for param in model.features.parameters():
    param.requires_grad = False # Don't train these; "freeze" layers

## Make classifier layer compatible with our use case
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# What does shape of our classifier output have to be
output_shape = len(class_names)

# New classifier layer
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)
).to(device)


### Step 4: Train model

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Setup training and save the results
results = engine.train(
    model = model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5,
    device=device
)