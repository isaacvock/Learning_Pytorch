##### CREATING VISION TRANSFORMER FROM SCRATCH

### STEP 0: Getting set up
import torch
import torchvision
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchinfo import summary
from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"
device

##### STEP 1: Download data (already done)

image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"


##### STEP 2: Create datasets and DataLoaders

# Transforms for image
IMG_SIZE = 224
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Turn images into DataLoaders
BATCH_SIZE = 32
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)


##### STEP 3: Read the paper; understand the architecture

### COMPONENT 1: PATCH EMBEDDING

# Create exmaple values
height = 224
width = 224
color_channels = 3
patch_size = 16

# Calculate N (number of patches)
number_of_patches = (width * height) / (patch_size ** 2)

# Input: 2D image of size H x W x C
# Output: flattened 2D patches with size N x (P^2 * C)
embedding_layer_input_shape = (height, width, color_channels)
embedding_layer_output_shape = (number_of_patches,
                                patch_size**2 * color_channels)

