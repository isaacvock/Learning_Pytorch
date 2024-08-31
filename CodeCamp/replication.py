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


# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
image.shape, label

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

print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
print(f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}")

## Patch embedding is really just a CNN
from torch import nn

patch_size = 16
conv2d = nn.Conv2d(in_channels=3,
                   out_channels=768,
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)

# Can pass image through this layer to get set of
# patches
image_out_of_conv = conv2d(image.unsqueeze(0))
print(image_out_of_conv.shape)

# Flatten patch embedding with Flatten()
# Goal dimension: 196 x 768
# we need to turn 14x14 image into 196 element vector
    # Only want to flattend th "spatial" dimension

flatten = nn.Flatten(start_dim=2,
                     end_dim=3)

image_out_of_conv_flattened = flatten(image_out_of_conv)

# Need to transpose this tensor, effectively
image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(0, 2, 1)


## Make it a Pytorch module!

class PatchEmbedding(nn.Module):

    def __init__(self, 
                 in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim:int=768):
        
        super().__init__()

        # Turn image into patches
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # Layer to flatten
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim = 3)
        
    # Forward pass
    def forward(self, x):
        # Check that inputs are write shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image must be divisible by patch size, image shape: {image_resolution}, patch_size: {patch_size}"

        # Perform forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)
    

# test out model on single image
patchify = PatchEmbedding()
patch_embedded_image = patchify(image.unsqueeze(0)) # Need to add extra batch dimension I guess
print(f"Output patch embedding shape: {patch_embedded_image.shape}")


### COMPONENT 2: CLASS TOKEN EMBEDDING