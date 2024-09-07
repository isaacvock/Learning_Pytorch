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
# Class token is a cheap way to perform flexible 
# classification without requiring a fully connected
# layer at the end for classification. Just prepend
# some token that will be learned and used as the
# classifier to the image.

batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

# Needs to be same size as emedding dimension
class_token = nn.Parameter(
    torch.randn(batch_size, 1, embedding_dimension),
    requires_grad=True
)

patch_embedded_image_with_class_embedding = torch.cat(
    (class_token, patch_embedded_image),
    dim = 1
)

### COMPONENT 3: POSITION EMBEDDING
# Something to specify ordering of patches. In this
# paper, they found that a fancy 2D position embedding
# (i.e., I assume something that tells the model the
# relative x and y positions of each patch) didn't
# outperform a simple 1D embedding (just giving the
# patches a sequential single numeric ID).

# Number of patches
number_of_patches = int((height * width) / patch_size**2)

# Embedding dimension
embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

# Create learnable 1D position embedding
position_embedding = nn.Parameter(
    torch.randn(1,
               number_of_patches + 1,
               embedding_dimension),
    requires_grad=True    
)


# Add them to our embedding
patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding



### COMPONENT 4: Multi-Head Attention (MSA)

class MultiheadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim = embedding_dim,
            num_heads = num_heads,
            dropout = attn_dropout,
            batch_first = True
        )


    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query = x,
            key = x,
            value = x,
            need_weights = False
        )

        return attn_output

# Create an instance of MSABlock
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768, # from Table 1
                                                             num_heads=12) # from Table 1
  

### COMPONENT 5: MULTILAYER PERCEPTRON

class MLPBlock(nn.Module):

    def __init__(self,
               embedding_dim:int=768,
               mlp_size:int=3072,
               dropout:float=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features = embedding_dim,
                      out_features = mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
# Create an instance of MLPBlock
mlp_block = MLPBlock(embedding_dim=768, # from Table 1
                     mlp_size=3072, # from Table 1
                     dropout=0.1) # from Table 3


### COMPONENT 6: TRANSFORMER ENCODER

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim = embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout
        )

        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )

    def forward(self, x):

        # Residual connection for MSA block
        x = self.msa_block(x) + x

        # Residual connection for MLP block
        x = self.mlp_block(x) + x

        return x
    
test_data = torch.randn(1, 768)
tform = TransformerEncoderBlock()
tform(test_data)

##### THE FULL SHEBANG

class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000):
        super().__init__()

        # Make sure image size is divisible by patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # Calculate the number of patches (height * width/patch^2)
        self.num_patches = ( img_size * img_size ) // patch_size ** 2

        # Create learnable class embedding
        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim,
                                                               requires_grad=True))
        
        # Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)
        
        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # Create Transformer Encoder blocks
        # `*`` means "all"
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                      num_heads=num_heads,
                                      mlp_size=mlp_size,
                                      mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)]
        )
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)             
        )

    # Create forward method
    def forward(self, x):

        # Get batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1) # -1 = infer dimension

        # Create patch embedding
        x = self.patch_embedding(x)

        # Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim = 1)

        # Add position embedding to patch embedding
        x = self.position_embedding + x

        # Run embedding dropout
        x = self.embedding_dropout(x)

        # Pass embeddings through transformer
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier
        x = self.classifier(x[:,0])

        return x



# Make sure it works
random_image_tensor = torch.randn(1, 3, 224, 224)
vit = ViT(num_classes=len(class_names))
vit(random_image_tensor)


##### TRAIN MODEL!

from going_modular.going_modular import engine

# Adam optimizer
optimizer = torch.optim.Adam(
    params=vit.parameters(),
    lr=3e-3,
    betas=(0.9, 0.999),
    weight_decay=0.3
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train
results = engine.train(
    model=vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)

from helper_functions import plot_loss_curves

# Plot our ViT model's loss curves
plot_loss_curves(results)

plt.show()