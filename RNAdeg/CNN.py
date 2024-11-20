### PURPOSE OF THIS SCRIPT
# Build a CNN trained on promoter sequences to predict isoform stabilities, because
# I am really curious as to how well such a prediction will work.

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import math
import torch.utils.data as data_utils

device = "cuda" if torch.cuda.is_available() else "cpu"


### Load in data

features = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\RNAdeg_feature_table.csv")
features_filter = features.loc[features['avg_lkd_se'] < math.exp(-2)]

promoters = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\mix_trimmed\\filtered\\threeprimeUTR_seqs.csv")

full_data = pd.merge(features_filter, promoters, on = ['transcript_id'], how = 'inner')

train_data = full_data[~full_data['seqnames'].isin(['chr1', 'chr22'])]
test_data = full_data[full_data['seqnames'].isin(['chr1', 'chr22'])]

char_to_index = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
}


### One-hot encode features
def onehote_np(seq, desired_len = 500):
    if len(seq) < desired_len:
        seq = seq + 'N' * (desired_len - len(seq))

    else:
        seq = seq[:desired_len]
    seq2 = [char_to_index[i] for i in seq]
    return torch.from_numpy(np.eye(5)[seq2])

train_tensor = [onehote_np(seq) for seq in train_data['seq']]
train_tensor_3d = torch.stack(train_tensor)

test_tensor = [onehote_np(seq) for seq in test_data['seq']]
test_tensor_3d = torch.stack(test_tensor)


train_tensor_3d.shape


### Convert log(kdeg) to Pytorch tensor

train_targets = torch.tensor(train_data['log_kdeg'].values)
test_targets = torch.tensor(test_data['log_kdeg'].values)

### Create DataLoader

train = data_utils.TensorDataset(train_tensor_3d.permute(0, 2, 1).float().to(device), train_targets.float().to(device))
train_loader = data_utils.DataLoader(train, batch_size = 32, shuffle=True)

test = data_utils.TensorDataset(test_tensor_3d.permute(0, 2, 1).float().to(device), test_targets.float().to(device))
test_loader = data_utils.DataLoader(test, batch_size = 32, shuffle=True)


# Shape of data: [11963, 2200, 5]
# Basically single channel image that is 2200 pixels wide and 5 pixels tall
train_features_batch, train_labels_batch = next(iter(train_loader))
train_features_batch.shape, train_labels_batch.shape

seq, label = train_features_batch[1], train_labels_batch[1]



### Build model

class simpleCNN(nn.Module):
    """
    Just want to get a CNN working with this data
    """
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                seq_len: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size= 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,
                            stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units, 
                kernel_size = 3, 
                padding =1
                ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units, 
                kernel_size = 3, 
                padding =1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1, 2),
            nn.Linear(seq_len * hidden_units, 1)
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(self.block_2(self.block_1(x)))
        return x


simple_model = simpleCNN(
    input_shape = 5,
    hidden_units=64,
    seq_len = 125
).to(device)


### SANDBOX: Walk through each layer of model
input_shape = 5
hidden_units = 32

## First block
test_block_1 = nn.Sequential(
    nn.Conv1d(
        in_channels = input_shape,
        out_channels = hidden_units,
        kernel_size= 3,
        stride = 1,
        padding = 1
    ),
    nn.ReLU(),
    nn.Conv1d(
        in_channels=hidden_units,
        out_channels=hidden_units,
        kernel_size=3,
        stride = 1,
        padding = 1
    ),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2,
                    stride=2)
).to(device)

seq.shape
# 1 x 5 x 2200

train_features_batch.shape

output_1 = test_block_1(train_features_batch)
output_1.shape
# 1 x 3 x 1100


## 2nd block
test_block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units, 
                kernel_size = 3, 
                padding =1
                ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units, 
                kernel_size = 3, 
                padding =1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        ).to(device)

output_1.shape
output_2 = test_block_2(output_1)
output_2.shape
# 1 x 2 x 551


## 3rd block
test_block_3 = nn.Sequential(
            nn.Flatten(1, 2),
            nn.Linear(hidden_units * 550, 1)
        ).to(device)

test_flatten = nn.Flatten(1, 2)

test_flatten(output_2).shape

output_2.shape
output_3 = test_block_3(output_2)
output_3.shape

### Setup loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    params=simple_model.parameters(),
    lr = 0.1
)


epochs = 10

train_losses = [0]*10
for epoch in range(epochs):

    simple_model.train()


    train_loss = 0
    for batch, (X, kdeg) in enumerate(train_loader):

        optimizer.zero_grad()


        kdeg_pred = simple_model(X)

        loss = loss_fn(kdeg_pred.squeeze(), kdeg)
        train_loss += loss


        loss.backward()

        optimizer.step()

    train_loss /= len(train_loader)
    train_losses[epoch] = train_loss.to('cpu').detach().numpy()

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss = 0
    simple_model.eval()
    with torch.inference_mode():
        for X, y in test_loader:
            # 1. Forward pass
            test_pred = simple_model(X)
           
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred.squeeze(), y) # accumulatively add up the loss per epoch

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_loader)


train_loss
test_loss

### Now plot estimate vs. truth
simple_model.eval()

predicted_kdeg = []
true_kdeg = []

with torch.inference_mode():
    for X, y in train_loader:
        
        y_pred = simple_model(X)

        predicted_kdeg.extend(y_pred.squeeze().cpu().detach().numpy())
        true_kdeg.extend(y.squeeze().cpu().detach().numpy())


plt.scatter(true_kdeg,
            predicted_kdeg,
            alpha = 0.5)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()


plt.scatter(list(range(epochs)),
            train_losses)
plt.show()