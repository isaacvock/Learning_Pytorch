### GOAL OF THIS SCRIPT
# Implement simple neural net model
# to predict RNA stability

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


### Build model
device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleMLP(nn.Module):
    def __init__(self,
                 embedding_dim:int=20,
                 in_features:int=6):
        super().__init__()

        self.layer_1 = nn.Linear(in_features = in_features, out_features = embedding_dim)
        self.layer_2 = nn.Linear(in_features = embedding_dim, out_features = embedding_dim)
        self.layer_3 = nn.Linear(in_features = embedding_dim, out_features = 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    

simple_nn = SimpleMLP().to(device)

### Load data
degdata = pd.read_csv('C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\RNAdeg_dataset.csv')


degdata_test = degdata[degdata['seqnames'].isin( ['chr1', 'chr22'])]
degdata_train = degdata[~degdata['seqnames'].isin(['chr1', 'chr22'])]


train_tensor = torch.tensor(degdata_train[['exonic_length',
                                   'num_exons',
                                   'fiveprimeUTR_lngth',
                                   'threeprimeUTR_lngth',
                                   'stop_to_lastEJ',
                                   'log_ksyn']].to_numpy(),
                                   dtype=torch.float32)
train_label = torch.tensor(degdata_train[['log_kdeg']].to_numpy(),
                           dtype=torch.float32)


test_tensor = torch.tensor(degdata_test[['exonic_length',
                                   'num_exons',
                                   'fiveprimeUTR_lngth',
                                   'threeprimeUTR_lngth',
                                   'stop_to_lastEJ',
                                   'log_ksyn']].to_numpy(),
                                   dtype=torch.float32)
test_label = torch.tensor(degdata_test[['log_kdeg']].to_numpy(),
                           dtype=torch.float32)

### Should standardize features to avoid overflow when calculating MSE



### Train the model!

# Loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(simple_nn.parameters(), lr=0.1)

epochs = 100

# Put train and test data on one device
train_tensor, train_label = train_tensor.to(device), train_label.to(device)
test_tensor, test_label = test_tensor.to(device), test_label.to(device) 

def mse_fn(lkdeg_true, lkdeg_pred):
    mse = sum((lkdeg_true - lkdeg_pred) ** 2)/len(lkdeg_true)
    return mse

for epoch in range(epochs):
    # 1. Forward pass
    log_kdeg_guess = simple_nn(train_tensor)

    # Calculate loss and accuracy
    loss = loss_fn(log_kdeg_guess, train_label)
    mse = mse_fn(train_label, log_kdeg_guess)

    # Optimizer incantation
    optimizer.zero_grad()

    # Backwards prop
    loss.backward()

    # Update
    optimizer.step()

    # Test
    simple_nn.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_lkdegs = simple_nn(test_tensor)

        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_lkdegs, test_label)

        test_mse = mse_fn(test_label, test_lkdegs)

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {mse:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_mse:.2f}%")
 