### GOAL OF THIS SCRIPT
# Implement simple neural net model
# to predict RNA stability

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


########## NEW FEATURE TABLE ###############


### Build model
device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleMLP(nn.Module):
    def __init__(self,
                 embedding_dim:int=30,
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
degdata = pd.read_csv('C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\RNAdeg_feature_table.csv')

data_cols = ['NMD_both',
                'log10_3primeUTR',
                'log_ksyn',
                'log10_5primeUTR',
                'log10_length',
                'log10_numexons']
obs_col = ['log_kdeg']

degdata_test = degdata[degdata['seqnames'].isin( ['chr1', 'chr22'])]

degdata_train = degdata[~degdata['seqnames'].isin(['chr1', 'chr22'])]


train_tensor = torch.tensor(degdata_train[data_cols].to_numpy(),
                                   dtype=torch.float32)
train_label = torch.tensor(degdata_train[obs_col].to_numpy(),
                           dtype=torch.float32)


test_tensor = torch.tensor(degdata_test[data_cols].to_numpy(),
                                   dtype=torch.float32)
test_label = torch.tensor(degdata_test[obs_col].to_numpy(),
                           dtype=torch.float32)


### Train the model!

# Loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(simple_nn.parameters(), lr=0.1)

epochs = 300

# Put train and test data on one device
train_tensor, train_label = train_tensor.to(device), train_label.to(device)
test_tensor, test_label = test_tensor.to(device), test_label.to(device) 

def mse_fn(lkdeg_true, lkdeg_pred):
    mse = sum((lkdeg_true - lkdeg_pred) ** 2)/len(lkdeg_true)
    return mse


# ## Gut check to see if my model works
# testlin = nn.Linear(in_features = 6, out_features = 20).to(device)
# testlin2 = nn.Linear(in_features = 20, out_features = 20).to(device)
# testlin3 = nn.Linear(in_features = 20, out_features = 1).to(device)
# testrelu = nn.ReLU().to(device)

# testlin(train_tensor)
# testrelu(testlin(train_tensor))
# testlin2(testrelu(testlin(train_tensor)))
# testrelu(testlin2(testrelu(testlin(train_tensor))))
# testlin3(testrelu(testlin2(testrelu(testlin(train_tensor)))))

### Train
train_losses = [1] * epochs
train_mses = [1] * epochs
test_losses = [1] * epochs
test_mses = [1] * epochs

for epoch in range(epochs):
    # 1. Forward pass
    log_kdeg_guess = simple_nn(train_tensor)

    # Calculate loss and accuracy
    loss = loss_fn(log_kdeg_guess, train_label)
    mse = mse_fn(train_label, log_kdeg_guess)

    train_losses[epoch] = loss.item()
    train_mses[epoch] = mse.item()

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

        test_losses[epoch] = test_loss.item()
        test_mses[epoch] = test_mse.item()

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.5f}, Accuracy: {mse.item():.2f}% | Test loss: {test_loss.item():.5f}, Test acc: {test_mse.item():.2f}%")
 

### RMSE
np.sqrt(test_losses[epochs-1])
    # Best I did was 0.735

### Plot losses
fig, ax = plt.subplots()
ax.plot(list(range(1, epochs+1)), test_mses)
fig.show()

fig, ax = plt.subplots()
ax.plot(list(range(1, epochs+1)), train_mses)
fig.show()


### Make predictions and compare them to the truth
simple_nn.eval()
with torch.inference_mode():

    lkdeg_pred = simple_nn(test_tensor)


lkdeg_pred_list = lkdeg_pred.squeeze().to('cpu').tolist()
lkdeg_truth = test_label.squeeze().to('cpu').tolist()


plt.scatter(lkdeg_truth, lkdeg_pred_list)
plt.plot(np.linspace(min(lkdeg_truth), max(lkdeg_truth), 100),
         np.linspace(min(lkdeg_truth), max(lkdeg_truth), 100),
         color = 'red', linestyle = '-', linewidth =2)
plt.show()



########## OLD FEATURE TABLE ###############

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

cols_to_log = ['exonic_length',
               'num_exons',
               'fiveprimeUTR_lngth',
               'threeprimeUTR_lngth',
               'stop_to_lastEJ']

for col in cols_to_log:
    degdata[f'{col}_log'] = np.log(abs(degdata[col])+1)

cols_to_standardize = ['exonic_length_log',
                        'num_exons_log',
                        'fiveprimeUTR_lngth_log',
                        'threeprimeUTR_lngth_log',
                        'stop_to_lastEJ_log',
                        'log_ksyn',
                        'log_kdeg']

data_cols = ['exonic_length_log',
                        'num_exons_log',
                        'fiveprimeUTR_lngth_log',
                        "threeprimeUTR_lngth_log",
                        'stop_to_lastEJ_log',
                        'log_ksyn']
obs_col = ['log_kdeg']
all_cols = data_cols + obs_col

degdata[cols_to_standardize] = (degdata[cols_to_standardize] - degdata[cols_to_standardize].mean())/degdata[cols_to_standardize].std()


degdata_test = degdata[degdata['seqnames'].isin( ['chr1', 'chr22'])]
#degdata_test = degdata[degdata['seqnames'].isin( ['chr22'])]
degdata_test = degdata_test[degdata_test[data_cols].notnull().all(1)]

degdata_train = degdata[~degdata['seqnames'].isin(['chr1', 'chr22'])]
#degdata_train = degdata[degdata['seqnames'].isin( ['chr21'])]
degdata_train = degdata_train[degdata_train[data_cols].notnull().all(1)]


train_tensor = torch.tensor(degdata_train[data_cols].to_numpy(),
                                   dtype=torch.float32)
train_label = torch.tensor(degdata_train[obs_col].to_numpy(),
                           dtype=torch.float32)


test_tensor = torch.tensor(degdata_test[data_cols].to_numpy(),
                                   dtype=torch.float32)
test_label = torch.tensor(degdata_test[obs_col].to_numpy(),
                           dtype=torch.float32)


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


### Gut check to see if my model works
# testlin = nn.Linear(in_features = 6, out_features = 20).to(device)
# testlin2 = nn.Linear(in_features = 20, out_features = 20).to(device)
# testlin3 = nn.Linear(in_features = 20, out_features = 1).to(device)
# testrelu = nn.ReLU().to(device)

# testlin(train_tensor)
# testrelu(testlin(train_tensor))
# testlin2(testrelu(testlin(train_tensor)))
# testrelu(testlin2(testrelu(testlin(train_tensor))))
# testlin3(testrelu(testlin2(testrelu(testlin(train_tensor)))))

### Train
train_losses = [1] * epochs
train_mses = [1] * epochs
test_losses = [1] * epochs
test_mses = [1] * epochs

for epoch in range(epochs):
    # 1. Forward pass
    log_kdeg_guess = simple_nn(train_tensor)

    # Calculate loss and accuracy
    loss = loss_fn(log_kdeg_guess, train_label)
    mse = mse_fn(train_label, log_kdeg_guess)

    train_losses[epoch] = loss.item()
    train_mses[epoch] = mse.item()

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

        test_losses[epoch] = test_loss.item()
        test_mses[epoch] = test_mse.item()

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.5f}, Accuracy: {mse.item():.2f}% | Test loss: {test_loss.item():.5f}, Test acc: {test_mse.item():.2f}%")
 


### Plot losses
fig, ax = plt.subplots()
ax.plot(list(range(1, epochs+1)), test_mses)
fig.show()

fig, ax = plt.subplots()
ax.plot(list(range(1, epochs+1)), train_mses)
fig.show()


### Make predictions and compare them to the truth
simple_nn.eval()
with torch.inference_mode():

    lkdeg_pred = simple_nn(test_tensor)


lkdeg_pred_list = lkdeg_pred.squeeze().to('cpu').tolist()
lkdeg_truth = test_label.squeeze().to('cpu').tolist()


plt.scatter(lkdeg_truth, lkdeg_pred_list)
plt.plot(np.linspace(min(lkdeg_truth), max(lkdeg_truth), 100),
         np.linspace(min(lkdeg_truth), max(lkdeg_truth), 100),
         color = 'red', linestyle = '-', linewidth =2)
plt.show()