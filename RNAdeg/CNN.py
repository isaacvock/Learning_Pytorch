### PURPOSE OF THIS SCRIPT
# Implement the Saluki model (Agarwal and Kelley 2022) in Pytorch
# so I can learn but also so I can retrain it on improved isoform-level
# degradation rate constant estimates

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import math
import torch.utils.data as data_utils
import statistics
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


##### SALUKI OHE 

features = pd.read_csv("C:\\Users\\isaac\\Box\\TimeLapse\\Annotation_gamut\\DataTables\\RNAdeg_data_model_features.csv")

ss_df = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\mix_trimed_fivepss_indices.csv")


train_data = features[~features['chrom'].isin(['chr1', 'chr22'])]
test_data = features[features['chrom'].isin(['chr1', 'chr22'])]

print(train_data.columns)


def num_to_one_hot(x, bits=4):
    """
    Equivalent to R's diag(1L, bits)[, x]
    x is a 1D array of integers (1-based indices).
    Returns a 2D matrix of shape (bits, len(x)) where each column is a one-hot vector.
    """
    # Convert 1-based indexing to 0-based for Python
    x_zero_based = np.array(x) - 1
    # Create an identity matrix and index columns
    # shape: (bits, len(x))
    one_hot = np.eye(bits)[x_zero_based].T
    return one_hot

def OHE(seq):
    """
    One-hot encodes a DNA sequence. 
    Maps A->1, C->2, G->3, T->4 and then creates a one-hot (4xN) matrix.
    """
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    indices = [mapping[base] for base in seq]
    return num_to_one_hot(indices, bits=4)

def OHE_phase(cds):
    """
    Computes the phase OHE for the coding region.
    Returns a 1D array (length = len(cds)), with '1' at positions modulo 3 == 1 (1-based), else '0'.
    """
    length = len(cds)
    positions = np.arange(1, length+1)  # 1-based indexing
    # positions % 3 == 1 -> phase = 1 else 0
    phase = np.where((positions % 3) == 1, 1, 0)
    return phase

def OHE_saluki(fivep, cds, threep, splices, max_nt=12288):
    """
    Translates the OHE_saluki R function into Python.
    fivep, cds, threep: DNA sequences (strings)
    splices: list/array of splice site positions (1-based along the concatenated transcript)
    max_nt: desired length to pad or truncate to.
    Returns: 6 x max_nt numpy array.
    """
    # One-hot encode each region (4 x length_of_region)
    fivep_ohe = OHE(fivep)
    cds_ohe = OHE(cds)
    threep_ohe = OHE(threep)

    # Create the phase row
    total_length = len(fivep) + len(cds) + len(threep)
    phase_ohe = np.zeros(total_length, dtype=int)
    phase_ohe[len(fivep):len(fivep)+len(cds)] = OHE_phase(cds)

    # Create the splice row
    splice_ohe = np.zeros(total_length, dtype=int)
    # # splices are 1-based positions; ensure not out of range
    # for s in splices:
    #     if 1 <= s <= total_length:
    #         splice_ohe[s-1] = 1
    for s in splices:
        if 1 <= s <= max_nt:
            splice_ohe[s-1] = 1

    # Combine OHE into one matrix
    # final_OHE shape initially: 4 rows (nucleotides), plus 1 row (phase), plus 1 row (splice) = 6 rows
    final_OHE = np.vstack([
        np.hstack([fivep_ohe, cds_ohe, threep_ohe]),
        phase_ohe[np.newaxis, :],
        splice_ohe[np.newaxis, :]
    ])

    # Pad or truncate
    current_len = final_OHE.shape[1]
    if current_len < max_nt:
        # Pad with zeros on the right
        pad_cols = max_nt - current_len
        final_OHE = np.hstack([final_OHE, np.zeros((6, pad_cols), dtype=int)])
    elif current_len > max_nt:
        # Truncate to max_nt
        final_OHE = final_OHE[:, :max_nt]

    return final_OHE


tensors = []
for idx, row in train_data.iterrows():
    fivep = row['fiveputr_seq']
    cds = row['CDS_seq']
    threep = row['threeputr_seq']
    transcript_id = row['transcript_id']
    splices = ss_df.loc[ss_df['transcript_id'] == transcript_id, 'fivepss'].tolist()

    encoded = OHE_saluki(fivep, cds, threep, splices)
    tensors.append(encoded)


train_tensor_3d = torch.tensor(np.array(tensors))


tensors = []
for idx, row in test_data.iterrows():
    fivep = row['fiveputr_seq']
    cds = row['CDS_seq']
    threep = row['threeputr_seq']
    transcript_id = row['transcript_id']
    splices = ss_df.loc[ss_df['transcript_id'] == transcript_id, 'fivepss'].tolist()

    encoded = OHE_saluki(fivep, cds, threep, splices)
    tensors.append(encoded)

test_tensor_3d = torch.tensor(np.array(tensors))


### Convert log(kdeg) to Pytorch tensor

train_targets = torch.tensor((train_data['log_kdeg_DMSO'].values - statistics.mean(train_data['log_kdeg_DMSO'].values)) / np.std(train_data['log_kdeg_DMSO'].values))
test_targets = torch.tensor((test_data['log_kdeg_DMSO'].values - statistics.mean(test_data['log_kdeg_DMSO'].values)) / np.std(test_data['log_kdeg_DMSO'].values))

# Normalize


### Create DataLoader

train = data_utils.TensorDataset(train_tensor_3d.permute(0, 2, 1).float().to(device), train_targets.float().to(device))
train_loader = data_utils.DataLoader(train, batch_size = 64, shuffle=True)

test = data_utils.TensorDataset(test_tensor_3d.permute(0, 2, 1).float().to(device), test_targets.float().to(device))
test_loader = data_utils.DataLoader(test, batch_size = 64, shuffle=True)


# Shape of data: [11963, 2200, 5]
# Basically single channel image that is 2200 pixels wide and 5 pixels tall
train_features_batch, train_labels_batch = next(iter(train_loader))
train_features_batch.shape, train_labels_batch.shape

seq, label = train_features_batch[1], train_labels_batch[1]



### Build model

def calc_size_after_pool(Lin, ksize, padding = 0,
                         dilation = 1, stride = None):
    
    if stride is None:
        stride = ksize

    Lout = ((Lin + 2 * padding - dilation * (ksize - 1) - 1) / stride) + 1
    Lout = math.floor(Lout)
    return Lout

class SalukiCNN(nn.Module):
    """
    Just want to get a CNN working with this data
    """
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                seq_len: int,
                ksize: int = 2,
                dilation: int = 1,
                padding: int = 0,
                stride: int = None):
        
        if stride is None:
            stride = ksize

        super().__init__()
        self.block_1 = nn.Sequential(
            nn.LayerNorm([input_shape, seq_len]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size= 5,
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(seq_len, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)


        self.block_2 = nn.Sequential(
            nn.LayerNorm([hidden_units, Lout]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size= 5
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(Lout, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)

        self.block_3 = nn.Sequential(
            nn.LayerNorm([hidden_units, Lout]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size= 5
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(Lout, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)

        self.block_4 = nn.Sequential(
            nn.LayerNorm([hidden_units, Lout]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size= 5
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(Lout, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)

        self.block_5 = nn.Sequential(
            nn.LayerNorm([hidden_units, Lout]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size= 5
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(Lout, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)


        self.block_6 = nn.Sequential(
            nn.LayerNorm([hidden_units, Lout]),
            nn.ReLU(),
            nn.Conv1d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size= 5
            ),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=ksize,
                            stride=stride,
                            dilation = dilation,
                            padding = padding)
        )

        Lout = calc_size_after_pool(Lout, ksize = 5, stride = 1)
        Lout = calc_size_after_pool(Lout, ksize = ksize,
                                    stride = stride,
                                    padding = padding,
                                    dilation = dilation)                                  

        self.classifier = nn.Sequential(
            nn.Flatten(1, 2),
            nn.Linear(Lout * hidden_units, 1)
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(self.block_6(self.block_5(self.block_4(self.block_3(self.block_2(self.block_1(x)))))))
        return x



simple_model = SalukiCNN(
    input_shape = 6,
    hidden_units=12,
    seq_len = 12288
).to(device)

simple_model(train_features_batch.transpose(1, 2))
train_features_batch.transpose(1, 2).shape

### Setup loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    params=simple_model.parameters(),
    betas=(0.9, 0.98)
)


epochs = 25

train_losses = [0]*epochs
test_losses = [0]*epochs
for epoch in range(epochs):

    simple_model.train()


    train_loss = 0
    for batch, (X, kdeg) in enumerate(train_loader):

        optimizer.zero_grad()


        kdeg_pred = simple_model(X.transpose(1,2))

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
            test_pred = simple_model(X.transpose(1,2))
           
            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred.squeeze(), y) # accumulatively add up the loss per epoch

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_loader)
        test_losses[epoch] = test_loss.to('cpu').detach().numpy()



train_loss
test_loss

### Now plot estimate vs. truth
simple_model.eval()

predicted_kdeg = []
true_kdeg = []

predicted_kdeg_test = []
true_kdeg_test = []

final_test_loss = 0
with torch.inference_mode():
    for X, y in train_loader:
        
        y_pred = simple_model(X.transpose(1,2))

        predicted_kdeg.extend(y_pred.squeeze().cpu().detach().numpy())
        true_kdeg.extend(y.squeeze().cpu().detach().numpy())

    for X, y in test_loader:
        
        y_pred = simple_model(X.transpose(1,2))

        final_test_loss = final_test_loss + loss_fn(y_pred.squeeze(), y)

        predicted_kdeg_test.extend(y_pred.squeeze().cpu().detach().numpy())
        true_kdeg_test.extend(y.squeeze().cpu().detach().numpy())


np.mean(true_kdeg)

plt.scatter(true_kdeg,
            predicted_kdeg,
            alpha = 0.5)
plt.xlabel('True values (train)')
plt.ylabel('Predicted values (train)')
plt.show()


plt.scatter(true_kdeg_test,
            predicted_kdeg_test,
            alpha = 0.5)
plt.xlabel('True values (test)')
plt.ylabel('Predicted values (test)')
plt.show()


plt.scatter(list(range(epochs)),
            train_losses)
plt.show()

plt.scatter(list(range(epochs)),
            test_losses)
plt.show()








##### ME HACKING AROUND


### Load in data

features = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\mix_trimmed\\filtered\\RNAdeg_feature_table.csv")
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
def onehote_np(seq, desired_len = 10000):
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

train_targets = torch.tensor((train_data['log_kdeg'].values - statistics.mean(train_data['log_kdeg'].values)) / np.std(train_data['log_kdeg'].values))
test_targets = torch.tensor((test_data['log_kdeg'].values - statistics.mean(test_data['log_kdeg'].values)) / np.std(test_data['log_kdeg'].values))

# Normalize


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
    seq_len = 2500
).to(device)


# ### SANDBOX: Walk through each layer of model
# input_shape = 5
# hidden_units = 32

# ## First block
# test_block_1 = nn.Sequential(
#     nn.Conv1d(
#         in_channels = input_shape,
#         out_channels = hidden_units,
#         kernel_size= 3,
#         stride = 1,
#         padding = 1
#     ),
#     nn.ReLU(),
#     nn.Conv1d(
#         in_channels=hidden_units,
#         out_channels=hidden_units,
#         kernel_size=3,
#         stride = 1,
#         padding = 1
#     ),
#     nn.ReLU(),
#     nn.MaxPool1d(kernel_size=2,
#                     stride=2)
# ).to(device)

# seq.shape
# # 1 x 5 x 2200

# train_features_batch.shape

# output_1 = test_block_1(train_features_batch)
# output_1.shape
# # 1 x 3 x 1100


# ## 2nd block
# test_block_2 = nn.Sequential(
#             nn.Conv1d(
#                 in_channels = hidden_units,
#                 out_channels = hidden_units, 
#                 kernel_size = 3, 
#                 padding =1
#                 ),
#             nn.ReLU(),
#             nn.Conv1d(
#                 in_channels = hidden_units,
#                 out_channels = hidden_units, 
#                 kernel_size = 3, 
#                 padding =1
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2)
#         ).to(device)

# output_1.shape
# output_2 = test_block_2(output_1)
# output_2.shape
# # 1 x 2 x 551


# ## 3rd block
# test_block_3 = nn.Sequential(
#             nn.Flatten(1, 2),
#             nn.Linear(hidden_units * 550, 1)
#         ).to(device)

# test_flatten = nn.Flatten(1, 2)

# test_flatten(output_2).shape

# output_2.shape
# output_3 = test_block_3(output_2)
# output_3.shape

### Setup loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    params=simple_model.parameters(),
    lr = 0.01
)


epochs = 15

train_losses = [0]*epochs
test_losses = [0]*epochs
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
        test_losses[epoch] = test_loss.to('cpu').detach().numpy()



train_loss
test_loss

### Now plot estimate vs. truth
simple_model.eval()

predicted_kdeg = []
true_kdeg = []

predicted_kdeg_test = []
true_kdeg_test = []

final_test_loss = 0
with torch.inference_mode():
    for X, y in train_loader:
        
        y_pred = simple_model(X)

        predicted_kdeg.extend(y_pred.squeeze().cpu().detach().numpy())
        true_kdeg.extend(y.squeeze().cpu().detach().numpy())

    for X, y in test_loader:
        
        y_pred = simple_model(X)

        final_test_loss = final_test_loss + loss_fn(y_pred.squeeze(), y)

        predicted_kdeg_test.extend(y_pred.squeeze().cpu().detach().numpy())
        true_kdeg_test.extend(y.squeeze().cpu().detach().numpy())


plt.scatter(true_kdeg,
            predicted_kdeg,
            alpha = 0.5)
plt.xlabel('True values (train)')
plt.ylabel('Predicted values (train)')
plt.show()


plt.scatter(true_kdeg_test,
            predicted_kdeg_test,
            alpha = 0.5)
plt.xlabel('True values (test)')
plt.ylabel('Predicted values (test)')
plt.show()


plt.scatter(list(range(epochs)),
            train_losses)
plt.show()

plt.scatter(list(range(epochs)),
            test_losses)
plt.show()