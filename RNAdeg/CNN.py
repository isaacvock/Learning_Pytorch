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

### Load in data

features = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\RNAdeg_feature_table.csv")
features_filter = features.loc[features['avg_lkd_se'] < math.exp(-2)]

promoters = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\promoter_seqs.csv")

full_data = pd.merge(features_filter, promoters, on = ['transcript_id'], how = 'inner')

train_data = full_data[~full_data['seqnames'].isin(['chr1', 'chr22'])]

char_to_index = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
}


### One-hot encode features
def onehote_np(seq):
    seq2 = [char_to_index[i] for i in seq]
    return torch.from_numpy(np.eye(5)[seq2])

train_tensor = [onehote_np(seq) for seq in train_data['seq']]
train_tensor_3d = torch.stack(train_tensor)

train_tensor_3d.shape


### Convert log(kdeg) to Pytorch tensor

train_targets = torch.tensor(train_data['log_kdeg'].values)

### Create DataLoader

train = data_utils.TensorDataset(train_tensor_3d, train_targets)
train_loader = data_utils.DataLoader(train, batch_size = 32, shuffle=True)


### Build model
