### PURPOSE OF THIS SCRIPT
# Make sure I can one-hot encode sequence information for eventual work
# developing sequence-based ML models

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

promoters = pd.read_csv("C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\promoter_seqs.csv")

testseq = promoters.loc[1, 'seq']
testseq

seq_df = pd.DataFrame(list(testseq), columns = ['nt'])

seq_ohe = pd.get_dummies(seq_df, columns = ['nt'])

len(testseq)

char_to_index = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4,
}


### The derpy way
def one_hot_encode(seq):

    one_hot = torch.zeros(len(seq), len(char_to_index))

    for i, char in enumerate(seq):
        one_hot[i, char_to_index[char]] = 1

    return one_hot


one_hot_tensor = [one_hot_encode(seq) for seq in promoters['seq']]
one_hot_tensor_3d = torch.stack(one_hot_tensor)

### The numpy way
def onehote_np(seq):
    seq2 = [char_to_index[i] for i in seq]
    return torch.from_numpy(np.eye(5)[seq2])

one_hot_tensor_np = [onehote_np(seq) for seq in promoters['seq']]
one_hot_tensor_np_3d = torch.stack(one_hot_tensor_np)