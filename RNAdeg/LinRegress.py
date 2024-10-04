### PURPOSE OF THIS SCRIPT
## Perform multi-linear regression as a good
## baseline to compare

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


degdata = pd.read_csv('C:\\Users\\isaac\\Documents\\ML_pytorch\\Data\\RNAdeg\\RNAdeg_dataset.csv')

#degdata.columns
degdata = degdata[degdata[["3'UTR_length", "log_ksyn", "nmd",
             "stop_to_lastEJ", "num_of_downEJs"]].notnull().all(1)]

X = degdata[["3'UTR_length", "log_ksyn", "nmd",
             "stop_to_lastEJ", "num_of_downEJs"]]

X.loc[:, 'nmd'] = X['nmd'].map({'yes': 1.0, 'no': 0.0})

y = degdata['log_kdeg']

cols_to_standardize = ["log_3'UTR_length", "log_ksyn", "nmd",
             "log_stop_to_lastEJ", "num_of_downEJs"]
cols_to_log = ["3'UTR_length",
             "stop_to_lastEJ"]

for col in cols_to_log:
    X.loc[:,f'log_{col}'] = np.sign(X.loc[:,col]) * np.log(np.abs(X.loc[:,col]) + 1)


X.loc[:,cols_to_standardize] = (X.loc[:,cols_to_standardize] - X.loc[:,cols_to_standardize].mean())/X.loc[:,cols_to_standardize].std()


regr = linear_model.LinearRegression()
regr.fit(X, y)