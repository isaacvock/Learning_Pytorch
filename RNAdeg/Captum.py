### PURPOSE OF THIS SCRIPT
## Practice using Captum to interpret ML models



##### Getting started - Titantic Data #####

import numpy as np
import torch

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
import pandas as pd

dataset_path = "C:/Users/isaac/Documents/ML_pytorch/Data/Captum/titanic3.csv"
titanic_data = pd.read_csv(dataset_path)

titanic_data = pd.concat([titanic_data,
                          pd.get_dummies(titanic_data['sex']),
                          pd.get_dummies(titanic_data['embarked'], prefix = "embark"),
                          pd.get_dummies(titanic_data['pclass'], prefix="class")], axis = 1)

titanic_data["age"] = titanic_data["age"].fillna(titanic_data["age"].mean())
titanic_data["fare"] = titanic_data["fare"].fillna(titanic_data["fare"].mean())
titanic_data = titanic_data.drop(['name','ticket','cabin','boat','body','home.dest','sex','embarked','pclass'], axis=1)

# Convert to NumPy arrays
np.random.seed(131254)
labels = titanic_data["survived"].to_numpy()
titanic_data = titanic_data.drop(['survived'], axis = 1)
feature_names = list(titanic_data.columns)
data = titanic_data.to_numpy()

# Train-test split
train_indices = np.random.choice(len(labels), int(0.7*len(labels)), replace = False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = np.array(data[train_indices], dtype = float)
train_labels = labels[train_indices]
test_features = np.array(data[test_indices], dtype = float)
test_labels = labels[test_indices]

# Model
import torch
import torch.nn as nn
torch.manual_seed(1)

class TitanicSimpleNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(12, 12)
        self.sig1 = nn.Sigmoid()
        
        self.lin2 = nn.Linear(12, 8)
        self.sig2 = nn.Sigmoid()

        self.lin3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        return self.softmax(self.lin3(self.sig2(self.lin2(self.sig1(self.lin1(input))))))


simple_model = TitanicSimpleNNModel()

simple_model(torch.from_numpy(test_features).float())

### Train that shit!

criterion = nn.CrossEntropyLoss()
num_epochs = 200
optimizer = torch.optim.Adam(
    params=simple_model.parameters(),
    betas=(0.9, 0.98)
)


train_losses = [0]*num_epochs
test_losses = [0]*num_epochs


# Convert to Pytorch tensors
input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
label_tensor = torch.from_numpy(train_labels)

test_input_tensor = torch.from_numpy(test_features).float()
test_label_tensor = torch.from_numpy(test_labels)

simple_model().train()
for i in range(num_epochs):

    optimizer.zero_grad()
    yest = simple_model(input_tensor)
    loss = criterion(yest, label_tensor)
    train_losses[i] = loss.to('cpu').detach().numpy()

    loss.backward()

    optimizer.step()

    simple_model.eval()

    yest_test = simple_model(test_input_tensor)
    test_loss = criterion(yest_test, test_label_tensor)
    test_losses[i] = test_loss.to('cpu').detach().numpy()



### Aside: assess model ###

import matplotlib.pyplot as plt

plt.scatter(
    np.array(range(num_epochs)),
    train_losses
)
plt.xlabel("Epoch")
plt.ylabel("Train loss")
plt.show()
    

plt.scatter(
    np.array(range(num_epochs)),
    test_losses
)
plt.xlabel("Epoch")
plt.ylabel("Test loss")
plt.show()


### 


##### CAPTUM BASIC TEST #####

import numpy as np
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(3, 2)

        # Initialize weights and biases
        self.lin1.weight = nn.Parameter(torch.arange(-4.0, 5.0).view(3, 3))
        self.lin1.bias = nn.Parameter(torch.zeros(1, 3))
        self.lin2.weight = nn.Parameter(torch.arange(-3.0, 3.0).view(2, 3))
        self.lin2.bias =  nn.Parameter(torch.ones(1, 2))

    def forward(self, input):
        return self.lin2(self.relu(self.lin1(input)))
    
model = ToyModel()
model.eval()

testlin = nn.Linear(3, 2)
testlin.weight.dim

torch.manual_seed(123)
np.random.seed(123)

input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)

ig = IntegratedGradients(model)
attributions, delta = ig.attribute(
    input,
    baseline,
    target = 0,
    return_convergence_delta=True
)

print('IG Attributions', attributions)
print('Convergence Delta:', delta)