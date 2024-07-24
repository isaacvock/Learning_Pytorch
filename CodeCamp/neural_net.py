import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np

### Make classification data: two different circles

# 1000 samples of data
n_samples = 1000

# Create circles (cute toy example)
X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state = 42)
    # x is data
    # y is label

# Import into data frame to aid vis
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
circles.head(10)

# Visualize
plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu);
plt.show()


# What are shapes of data
X.shape
    # 2D tensor of x and y coordinates
y.shape
    # 1D tensor of labels


### Turn data into tensors and train-test split

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


### Build a device agnostic model

# Make diagnostic agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"


# Construct model class
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # Create 2 layers able to handle input and output
        self.layer_1 = nn.Linear(in_features=2, out_features = 5) # accept X input
        self.layer_2 = nn.Linear(in_features = 5, out_features = 1) # y prediction

    def forward(self, x):

        # Layer 2 output is what we are interested in
        return self.layer_2(self.layer_1(x))
    
# Create an instance of our model
model_0 = CircleModelV0().to(device)

# Make predictions with the model
untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")


### Specify loss function and evaluation crtierion

# Create a loss function
# loss_fn = nn.BCELoss() # BCELoss = no sigmoid built-in
loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


### Train

# Check current pass through forward method
y_preds = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train();

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze();
    y_pred = torch.round(torch.sigmoid(y_logits));

    # 2. Calculate loss
    loss = loss_fn(y_logits,
                   y_train);
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred);
    
    # 3. Optimizer zero grad, totally unsure why this is necessary
    optimizer.zero_grad();

    # 4. Backwards propagation
    loss.backward();

    # 5. Update
    optimizer.step();

    ### Testing
    model_0.eval();
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze();
        test_pred = torch.round(torch.sigmoid(test_logits));
        
        # 2. Calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test);

        test_acc = accuracy_fn(y_true = y_test,
                               y_pred = test_pred);

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")


# CONCLUSION: MODEL UNDERFITS BY USING A SIMPLE LINE AS THE DECISION
# BOUNDARY.


### Improving the model

# More hidden units
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10) # extra layer
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x): # note: always make sure forward is spelt correctly!
        # Creating a model like this is the same as below, though below
        # generally benefits from speedups where possible.
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = CircleModelV1().to(device)
model_1


loss_fn = nn.BCEWithLogitsLoss() # Does not require sigmoid on input
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)


# Train for longer

epochs = 1000 # Train for longer

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> predicition probabilities -> prediction labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")



### Model with non-linearity

# Build model with non-linear activation function
from torch import nn
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=20)
        self.layer_2 = nn.Linear(in_features=20, out_features=20)
        self.layer_3 = nn.Linear(in_features=20, out_features=1)

        self.relu = nn.ReLU() # <- add in ReLU activation function
        
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model_3 = CircleModelV2().to(device)
print(model_3)


# Setup loss and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

results = np.zeros((epochs, 4))


for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> prediction probabilities -> prediction labels
    
    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train) # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
      # 1. Forward pass
      test_logits = model_3(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
      # 2. Calcuate loss and accuracy
      test_loss = loss_fn(test_logits, y_test)
      test_acc = accuracy_fn(y_true=y_test,
                             y_pred=test_pred)

    # Print out what's happening
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")


    results[epoch, 0] = loss
    results[epoch, 1] = acc
    results[epoch, 2] = test_loss
    results[epoch, 3] = test_acc



# plot loss over "time"
plt.scatter(x = np.linspace(1, epochs, num = epochs),
            y = results[:,0]);
plt.show()

# plot accuracy over "time"
plt.scatter(x = np.linspace(1, epochs, num = epochs),
            y = results[:,1]);
plt.show()


# plot test loss over "time"
plt.scatter(x = np.linspace(1, epochs, num = epochs),
            y = results[:,2]);
plt.show()

# plot test accuracy over "time"
plt.scatter(x = np.linspace(1, epochs, num = epochs),
            y = results[:,3]);
plt.show()


## Plot decision boundary
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
plt.show()