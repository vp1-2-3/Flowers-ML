import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
# print(my_df)

# Change last column from string to int
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)
# print(my_df)

# Create model class that inherits nn.Module 

class Model(nn.Module):
    # Import layer (4 features of flower)
    # -> HIDDEN1
    # -> HIDDEN2
    # -> OUPTUT
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() # Instantiate the module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.out(x))
        
        return x

# Random seed for randomization
torch.manual_seed(41)

model = Model()

# Train Test Split Set x,y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convery to np arrays
X = X.values
y = y.values
# print(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set criterion of model to measure error, how far off predictions are from actual
criterion = nn.CrossEntropyLoss()

# Choose Optimizer, lr - learning rate (if error doesn;t go down after epochs, lower lr)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model.parameters)

# Train model
# Epochs? (one run through all data in network)
epochs = 100
losses = []
for i in range(epochs):
    # Go forward and get prediction
    y_pred = model.forward(X_train) # Get predicted results
    
    # Measure loss/error, high at first
    loss = criterion(y_pred, y_train) # Predicted versus y_train
    
    # Keep track of losses
    losses.append(loss.detach().numpy())
    
    # Print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i} and loss: {loss}')
        
        # Back Propogation: take error rate of forward propogation
        # and feed through the network to fine tune the weights
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# Graph out
# plt.plot(range(epochs), losses) 
# plt.ylabel("Loss/Error")
# plt.xlabel('Epochs')
# plt.show()

# Evaluate model on test data set (Validation)
with torch.no_grad(): # Turn off back prop
    y_eval = model.forward(X_test) # X_test features from test set, y_eval will be predictions
    loss = criterion(y_eval, y_test) # Find loss or error
    # print(loss)
    
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        
        # if y_test[i] == 0:
        #     x = 'setosa'
        # elif y_test[i] == 1:
        #     x = 'versicolor'
        # else: 
        #     x = 'virginica'
        
        # Tell what type of flower thinks it is
        print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')
        
        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'We got {correct} correct!')