'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-25 15:01:22
LastEditors: Ian Yang
LastEditTime: 2024-04-28 20:25:16
'''
#  export PYTHONPATH="${PYTHONPATH}:/mnt/d/NTU/112-2/ddos-sdn/"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from app.model.models import GRUModel, DNN

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Read csv file
df = pd.read_csv('dataset.csv')

# Read columns and build matrix
X = df[["speed_src_ip", "std_n_packets", "std_bytes", "bytes_per_flow", "n_int_flows"]].values
y = df["class"].values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float().unsqueeze(1).to(device)  # Add a sequence dimension and move to GPU
X_test = torch.from_numpy(X_test).float().unsqueeze(1).to(device)
y_train = torch.from_numpy(y_train).long().to(device)
y_test = torch.from_numpy(y_test).long().to(device)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
input_size = 5  # Input size for GRU is 1 (sequence of features)
hidden_size = 64  # You can adjust this value
num_classes = len(np.unique(y))
# model = GRUModel(input_size, hidden_size, num_classes).to(device)  # Move the model to GPU
model = DNN(input_size, hidden_size, num_classes).to(device)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")

# Save the model
torch.save(model.state_dict(), './model.pth')
