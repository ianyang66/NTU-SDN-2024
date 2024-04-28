'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-25 23:02:00
LastEditors: Ian Yang
LastEditTime: 2024-04-25 22:29:29
'''

import torch
import torch.nn as nn

# Define the neural network
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 96)
        self.fc5 = nn.Linear(96, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x