'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-24 19:19:38
LastEditors: Ian Yang
LastEditTime: 2024-04-28 20:27:53
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Get the last output for classification
        out = self.fc(out)
        return out



class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNN, self).__init__()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 96)
        self.fc5 = nn.Linear(96, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        out = x.squeeze(1)
        return out
