'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-21 23:41:23
LastEditors: Ian Yang
LastEditTime: 2024-04-25 22:15:35
'''

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

print(torch.cuda.is_available())
# Read and preprocess the data
csv_files = os.listdir('../InSDN_DatasetCSV')
df = pd.DataFrame()

li = []

for filename in csv_files:
    df = pd.read_csv(os.path.join('../InSDN_DatasetCSV', filename), low_memory=False, index_col=None, header=0)
    li.append(df)
    print("Read in {}".format(filename))

df = pd.concat(li, axis=0, ignore_index=True)

print("Finisehd reading in {} entires".format(str(df.shape[0])))


metadata = ['Flow ID',
'Src IP',
'Src Port',
'Dst IP',
'Dst Port',
'Protocol',
'Timestamp',
'Flow Duration',
'Tot Fwd Pkts',
'Tot Bwd Pkts',
'TotLen Fwd Pkts',
'TotLen Bwd Pkts',
'Fwd Pkt Len Max',
'Fwd Pkt Len Min',
'Fwd Pkt Len Mean',
'Fwd Pkt Len Std',
'Bwd Pkt Len Max',
'Bwd Pkt Len Min',
'Bwd Pkt Len Mean',
'Bwd Pkt Len Std',
'Flow Byts/s',
'Flow Pkts/s',
'Flow IAT Mean',
'Flow IAT Std',
'Flow IAT Max',
'Flow IAT Min',
'Fwd IAT Tot',
'Fwd IAT Mean',
'Fwd IAT Std',
'Fwd IAT Max',
'Fwd IAT Min',
'Bwd IAT Tot',
'Bwd IAT Mean',
'Bwd IAT Std',
'Bwd IAT Max',
'Bwd IAT Min',
'Fwd PSH Flags',
'Bwd PSH Flags',
'Fwd URG Flags',
'Bwd URG Flags',
'Fwd Header Len',
'Bwd Header Len',
'Fwd Pkts/s',
'Bwd Pkts/s',
'Pkt Len Min',
'Pkt Len Max',
'Pkt Len Mean',
'Pkt Len Std',
'Pkt Len Var',
'FIN Flag Cnt',
'SYN Flag Cnt',
'RST Flag Cnt',
'PSH Flag Cnt',
'ACK Flag Cnt',
'URG Flag Cnt',
'CWE Flag Count',
'ECE Flag Cnt',
'Down/Up Ratio',
'Pkt Size Avg',
'Fwd Seg Size Avg',
'Bwd Seg Size Avg',
'Fwd Byts/b Avg',
'Fwd Pkts/b Avg',
'Fwd Blk Rate Avg',
'Bwd Byts/b Avg',
'Bwd Pkts/b Avg',
'Bwd Blk Rate Avg',
'Subflow Fwd Pkts',
'Subflow Fwd Byts',
'Subflow Bwd Pkts',
'Subflow Bwd Byts',
'Init Fwd Win Byts',
'Init Bwd Win Byts',
'Fwd Act Data Pkts',
'Fwd Seg Size Min',
'Active Mean',
'Active Std',
'Active Max',
'Active Min',
'Idle Mean',
'Idle Std',
'Idle Max',
'Idle Min',
'Label'
]


df.columns = metadata


df["Label"].value_counts()


from scipy.stats import zscore

def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))
        
def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

import matplotlib.pyplot as plt
from matplotlib.pyplot import *

plt.figure(figsize=(20,20))
fig, ax = plt.subplots(figsize=(20,20))

class_distribution = df['Label'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of InSDN Training Data Before Cleaning')
# plt.grid()
# plt.show()
fig.savefig('InSDN_Data_Distribution.pdf') 


# Before Cleaning Data set for Duplicate
sorted_ds = np.argsort(-class_distribution.values)
for i in sorted_ds:
    print('Number of data points in class', class_distribution.index[i],':', class_distribution.values[i], 
          '(', np.round((class_distribution.values[i]/df.shape[0]*100), 3), '%)')


#drop na values and reset index
data_clean = df.dropna().reset_index()

# Checkng for DUPLICATE values
data_clean.drop_duplicates(keep='first', inplace = True)

data_clean['Label'].value_counts()

print("Read {} rows.".format(len(data_clean)))


# Remove columns with only values of 0
useless_columns = ['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port']
df.drop(labels=useless_columns, axis='columns', inplace=True)
print('After dropping some columns: \n\t there are {} columns and {} rows'.format(len(df.columns), len(df)))

#features = df.columns


analyze(df)


plt.figure(figsize=(15,7))
class_distribution = data_clean['Label'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of Cleaned CICIDS2017 Training Data')
# plt.grid()
# plt.show()


# After Cleaning Data set for Duplicate
sorted_ds = np.argsort(-class_distribution.values)
for i in sorted_ds:
    print('Number of data points in class', class_distribution.index[i],':', class_distribution.values[i], 
          '(', np.round((class_distribution.values[i]/df.shape[0]*100), 3), '%)')


# Convert to numpy - Classification
x_columns = df.columns.drop('Label')
x = df[x_columns].values
dummies = pd.get_dummies(df['Label']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values


#drop na values and reset index
data_clean = df.dropna().reset_index()


# Label encoding
label_encoder = LabelEncoder()
data_clean['Label'] = label_encoder.fit_transform(data_clean['Label'])

data_np = data_clean.to_numpy(dtype="float32")
data_np = data_np[~np.isinf(data_np).any(axis=1)]

X = data_np[:, 0:77]
Y = data_np[:, 78]

# One-hot encoding for labels
Y = torch.tensor(Y, dtype=torch.long)

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating PyTorch tensors and datasets
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor, Y)

# Splitting the dataset into train and test
train_ratio = 0.75
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Creating data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# Initialize the model, loss function, and optimizer
input_size = X_tensor.shape[1]
output_size = len(label_encoder.classes_)
print("input_size",input_size)
print("output_size",output_size)
model = DNN(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")

# Evaluation on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")

# Save the model
torch.save(model.state_dict(), './model_cnn.h5')