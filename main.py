import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import copy

# 读取数据
col_read = ['class', 'posx', 'posy', 'spdx', 'spdy']
df = pd.read_csv('mixalldata_clean.csv', usecols=col_read)
merged_df = df.head(10000)


# 数据预处理
def set_label(class_value):
    if class_value == 0:
        return 0
    elif 1 <= class_value <= 9:
        return 1
    else:
        return 2


merged_df['class'] = merged_df['class'].apply(set_label)
columns_to_scale = merged_df.columns.drop('class')
merged_df[columns_to_scale] = StandardScaler().fit_transform(merged_df[columns_to_scale])
df_new = pd.concat([
    merged_df.loc[merged_df['class'] == 0].sample(frac=0.3),
    merged_df.loc[merged_df['class'] == 1],
    merged_df.loc[merged_df['class'] == 2]
], ignore_index=True)

le = LabelEncoder()
df_new['class'] = le.fit_transform(df_new['class'])


# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['posx', 'posy', 'spdx', 'spdy']].values
        self.targets = dataframe['class'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return data, target


# 模型定义
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))  # Add batch_first dimension
        out = self.fc(out[:, -1, :])  # Output at the last timestep
        return out


# 联邦学习函数
def federated_averaging(models):
    global_model = copy.deepcopy(models[0])
    with torch.no_grad():
        for param in global_model.parameters():
            param.data.zero_()
        for model in models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                global_param.data += local_param.data / len(models)
    return global_model


# 数据分割与客户端定义
num_clients = 5
client_data = np.array_split(df_new, num_clients)
clients = [CustomDataset(client_df) for client_df in client_data]

# 全局参数
input_size = 4
hidden_size = 256
num_layers = 3
num_classes = 3
learning_rate = 0.001
num_epochs = 10
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化客户端模型和优化器
client_models = [LSTM(input_size, hidden_size, num_layers, num_classes).to(device) for _ in range(num_clients)]
client_optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in client_models]

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 联邦训练
global_model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

for round in range(5):  # 联邦学习的轮数
    print(f"Round {round + 1}")

    # 每个客户端本地训练
    for client_idx in range(num_clients):
        client_model = client_models[client_idx]
        optimizer = client_optimizers[client_idx]
        client_dataset = clients[client_idx]
        train_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

        client_model.train()
        for epoch in range(num_epochs):
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = client_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f"Client {client_idx + 1}: Loss = {loss.item():.4f}")

    # 聚合更新全局模型
    global_model = federated_averaging(client_models)

    # 将全局模型更新到每个客户端
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

# 测试全局模型
test_dataset = CustomDataset(df_new)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

global_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = global_model(data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Global Model Accuracy: {100 * correct / total:.2f}%")
