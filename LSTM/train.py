import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

manual_columns = [
    "DateTime",
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
# 加载数据
train_df = pd.read_csv('/data/wlchen/htl/machine_learning/data/train.csv', parse_dates=['DateTime'])
train_df['Date'] = train_df['DateTime'].dt.date
test_df = pd.read_csv('/data/wlchen/htl/machine_learning/data/test.csv', header=None, names=manual_columns, parse_dates=["DateTime"],low_memory=False)
test_df['Date'] = test_df['DateTime'].dt.date

numeric_columns = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]

for col in numeric_columns:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

for col in numeric_columns:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# 可选：计算 sub_metering_remainder
for df in [train_df, test_df]:
    df["sub_metering_remainder"] = (df["Global_active_power"] * 1000 / 60) - (
        df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
    )

# 按天聚合
daily_train = train_df.groupby('Date').agg({
    "Global_active_power": 'sum',
    "Global_reactive_power": 'sum',
    "Voltage": 'mean',
    "Global_intensity": 'mean',
    "Sub_metering_1": 'sum',
    "Sub_metering_2": 'sum',
    "Sub_metering_3": 'sum',
    "sub_metering_remainder": 'sum',
    "RR": 'first',
    "NBJRR1": 'first',
    "NBJRR5": 'first',
    "NBJRR10": 'first',
    "NBJBROU": 'first'
}).reset_index()

    # 填缺失
daily_train.fillna(method='ffill', inplace=True)
daily_train.fillna(method='bfill', inplace=True)

daily_test = test_df.groupby('Date').agg({
    "Global_active_power": 'sum',
    "Global_reactive_power": 'sum',
    "Voltage": 'mean',
    "Global_intensity": 'mean',
    "Sub_metering_1": 'sum',
    "Sub_metering_2": 'sum',
    "Sub_metering_3": 'sum',
    "sub_metering_remainder": 'sum',
    "RR": 'first',
    "NBJRR1": 'first',
    "NBJRR5": 'first',
    "NBJRR10": 'first',
    "NBJBROU": 'first'
}).reset_index()

    # 填缺失
daily_test.fillna(method='ffill', inplace=True)
daily_test.fillna(method='bfill', inplace=True)

# 特征列和目标列
features = [
    'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]
target = 'Global_active_power'

# 归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(daily_train[features])
y_train = scaler_y.fit_transform(daily_train[[target]])

X_test = scaler_X.transform(daily_test[features])
y_test = scaler_y.transform(daily_test[[target]])

class MultiStepSequenceDataset(Dataset):
    def __init__(self, X, y, input_len=90, pred_len=365):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.X) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.input_len]                   # 过去90天输入
        y_seq = self.y[idx + self.input_len : idx + self.input_len + self.pred_len]  # 未来90天目标
        return X_seq, y_seq.squeeze()  # shape: (90, features), (90,)
train_dataset = MultiStepSequenceDataset(X_train, y_train, input_len=90, pred_len=365)
test_dataset = MultiStepSequenceDataset(X_test, y_test, input_len=90, pred_len=365)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

import torch.nn as nn

class LSTM_MultiStep(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_len=365):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, 30, H)
        out = out[:, -1, :]    # (B, H)
        out = self.fc(out)     # (B, 90)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM_MultiStep(input_size=X_train.shape[1], output_len=365).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        output = model(X_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}")

model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch).cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        predictions.append(y_pred)
        ground_truth.append(y_batch)

# 合并所有批次（注意 axis=0）
predictions = np.concatenate(predictions, axis=0)  # shape (N, 90)
ground_truth = np.concatenate(ground_truth, axis=0)  # shape (N, 90)

# 反归一化
predictions = scaler_y.inverse_transform(predictions)
ground_truth = scaler_y.inverse_transform(ground_truth)
# 计算平均 MSE / MAE
plt.figure(figsize=(12, 6))
plt.plot(predictions[0], label='Predicted')
plt.plot(ground_truth[0], label='Ground Truth')
plt.title('365-day Power Prediction vs Ground Truth')
plt.xlabel('Day')
plt.ylabel('Global Active Power')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_vs_truth.png')  # 保存为图片
plt.show()
mse = np.mean([
    mean_squared_error(np.squeeze(gt), np.squeeze(pred))
    for gt, pred in zip(ground_truth, predictions)
])

mae = np.mean([
    mean_absolute_error(np.squeeze(gt), np.squeeze(pred))
    for gt, pred in zip(ground_truth, predictions)
])

print(f"Direct 365-day Prediction - MSE: {mse:.4f}, MAE: {mae:.4f}")