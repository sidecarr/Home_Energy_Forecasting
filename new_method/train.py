import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_layers, dropout=0.1, output_len=90):
        super().__init__()

        self.pre_rnn = nn.GRU(input_dim, input_dim, batch_first=True, bidirectional=False)

        # CNN: 增强局部特征提取能力（1D卷积）
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=7, padding=1)

        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.post_rnn = nn.GRU(model_dim, model_dim, batch_first=True)

        self.decoder = nn.Linear(model_dim, output_len)

    def forward(self, x):  # x: [B, 90, input_dim]
        x, _ = self.pre_rnn(x)  # [B, 90, input_dim]

        # CNN expects [B, C, T], so we permute
        x = x.permute(0, 2, 1)              # → [B, input_dim, 90]
        x = self.cnn(x)                     # → [B, model_dim, 90]
        x = x.permute(0, 2, 1)              # → [B, 90, model_dim]

        x = self.pos_encoder(x)            # + Positional Encoding
        x = self.transformer(x)            # Transformer Encoder

        x, _ = self.post_rnn(x)            # 后置 GRU

        x = x[:, -1, :]                    # 取最后时间步
        return self.decoder(x)             # [B, output_len]




from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(X, y, output_len=365, epochs=50, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转换为 Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型定义
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        model_dim=64,
        nhead=4,
        num_layers=2,
        output_len=output_len
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.6f}")
    return model

# 加载数据
df = pd.read_csv("/data/wlchen/htl/machine_learning/data/train.csv", parse_dates=["DateTime"])
df['Date'] = df['DateTime'].dt.date

# 转换为数值，逐列避免内存溢出
numeric_columns = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df["sub_metering_remainder"] = (
    (df["Global_active_power"] * 1000 / 60)
    - (df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"])
)
# 聚合：按天汇总
daily_df = df.groupby('Date').agg({
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

# 填补缺失
daily_df.fillna(method='ffill', inplace=True)
daily_df.fillna(method='bfill', inplace=True)

# 选择输入和目标
input_features = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "sub_metering_remainder",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
]
target_feature = "Global_active_power"

# 归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(daily_df[input_features])
y_scaled = scaler_y.fit_transform(daily_df[[target_feature]])

# 构造滑动窗口序列
def generate_sequences(X, y, input_len, output_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_len - output_len + 1):
        X_seq.append(X[i:i+input_len])
        y_seq.append(y[i+input_len:i+input_len+output_len].squeeze())
    return np.array(X_seq), np.array(y_seq)

# 短期预测 (90 -> 90)，长期预测 (90 -> 365)
X_short, y_short = generate_sequences(X_scaled, y_scaled, 90, 90)
X_long, y_long = generate_sequences(X_scaled, y_scaled, 90, 365)

model = train_model(X_long, y_long, output_len=365)

def evaluate_model(model, X, y_true, scaler_y, output_len):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        y_pred = model(X_tensor).cpu().numpy()

    # 反归一化
    y_true_rescaled = scaler_y.inverse_transform(y_true)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)

    # 画出前一组样本预测效果
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred_rescaled[0], label="Prediction")
    plt.plot(y_true_rescaled[0], label="Ground Truth")
    
    plt.title('365-day Power Prediction vs Ground Truth')
    plt.xlabel('Day')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_truth.png')  # 保存为图片
    plt.show()

    # 返回指标
    mse = mean_squared_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten())
    mae = mean_absolute_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten())
    return mse, mae

def load_and_prepare_test(test_csv_path, scaler_X, scaler_y, input_len=90, output_len=365):
    manual_columns = [
    "DateTime",
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]

    df_test = pd.read_csv(test_csv_path, header=None, names=manual_columns, parse_dates=["DateTime"],low_memory=False)
    df_test['Date'] = df_test['DateTime'].dt.date

    # 转换为数值，跟训练一致
    numeric_columns = [
        "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
    for col in numeric_columns:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    df_test["sub_metering_remainder"] = (
        (df_test["Global_active_power"] * 1000 / 60)
        - (df_test["Sub_metering_1"] + df_test["Sub_metering_2"] + df_test["Sub_metering_3"])
    )

    # 按天聚合
    daily_test = df_test.groupby('Date').agg({
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


    input_features = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "sub_metering_remainder",
    "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"
    ]
    target_feature = "Global_active_power"
    # 归一化（用训练时的scaler）
    X_test_scaled = scaler_X.transform(daily_test[input_features])
    y_test_scaled = scaler_y.transform(daily_test[[target_feature]])

    # 构造序列
    X_test_seq, y_test_seq = generate_sequences(X_test_scaled, y_test_scaled, input_len, output_len)
    return X_test_seq, y_test_seq, daily_test


model.eval()
test_csv_path = "/data/wlchen/htl/machine_learning/data/test.csv"
X_test_seq, y_test_seq, daily_test = load_and_prepare_test(test_csv_path, scaler_X, scaler_y)
mse, mae = evaluate_model(model, X_test_seq, y_test_seq, scaler_y, output_len=365)
print(f"Direct 365-day Prediction - MSE: {mse:.4f}, MAE: {mae:.4f}")