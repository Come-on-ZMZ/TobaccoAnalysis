import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from matplotlib import rcParams

# 设置全局字体为Times New Roman, 12号
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'  # 普通字体
#rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体字体
#rcParams['mathtext.bf'] = 'Times New Roman:bold'  # 粗体字体
title_font = {'size': 12, 'color': 'k', 'family': 'Times New Roman'}  # 标题字体

# 随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# 加载数据
file_path = "Data/stdTSR.xlsx"
data_22 = pd.read_excel(file_path)
data_23 = pd.read_excel("Data/std23.xlsx")

file_name = os.path.basename(file_path)  # 获取文件名stdLSTM.xlsx
model_name = file_name.split('.')[0][3:]  # 提取std后面的部分LSTM

data = pd.concat([data_22, data_23], ignore_index=True)
print(data.shape)

# 提取特征和标签
features = data.iloc[:, 3:].values
labels = data.iloc[:, 2].values  # 第2列氮、第3列烟碱

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=45)
N_train, N_test = len(X_train), len(X_test)  # 提取样本数

# 特征重塑(样本数, 时间步, 特征数)
time_step = 2  # 时间步
input_num = 7  # 特征数
X_train_new = X_train.reshape(-1, time_step, input_num)
X_test_new = X_test.reshape(-1, time_step, input_num)

# 标签标准化
label_std = StandardScaler()
y_train_std = label_std.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_std = label_std.transform(y_test.reshape(-1, 1)).flatten()

# 数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train_new, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_std, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_new, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_std, dtype=torch.float32).view(-1, 1)

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # GRU前向传播
        out, hn = self.gru(x, h0)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 超参数设置
hidden_size = 32  # 隐藏层大小
num_layers = 2    # GRU层数
output_size = 1   # 输出(1维)
learning_rate = 0.001  # 学习率
num_epochs = 40   # 训练轮数
batch_size = 12   # 批处理大小

# 初始化GRU模型、损失函数和优化器
model = GRUModel(input_num, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'第 {epoch + 1}/{num_epochs} 轮, 损失值: {avg_loss:.3f}')

# 定义评估函数
def evaluate_model(model, X_tensor, y_tensor, label_std):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor).numpy()
        targets = y_tensor.numpy()

    # 反标准化
    outputs_unstd = label_std.inverse_transform(outputs)
    targets_unstd = label_std.inverse_transform(targets)

    # 计算评估指标
    mse = mean_squared_error(targets_unstd, outputs_unstd)
    mape = mean_absolute_percentage_error(targets_unstd, outputs_unstd) * 100
    mae = mean_absolute_error(targets_unstd, outputs_unstd)
    r2 = r2_score(targets_unstd, outputs_unstd)
    rmse = np.sqrt(mse)

    return {'mse': mse, 'mape': mape, 'mae': mae, 'r2': r2, 'rmse': rmse,
            'targets': targets_unstd.flatten(), 'predictions': outputs_unstd.flatten()}

# 评估模型
train_metrics = evaluate_model(model, X_train_tensor, y_train_tensor, label_std)
test_metrics = evaluate_model(model, X_test_tensor, y_test_tensor, label_std)

# 打印结果
def print_metrics(metrics, dataset_name):
    print(f'\n{dataset_name}集评估指标:')
    print(f'均方误差MSE: {metrics["mse"]:.3f}')
    print(f'平均绝对百分比误差MAPE: {metrics["mape"]:.3f}%')
    print(f'平均绝对误差MAE: {metrics["mae"]:.3f}%')
    print(f'相关系数R2: {metrics["r2"]:.3f}')
    print(f'均方根误差RMSE: {metrics["rmse"]:.3f}')

print_metrics(train_metrics, '训练')
print_metrics(test_metrics, '测试')

# 绘图
def plot(train_metrics, test_metrics, N_train, N_test):
    x1 = np.linspace(10, 50, 100)
    y1 = x1

    # # 定义线性拟合函数
    # def f_linear(x, A, B):
    #     return A * x + B
    #
    # # 训练集拟合直线
    # A1, B1 = optimize.curve_fit(f_linear, train_metrics['targets'], train_metrics['predictions'])[0]
    # y2 = f_linear(train_metrics['targets'], A1, B1)
    #
    # # 测试集拟合直线
    # C1, D1 = optimize.curve_fit(f_linear, test_metrics['targets'], test_metrics['predictions'])[0]
    # y3 = f_linear(test_metrics['targets'], C1, D1)

    # 设置图像
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)
    ax.set_title(r'TSR-GRU for cNt', title_font, pad=10)
    #ax.set_title(r'AE$_{\text{'+ model_name + r'}}$-GRU for cNt', title_font, pad=10)
    ax.plot(x1, y1, color='k', linewidth=1.2, linestyle='--')

    # 绘制散点图
    color1 = np.array([22, 118, 177]) / 255  # 系统蓝
    color2 = np.array([240, 135, 68]) / 255  # 淡橘色
    ax.scatter(train_metrics['targets'], train_metrics['predictions'], s=12, marker='^', label='Training')
    ax.scatter(test_metrics['targets'], test_metrics['predictions'], color='red', s=12, marker='x', label='Validation')
    # ax.plot(y_train, y2, color=color1, linewidth=1.2, linestyle='-')  # 拟合直线
    # ax.plot(y_test, y3, color=color2, linewidth=1.2, linestyle='-')
    ax.grid()
    ax.legend(loc='lower left', prop={'size': 12}, handletextpad=0.2, borderaxespad=0.4, framealpha=0.8)

    # 设置坐标轴
    ax.set_xlim((10, 50))
    ax.set_ylim((10, 50))
    ax.set_xticks(np.arange(10, 50.1, step=8))
    ax.set_yticks(np.arange(10, 50.1, step=8))
    ax.set_xlabel("Measured cNt (mg/g)")
    ax.set_ylabel("Predicted cNt (mg/g)")

    # 训练集
    ax.text(15.6, 47.0, r'$R^2_t$={:.2f}'.format(train_metrics['r2']))
    ax.text(10.8, 47.0, r'$N_t$=' + str(N_train))
    ax.text(10.8, 44.5, r'$RMSE_t$={:.2f}mg/g'.format(train_metrics['rmse']))
    ax.text(10.8, 42.0, r'$MAE_t$={:.2f}mg/g'.format(train_metrics['mae']))
    ax.text(10.8, 39.5, r'$MAPE_t$={:.2f}%'.format(train_metrics['mape']))

    # 测试集
    ax.text(43.0, 19.0, r'$R^2_v$={:.2f}'.format(test_metrics['r2']))
    ax.text(38.8, 19.0, r'$N_v$=' + str(N_test))
    ax.text(38.8, 16.5, r'$RMSE_v$={:.2f}mg/g'.format(test_metrics['rmse']))
    ax.text(38.8, 14.0, r'$MAE_v$={:.2f}mg/g'.format(test_metrics['mae']))
    ax.text(38.8, 11.5, r'$MAPE_v$={:.2f}%'.format(test_metrics['mape']))

    plt.tight_layout()
    plt.savefig(f'Figure/{model_name}-GNt.pdf', format='pdf', bbox_inches='tight')
    plt.show()

plot(train_metrics, test_metrics, N_train, N_test)
print(f'图片已保存到 Figure/{model_name}-GNt.pdf')

