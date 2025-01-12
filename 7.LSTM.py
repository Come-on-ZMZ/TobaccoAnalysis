import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score
from matplotlib import rcParams

# 设置全局字体为Times New Roman, 12号
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'  # 普通字体
#rcParams['mathtext.it'] = 'Times New Roman:italic'  # 斜体字体
#rcParams['mathtext.bf'] = 'Times New Roman:bold'  # 粗体字体
title_font = {'size': 12, 'color': 'k', 'family': 'Times New Roman'}  # 标题字体

# 显示全部行列
torch.set_printoptions(threshold=float('inf'))

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(42)

# 加载数据 Epoch [200/200], 训练损失: 0.042, 验证损失: 0.093, 训练R2: 0.957, 验证R2: 0.913
data = pd.read_excel('Data/Data_22.xlsx')
print("数据集形状:", data.shape)
print("前5行数据:\n", data.head(5))

# 样本ID、cN、cA
ID = data.iloc[:, 0].values  # 样本ID
cN = data.iloc[:, 1].values  # 烤后氮素
cA = data.iloc[:, 2].values  # 烤后烟碱

# 提取特征
feature = data.iloc[:, 3:24].values
print("特征数据形状:", feature.shape)

# 重塑特征形状(样本数, 3, 7)
sample_num = 105  # 样本数
time_step = 3     # 时间步
feature_num = 7   # 特征数
feature_new = feature.reshape(sample_num, time_step, feature_num)
print("特征形状重塑为:", feature_new.shape)

# 标准化数据
def std_data(data):
    scaler = StandardScaler()
    data_reshaped = data.reshape(-1, feature_num)  # (样本数 * 时间步, 特征数)
    data_scaled_reshaped = scaler.fit_transform(data_reshaped)
    data_scaled = data_scaled_reshaped.reshape(sample_num, time_step, feature_num)
    return data_scaled, scaler

feature_scaled, scaler = std_data(feature_new)

# 划分训练集和验证集
X_train, X_val = train_test_split(feature_scaled, test_size=0.3, random_state=42)
print("训练集特征形状:", X_train.shape)
print("验证集特征形状:", X_val.shape)

# 转换为PyTorch张量
def to_tensor(data):
    return torch.from_numpy(data).float()

X_train_tensor = to_tensor(X_train)
X_val_tensor = to_tensor(X_val)

# 创建TensorDataset
batch_size = 20
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # 自编码器的目标是输入本身
val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义LSTM自编码器
class LSTM_AutoEncoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=28, latent_size=7):
        super(LSTM_AutoEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # 编码器
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.encoder_fc = nn.Linear(3 * hidden_size, 2 * latent_size)
        self.encoder_active = nn.LeakyReLU(0.1)
        # 解码器
        self.decoder_fc = nn.Linear(2 * latent_size, 3 * hidden_size)
        self.decoder_active = nn.LeakyReLU(0.1)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size = x.size(0)
        # 编码器
        h, _ = self.encoder_lstm(x)  # h: (batch_size, 3, hidden_size)
        h_flat = h.reshape(batch_size, -1)  # (batch_size, 3 * hidden_size)
        z = self.encoder_fc(h_flat)  # (batch_size, 2 * latent_size)
        z = self.encoder_active(z)  # 应用激活函数
        z_latent = z.reshape(batch_size, 2, self.latent_size)  # (batch_size, 2, latent_size)
        # 解码器
        z_flat = z.reshape(batch_size, -1)  # (batch_size, 2 * latent_size)
        h_decoder_input = self.decoder_fc(z_flat)  # (batch_size, 3 * hidden_size)
        h_decoder_input = self.decoder_active(h_decoder_input)  # 应用激活函数
        h_decoder_input = h_decoder_input.reshape(batch_size, 3, self.hidden_size)
        decoded_h, _ = self.decoder_lstm(h_decoder_input)
        x_reconstructed = self.output_layer(decoded_h)  # (batch_size, 3, input_size)
        return x_reconstructed, z_latent


# 初始化模型、损失函数和优化器
model = LSTM_AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试过程
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    for x, targets in loader:
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        all_targets.append(targets.detach().numpy())
        all_outputs.append(outputs.detach().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    epoch_r2 = r2_score(all_targets.reshape(-1), all_outputs.reshape(-1))
    return epoch_loss, epoch_r2

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for x, targets in loader:
            outputs, _ = model(x)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * x.size(0)
            all_targets.append(targets.numpy())
            all_outputs.append(outputs.numpy())
    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    epoch_r2 = r2_score(all_targets.reshape(-1), all_outputs.reshape(-1))
    return epoch_loss, epoch_r2

# 训练模型
num_epochs = 200
train_losses = []
val_losses = []
train_r2_scores = []
val_r2_scores = []

for epoch in range(num_epochs):
    train_loss, train_r2 = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_r2 = validate_epoch(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_r2_scores.append(train_r2)
    val_r2_scores.append(val_r2)
    # 每10个epoch打印一次进度
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], 训练损失: {train_loss:.2f}, 验证损失: {val_loss:.2f}, "
              f"训练R2: {train_r2:.2f}, 验证R2: {val_r2:.2f}")

# 绘制训练过程的Loss和R2曲线
fig, ax1 = plt.subplots(figsize=(6.0, 4.5), dpi=100)
# 调配颜色
color1 = np.array([22, 118, 177]) / 255  # 系统蓝
color2 = np.array([240, 135, 68]) / 255  # 淡橘色
epochs = range(1, num_epochs + 1)
ax1.set_xlim((0, 200))
ax1.set_ylim((0, 1.0))
ax1.set_xticks(np.arange(0, 201, step=40))
ax1.set_yticks(np.arange(0, 1.1, step=0.2))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.plot(epochs, train_losses, color=color1, linestyle='-')
ax1.plot(epochs, val_losses, color=color1, linestyle='--')
ax1.tick_params(axis='y')
ax1.grid(ls='--')

ax2 = ax1.twinx()  # 共享x轴
ax2.set_ylim((0, 1.0))
ax2.set_ylabel('$R^2$')
ax2.plot(epochs, train_r2_scores, color=color2, linestyle='-')
ax2.plot(epochs, val_r2_scores, color=color2, linestyle='--')
ax2.tick_params(axis='y')

plt.title('Loss (MSE) and $R^2$ curves of the AE$_{\\text{LSTM}}$', title_font, pad=10)
plt.plot([], [], '-', color=color1, label='Train loss (0.05)')
plt.plot([], [], '--', color=color1, label='Test loss (0.10)')
plt.plot([], [], '-', color=color2, label='Train $R^2$ (0.95)')
plt.plot([], [], '--', color=color2, label='Test $R^2$ (0.91)')
plt.legend(loc='center right', borderaxespad=1.0, prop={'size': 11})
plt.tight_layout()
plt.savefig('Figure/AE_LSTM.pdf', format='pdf', bbox_inches='tight')
plt.show()

# 模型评估，输出最终的MSE和R2
print(f"最终验证集重构误差 (MSE): {val_losses[-1]:.2f}")
print(f"最终验证集R²得分: {val_r2_scores[-1]:.2f}")

# 提取潜在表示
def extract_latent(model, data_scaled):
    model.eval()
    latent_vectors = []
    batch_size = 20
    num_samples = data_scaled.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            x_batch = torch.from_numpy(data_scaled[start_idx:end_idx]).float()
            _, z_latent = model(x_batch)
            latent_vectors.append(z_latent.numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    return latent_vectors  # (样本数, 2, 7)

# 提取潜在表示
latent_sequences = extract_latent(model, feature_scaled)
print("最终的潜在表示形状:", latent_sequences.shape)

# 导出潜在表示
def export_data(latent_sequences, filename, add_line=None):
    # 将潜在表示展平为 (样本数, 时间步, latent_size)
    latent_vectors_flat = latent_sequences.reshape(sample_num, -1)
    latent_size = latent_sequences.shape[2]
    feature_columns = [f'T{i}_F{j}' for i in range(1, 3) for j in range(1, latent_size + 1)]
    df_features = pd.DataFrame(latent_vectors_flat, columns=feature_columns)
    if add_line is not None:
        df_info = pd.DataFrame(add_line)
        df = pd.concat([df_info, df_features], axis=1)
    else:
        df = df_features

    # 导出到Excel
    df.to_excel(filename, index=False)
    print(f"数据已保存到 {filename}")

# 保留原始数据前三列
add_line = {
    'ID': ID,
    'cNitrogen(mg/g)': cN,
    'cAlkaloid(mg/g)': cA
}

# 导出文件
export_data(latent_sequences, 'Data/Latent_LSTM.xlsx', add_line=add_line)
