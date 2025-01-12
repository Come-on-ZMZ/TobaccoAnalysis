import pandas as pd
from sklearn.preprocessing import StandardScaler

# 文件列表
input_files = ['Latent_FCN.xlsx', 'Latent_CNN.xlsx', 'Latent_LSTM.xlsx', 'Data_23.xlsx','Data_T1.xlsx']
output_files = ['stdFCN.xlsx', 'stdConv1D.xlsx', 'stdLSTM.xlsx', 'std23.xlsx', 'stdTSR.xlsx']


def process(input_file, output_file, data_dir='Data'):
    # 读取Excel文件
    df = pd.read_excel(f'{data_dir}/{input_file}')

    # 提取标签和特征
    label = df.iloc[:, :3]
    features = df.iloc[:, 3:]

    # 标准化特征
    scaler = StandardScaler()
    features_std = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # 合并标签和标准化后的特征
    df_std = pd.concat([label, features_std], axis=1)

    # 导出Excel文件
    df_std.to_excel(f'{data_dir}/{output_file}', index=False)

# 批处理
for in_file, out_file in zip(input_files, output_files):
    process(in_file, out_file)